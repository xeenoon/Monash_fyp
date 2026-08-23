#!/usr/bin/env python3
"""Plan or download a 2 m, remote-natural Swiss-Alps source cache.

This is deliberately a *source acquisition* tool.  The Vulkan runtime only
loads the .trn pyramid produced by terrain_tiles.py; it cannot read GeoTIFFs.
The source cache created here is therefore an input to a later GDAL mosaic/
warp step, not a renderer dataset.

The selection favours high, rugged Alpine cells.  It rejects tiles whose whole
2 m terrain raster is not remote alpine terrain, which reliably removes towns
and almost all public roads.  It is not a substitute for a vector road/building
mask: do that before publishing imagery as a final natural-only texture set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat

Image.MAX_IMAGE_PIXELS = None

STAC_ROOT = "https://data.geo.admin.ch/api/stac/v1"
ALTI = "ch.swisstopo.swissalti3d"
IMAGE = "ch.swisstopo.swissimage-dop10"
GIB = 1024 ** 3

# WGS84 regions cover the Swiss high-Alps spine.  They intentionally omit the
# Plateau, Jura, and low Alpine valley floors, where settlements and roads are
# common.  The DEM screen below is the final admission test.
REGIONS = {
    "valais": (6.80, 45.80, 8.25, 46.45),
    "bern_uri": (7.45, 46.35, 9.00, 46.95),
    "ticino": (8.20, 45.70, 9.20, 46.45),
    "graubuenden": (8.75, 46.45, 10.65, 47.10),
}
TILE_ID = re.compile(r"_(\d{4}-\d{4})$")


@dataclass(frozen=True)
class Pair:
    tile_id: str
    dem_item: dict[str, Any]
    image_item: dict[str, Any]
    dem_asset: dict[str, Any]
    image_asset: dict[str, Any]


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "terrain-gen/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def stac_items(collection: str, bbox: tuple[float, float, float, float]) -> Iterable[dict[str, Any]]:
    query = urllib.parse.urlencode({"bbox": ",".join(map(str, bbox)), "limit": 500})
    url = f"{STAC_ROOT}/collections/{collection}/items?{query}"
    while url:
        page = request_json(url)
        yield from page.get("features", [])
        url = next((link["href"] for link in page.get("links", [])
                    if link.get("rel") == "next"), "")


def item_tile_id(item: dict[str, Any]) -> str | None:
    match = TILE_ID.search(item.get("id", ""))
    return match.group(1) if match else None


def newest_by_tile(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        tile_id = item_tile_id(item)
        if not tile_id:
            continue
        date = item.get("properties", {}).get("datetime", "")
        old = result.get(tile_id)
        if old is None or date > old.get("properties", {}).get("datetime", ""):
            result[tile_id] = item
    return result


def geotiff_asset(item: dict[str, Any], gsd: float) -> dict[str, Any] | None:
    assets = item.get("assets", {}).values()
    matches = [asset for asset in assets
               if asset.get("gsd") == gsd and asset.get("href", "").endswith(".tif")]
    return matches[0] if len(matches) == 1 else None


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "terrain-gen/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError(f"server supplied no Content-Length: {url}")
    return int(length)


def fetch_with_curl(url: str, destination: Path) -> None:
    """Resume into a sidecar file, then atomically expose a completed asset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run([
        "curl", "--fail", "--location", "--continue-at", "-", "--retry", "4",
        "--retry-all-errors", "--show-error", "--output", str(partial), url,
    ], check=True)
    partial.replace(destination)


def terrain_metrics(dem_path: Path) -> tuple[float, float, float]:
    """Return minimum, median, and relief in metres for a small 2 m DEM tile."""
    with Image.open(dem_path) as image:
        image = image.convert("F")
        # 500x500 is cheap, but this prevents a future higher-res asset choice
        # from consuming excessive RAM merely to screen it.
        image.thumbnail((256, 256), Image.Resampling.BOX)
        values = sorted(float(value) for value in image.getdata() if value == value)
    if not values:
        raise RuntimeError(f"no valid elevation samples in {dem_path}")
    return values[0], values[len(values) // 2], values[-1] - values[0]


def natural_remote(minimum: float, median: float, relief: float,
                   minimum_height: float, minimum_relief: float) -> bool:
    # Requiring the *lowest* point to be high is purposeful: a tile containing a
    # populated valley, road pass, or resort is not admitted just because a peak
    # rises from it.  This is conservative and leaves a natural-only corpus.
    return minimum >= minimum_height and median >= minimum_height + 150.0 and \
        relief >= minimum_relief


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def local_name(prefix: str, item: dict[str, Any], asset: dict[str, Any]) -> str:
    suffix = Path(urllib.parse.urlparse(asset["href"]).path).suffix
    return f"{prefix}_{item['id']}{suffix}"


def discover_pairs(regions: list[str]) -> list[Pair]:
    dem_items: dict[str, dict[str, Any]] = {}
    image_items: dict[str, dict[str, Any]] = {}
    for region in regions:
        bbox = REGIONS[region]
        dem_items.update(newest_by_tile(stac_items(ALTI, bbox)))
        image_items.update(newest_by_tile(stac_items(IMAGE, bbox)))
    pairs: list[Pair] = []
    for tile_id in sorted(dem_items.keys() & image_items.keys()):
        dem_item, image_item = dem_items[tile_id], image_items[tile_id]
        dem_asset = geotiff_asset(dem_item, 2.0)
        image_asset = geotiff_asset(image_item, 2.0)
        if dem_asset and image_asset:
            pairs.append(Pair(tile_id, dem_item, image_item, dem_asset, image_asset))
    # Spread the fixed byte budget over all Alpine regions instead of consuming
    # the western-most lexicographic run of tiles first.  The rank is stable, so
    # interrupted runs and future re-plans choose the same corpus.
    return sorted(pairs, key=lambda pair: hashlib.sha256(
        pair.tile_id.encode("ascii")).digest())


def plan(args: argparse.Namespace) -> int:
    pairs = discover_pairs(args.regions)
    selected: list[dict[str, Any]] = []
    total = 0
    for pair in pairs:
        dem_bytes = content_length(pair.dem_asset["href"])
        image_bytes = content_length(pair.image_asset["href"])
        if total + dem_bytes + image_bytes > args.budget_gib * GIB:
            continue
        total += dem_bytes + image_bytes
        selected.append({
            "tile_id": pair.tile_id,
            "dem_item": pair.dem_item["id"], "image_item": pair.image_item["id"],
            "dem_url": pair.dem_asset["href"], "image_url": pair.image_asset["href"],
            "dem_bytes": dem_bytes, "image_bytes": image_bytes,
        })
    write_json(args.output / "download-plan.json", {
        "purpose": "2 m remote-natural Swiss Alps source cache; imagery remains unmasked",
        "budget_bytes": args.budget_gib * GIB,
        "planned_bytes": total,
        "regions": args.regions,
        "screen": {"minimum_height_m": args.minimum_height,
                   "minimum_relief_m": args.minimum_relief},
        "tiles": selected,
    })
    print(f"planned {len(selected)} tile pairs, {total / GIB:.2f} GiB maximum")
    print(f"plan written to {args.output / 'download-plan.json'}")
    return 0


def download(args: argparse.Namespace) -> int:
    if shutil.which("curl") is None:
        raise RuntimeError("download mode requires curl for reliable resume support")
    output = args.output
    state_path = output / "manifest.json"
    state = {"accepted": [], "rejected": [], "bytes": 0}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    accepted_ids = {entry["tile_id"] for entry in state["accepted"]}
    rejected_ids = {entry["tile_id"] for entry in state["rejected"]}
    for pair in discover_pairs(args.regions):
        if pair.tile_id in accepted_ids or pair.tile_id in rejected_ids:
            continue
        dem_name = local_name("dem", pair.dem_item, pair.dem_asset)
        dem_path = output / "dem" / dem_name
        fetch_with_curl(pair.dem_asset["href"], dem_path)
        minimum, median, relief = terrain_metrics(dem_path)
        if not natural_remote(minimum, median, relief, args.minimum_height, args.minimum_relief):
            dem_path.unlink()
            state["rejected"].append({"tile_id": pair.tile_id, "minimum_m": minimum,
                                      "median_m": median, "relief_m": relief})
            write_json(state_path, state)
            continue
        image_size = content_length(pair.image_asset["href"])
        if state["bytes"] + dem_path.stat().st_size + image_size > args.budget_gib * GIB:
            break
        image_name = local_name("image", pair.image_item, pair.image_asset)
        image_path = output / "imagery" / image_name
        fetch_with_curl(pair.image_asset["href"], image_path)
        write_json(output / "metadata" / f"{pair.tile_id}.dem.json", pair.dem_item)
        write_json(output / "metadata" / f"{pair.tile_id}.image.json", pair.image_item)
        state["bytes"] += dem_path.stat().st_size + image_path.stat().st_size
        state["accepted"].append({
            "tile_id": pair.tile_id, "dem": str(dem_path.relative_to(output)),
            "imagery": str(image_path.relative_to(output)), "minimum_m": minimum,
            "median_m": median, "relief_m": relief,
        })
        write_json(state_path, state)
        print(f"accepted {pair.tile_id}: {state['bytes'] / GIB:.2f} GiB")
        time.sleep(args.request_delay)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("plan", "download"))
    result.add_argument("--output", type=Path, default=Path("alps-data/large-2m"))
    result.add_argument("--budget-gib", type=float, default=20.0)
    result.add_argument("--regions", nargs="+", choices=sorted(REGIONS),
                        default=sorted(REGIONS))
    result.add_argument("--minimum-height", type=float, default=1800.0)
    result.add_argument("--minimum-relief", type=float, default=180.0)
    result.add_argument("--request-delay", type=float, default=0.1)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.budget_gib <= 0:
        raise ValueError("--budget-gib must be positive")
    return plan(args) if args.mode == "plan" else download(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, urllib.error.URLError, ValueError) as error:
        print(f"download_swiss_alps: {error}", file=sys.stderr)
        raise SystemExit(1)
