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
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

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


def open_with_retry(request: urllib.request.Request, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=45)
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(
                    f"request failed after {attempts} attempts: {request.full_url}") from error
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "terrain-gen/1.0"})
    with open_with_retry(request) as response:
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
    with open_with_retry(request) as response:
        length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError(f"server supplied no Content-Length: {url}")
    return int(length)


def fetch_many_with_curl(downloads: list[tuple[str, Path]], workers: int) -> None:
    """Fetch a batch with connection reuse, exposing files only on success."""
    pending: list[tuple[Path, Path]] = []
    command = [
        "curl", "--fail", "--location", "--retry", "4", "--retry-all-errors",
        "--silent", "--show-error", "--parallel", "--parallel-immediate",
        "--parallel-max", str(workers),
    ]
    for url, destination in downloads:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        command += ["--url", url, "--output", str(partial), "--continue-at", "-"]
        pending.append((partial, destination))
    if not pending:
        return
    subprocess.run(command, check=True)
    for partial, destination in pending:
        partial.replace(destination)


def terrain_metrics(dem_path: Path) -> tuple[float, float, float]:
    """Return exact minimum, median, and relief for a bounded-size DEM tile."""
    with Image.open(dem_path) as image:
        image = image.convert("F")
        # The admission rule is explicitly based on the lowest pixel. A BOX
        # thumbnail smooths local lows and can therefore admit a tile that does
        # not satisfy the rule. Current 1 km / 2 m swissALTI3D assets are only
        # 500x500, so inspect every sample and fail closed if a future product
        # silently changes to an unexpectedly large raster.
        if image.width * image.height > 4_000_000:
            raise RuntimeError(
                f"DEM is too large for exact screening ({image.width}x{image.height}): "
                f"{dem_path}")
        get_data = getattr(image, "get_flattened_data", image.getdata)
        values = sorted(float(value) for value in get_data()
                        if math.isfinite(float(value)))
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
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def local_name(prefix: str, item: dict[str, Any], asset: dict[str, Any]) -> str:
    suffix = Path(urllib.parse.urlparse(asset["href"]).path).suffix
    return f"{prefix}_{item['id']}{suffix}"


def selection_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "regions": sorted(args.regions),
        "minimum_height_m": args.minimum_height,
        "minimum_relief_m": args.minimum_relief,
        "dem_collection": ALTI,
        "imagery_collection": IMAGE,
        "gsd_m": 2.0,
    }


def validate_resume_state(output: Path, state: dict[str, Any],
                          expected_selection: dict[str, Any], budget: int) -> None:
    if state.get("selection") != expected_selection:
        raise RuntimeError(
            "manifest selection settings do not match this invocation; "
            "use the original settings or a new output directory")
    accepted = state.get("accepted")
    rejected = state.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise RuntimeError("manifest accepted/rejected fields must be arrays")
    accepted_ids = [entry.get("tile_id") for entry in accepted
                    if isinstance(entry, dict)]
    rejected_ids = [entry.get("tile_id") for entry in rejected
                    if isinstance(entry, dict)]
    if len(accepted_ids) != len(accepted) or len(rejected_ids) != len(rejected) or any(
            not isinstance(tile_id, str) or not tile_id
            for tile_id in accepted_ids + rejected_ids):
        raise RuntimeError("manifest contains an invalid tile entry")
    if len(set(accepted_ids)) != len(accepted_ids) or len(set(rejected_ids)) != len(
            rejected_ids) or set(accepted_ids) & set(rejected_ids):
        raise RuntimeError("manifest contains duplicate or conflicting tile IDs")
    actual_bytes = 0
    root = output.resolve()
    for entry in accepted:
        for field in ("dem", "imagery"):
            relative = entry.get(field)
            if not isinstance(relative, str):
                raise RuntimeError(f"manifest tile {entry['tile_id']} has no {field} path")
            relative_path = Path(relative)
            path = (output / relative_path).resolve()
            if relative_path.is_absolute() or not path.is_relative_to(root):
                raise RuntimeError(f"manifest contains unsafe asset path: {relative}")
            if not path.is_file():
                raise RuntimeError(f"manifest asset does not exist: {path}")
            actual_bytes += path.stat().st_size
    if state.get("bytes") != actual_bytes:
        raise RuntimeError(
            f"manifest byte count is {state.get('bytes')}, actual assets total {actual_bytes}")
    if actual_bytes > budget:
        raise RuntimeError("existing accepted assets exceed --budget-gib")


def discover_pairs(regions: list[str], workers: int = 8) -> list[Pair]:
    dem_items: dict[str, dict[str, Any]] = {}
    image_items: dict[str, dict[str, Any]] = {}
    requests = [(collection, region) for collection in (ALTI, IMAGE)
                for region in regions]

    def discover(request: tuple[str, str]) -> dict[str, dict[str, Any]]:
        collection, region = request
        return newest_by_tile(stac_items(collection, REGIONS[region]))

    # The STAC service caps pages at 100 items even when a larger limit is
    # requested. Alpine discovery therefore involves many independent metadata
    # requests; bounded region/collection concurrency avoids a multi-minute
    # serial prelude without changing selection order.
    with ThreadPoolExecutor(max_workers=min(workers, len(requests))) as executor:
        discovered = executor.map(discover, requests)
        for (collection, _), items in zip(requests, discovered):
            (dem_items if collection == ALTI else image_items).update(items)
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


def cached_pairs(output: Path, regions: list[str], workers: int,
                 refresh: bool = False) -> list[Pair]:
    path = output / "discovery.json"
    expected = {
        "regions": sorted(regions),
        "dem_collection": ALTI,
        "imagery_collection": IMAGE,
        "gsd_m": 2.0,
    }
    if path.is_file() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("selection") != expected or not isinstance(cached.get("pairs"), list):
            raise RuntimeError(
                f"{path}: cached discovery settings do not match this invocation")
        pairs = []
        for record in cached["pairs"]:
            dem_item = record.get("dem_item")
            image_item = record.get("image_item")
            if not isinstance(dem_item, dict) or not isinstance(image_item, dict):
                raise RuntimeError(f"{path}: invalid cached item")
            tile_id = record.get("tile_id")
            dem_asset = geotiff_asset(dem_item, 2.0)
            image_asset = geotiff_asset(image_item, 2.0)
            if tile_id != item_tile_id(dem_item) or tile_id != item_tile_id(image_item) or not (
                    dem_asset and image_asset):
                raise RuntimeError(f"{path}: invalid cached pair {tile_id!r}")
            pairs.append(Pair(tile_id, dem_item, image_item, dem_asset, image_asset))
        print(f"loaded {len(pairs)} candidate pairs from {path}", flush=True)
        return pairs

    pairs = discover_pairs(regions, workers)
    write_json(path, {
        "selection": expected,
        "pairs": [{"tile_id": pair.tile_id, "dem_item": pair.dem_item,
                   "image_item": pair.image_item} for pair in pairs],
    })
    print(f"cached {len(pairs)} candidate pairs in {path}", flush=True)
    return pairs


def plan(args: argparse.Namespace) -> int:
    pairs = cached_pairs(args.output, args.regions, args.metadata_workers,
                         args.refresh_discovery)
    selected: list[dict[str, Any]] = []
    total = 0
    def sizes(pair: Pair) -> tuple[int, int]:
        return (content_length(pair.dem_asset["href"]),
                content_length(pair.image_asset["href"]))

    with ThreadPoolExecutor(max_workers=args.metadata_workers) as executor:
        sized_pairs = zip(pairs, executor.map(sizes, pairs))
        for pair, (dem_bytes, image_bytes) in sized_pairs:
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
    selection = selection_config(args)
    state = {"accepted": [], "rejected": [], "bytes": 0,
             "selection": selection}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_resume_state(output, state, selection, int(args.budget_gib * GIB))
    accepted_ids = {entry["tile_id"] for entry in state["accepted"]}
    rejected_ids = {entry["tile_id"] for entry in state["rejected"]}
    pairs = [pair for pair in cached_pairs(
        args.output, args.regions, args.metadata_workers, args.refresh_discovery)
             if pair.tile_id not in accepted_ids and pair.tile_id not in rejected_ids]

    def dem_path_for(pair: Pair) -> Path:
        dem_name = local_name("dem", pair.dem_item, pair.dem_asset)
        return output / "dem" / dem_name

    def screen(pair: Pair) -> tuple[Pair, Path, float, float, float, int | None]:
        dem_path = dem_path_for(pair)
        minimum, median, relief = terrain_metrics(dem_path)
        image_size = None
        if natural_remote(minimum, median, relief,
                          args.minimum_height, args.minimum_relief):
            image_size = content_length(pair.image_asset["href"])
        if args.request_delay:
            time.sleep(args.request_delay)
        return pair, dem_path, minimum, median, relief, image_size

    def image_path_for(pair: Pair) -> Path:
        image_name = local_name("image", pair.image_item, pair.image_asset)
        return output / "imagery" / image_name

    budget = args.budget_gib * GIB
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        batch_size = args.download_workers * 4
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            fetch_many_with_curl([
                (pair.dem_asset["href"], dem_path_for(pair)) for pair in batch
            ], args.download_workers)
            screened = list(executor.map(screen, batch))
            selected: list[tuple[Pair, Path, float, float, float, int]] = []
            exhausted = False
            state_changed = False
            projected_bytes = state["bytes"]
            for pair, dem_path, minimum, median, relief, image_size in screened:
                if image_size is None:
                    dem_path.unlink()
                    state["rejected"].append({
                        "tile_id": pair.tile_id, "minimum_m": minimum,
                        "median_m": median, "relief_m": relief,
                    })
                    state_changed = True
                    continue
                pair_bytes = dem_path.stat().st_size + image_size
                if projected_bytes + pair_bytes > budget:
                    dem_path.unlink()
                    exhausted = True
                    continue
                projected_bytes += pair_bytes
                selected.append((pair, dem_path, minimum, median, relief, image_size))

            image_paths = [image_path_for(item[0]) for item in selected]
            fetch_many_with_curl([
                (item[0].image_asset["href"], image_path)
                for item, image_path in zip(selected, image_paths)
            ], args.download_workers)
            for item, image_path in zip(selected, image_paths):
                pair, dem_path, minimum, median, relief, _ = item
                write_json(output / "metadata" / f"{pair.tile_id}.dem.json", pair.dem_item)
                write_json(output / "metadata" / f"{pair.tile_id}.image.json", pair.image_item)
                state["bytes"] += dem_path.stat().st_size + image_path.stat().st_size
                state["accepted"].append({
                    "tile_id": pair.tile_id, "dem": str(dem_path.relative_to(output)),
                    "imagery": str(image_path.relative_to(output)), "minimum_m": minimum,
                    "median_m": median, "relief_m": relief,
                })
                state_changed = True
                print(f"accepted {pair.tile_id}: {state['bytes'] / GIB:.2f} GiB", flush=True)
            if state_changed:
                write_json(state_path, state)
            if exhausted:
                break
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
    result.add_argument("--metadata-workers", type=int, default=8,
                        help="concurrent STAC/HEAD metadata requests (default: 8)")
    result.add_argument("--download-workers", type=int, default=4,
                        help="concurrent DEM screening/downloads (default: 4)")
    result.add_argument("--refresh-discovery", action="store_true",
                        help="discard the cached STAC candidate catalogue")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.budget_gib <= 0:
        raise ValueError("--budget-gib must be positive")
    if not 1 <= args.metadata_workers <= 32:
        raise ValueError("--metadata-workers must be between 1 and 32")
    if not 1 <= args.download_workers <= 16:
        raise ValueError("--download-workers must be between 1 and 16")
    return plan(args) if args.mode == "plan" else download(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, urllib.error.URLError, ValueError) as error:
        print(f"download_swiss_alps: {error}", file=sys.stderr)
        raise SystemExit(1)
