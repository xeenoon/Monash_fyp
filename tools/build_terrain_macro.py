#!/usr/bin/env python3
"""Build a water-free Alpine macro colour and grass/rock blend map.

Only pixels explicitly classified as rock or grass are allowed to contribute
colour. Water, snow, roads, and unclassified pixels are holes: normalized
convolution blurs across valid samples and the remaining holes are filled from
the nearest classified macro texel. RGB stores macro colour; alpha stores the
same cross-sampled grass weight used to blend the close-range PBR banks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "assets" / "terrain_albedo.png"
DEFAULT_OUTPUT = ROOT / "assets" / "terrain_macro.png"
DEFAULT_LABELS = ROOT / "assets" / "terrain_macro_labels.png"


def classify_materials(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mutually-exclusive grass, rock, and rejected masks.

    These intentionally conservative labels are the small version of the
    classifier in the texture-mapping worktree.  False negatives become holes;
    that is safer than allowing water or snow into a material palette.
    """
    rgb = rgb_u8.astype(np.float32) / 255.0
    red, green, blue = np.moveaxis(rgb, -1, 0)
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0e-5)
    luminance = rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)

    water = (
        (blue > red * 1.08)
        & (green > red * 1.04)
        & (blue > green * 1.035)
        & ((blue + green) > (red * 2.18))
        & (luminance > 0.08)
        & (luminance < 0.78)
    )
    snow = (luminance > 0.64) & (saturation < 0.28)
    grass = (
        (green > red * 1.035)
        & (green > blue * 1.10)
        & (luminance > 0.08)
        & (luminance < 0.64)
        & ~water
    )
    rock = (
        ~grass
        & ~water
        & ~snow
        & (saturation < 0.38)
        & (luminance > 0.07)
        & (luminance < 0.76)
    )
    return grass, rock, ~(grass | rock)


def _area_resize(values: np.ndarray, size: int) -> np.ndarray:
    channels = values.shape[2] if values.ndim == 3 else 1
    data = values if values.ndim == 3 else values[..., None]
    if channels == 1:
        image = Image.fromarray(data[..., 0].astype(np.float32), mode="F")
        result = np.asarray(image.resize((size, size), Image.Resampling.BOX), dtype=np.float32)
    else:
        result = np.stack(
            [np.asarray(Image.fromarray(data[..., channel], mode="F").resize(
                (size, size), Image.Resampling.BOX), dtype=np.float32)
             for channel in range(channels)], axis=-1
        )
    return result


def _filtered_class_colour(
    linear_rgb: np.ndarray, mask: np.ndarray, size: int, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    mass = _area_resize(mask.astype(np.float32), size)
    numerator = _area_resize(linear_rgb * mask[..., None], size)
    if sigma > 0.0:
        mass = ndimage.gaussian_filter(mass, sigma=sigma, mode="nearest")
        numerator = ndimage.gaussian_filter(
            numerator, sigma=(sigma, sigma, 0.0), mode="nearest"
        )
    colour = numerator / np.maximum(mass[..., None], 1.0e-6)
    return colour, mass


def _nearest_valid_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise ValueError("source contains no pixels classified as grass or rock")
    if valid.all():
        return values
    indices = ndimage.distance_transform_edt(~valid, return_distances=False,
                                             return_indices=True)
    result = values.copy()
    result[~valid] = values[indices[0][~valid], indices[1][~valid]]
    return result


def build_macro(
    source: Path, output: Path, size: int, blur_radius: float,
    labels_output: Path | None = None,
) -> None:
    if size < 2:
        raise ValueError("size must be at least 2 pixels")
    if blur_radius < 0.0:
        raise ValueError("blur radius cannot be negative")

    with Image.open(source) as image:
        rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    grass, rock, rejected = classify_materials(rgb_u8)

    # Filtering in linear light avoids the dark halos produced by averaging
    # sRGB values. Each material is normalized by its own accepted sample mass.
    srgb = rgb_u8.astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    grass_colour, grass_mass = _filtered_class_colour(
        linear, grass, size, blur_radius
    )
    rock_colour, rock_mass = _filtered_class_colour(
        linear, rock, size, blur_radius
    )
    total_mass = grass_mass + rock_mass
    grass_weight = grass_mass / np.maximum(total_mass, 1.0e-6)
    macro_linear = (
        grass_colour * grass_weight[..., None]
        + rock_colour * (1.0 - grass_weight[..., None])
    )
    valid = total_mass > 2.0e-3
    macro_linear = _nearest_valid_fill(macro_linear, valid)
    grass_weight = _nearest_valid_fill(grass_weight, valid)
    macro_srgb = np.where(macro_linear <= 0.0031308, macro_linear * 12.92,
                          1.055 * np.maximum(macro_linear, 0.0) ** (1.0 / 2.4) - 0.055)
    macro_u8 = np.clip(np.rint(macro_srgb * 255.0), 0, 255).astype(np.uint8)

    output.parent.mkdir(parents=True, exist_ok=True)
    grass_u8 = np.clip(np.rint(grass_weight * 255.0), 0, 255).astype(np.uint8)
    macro_rgba = np.dstack((macro_u8, grass_u8))
    Image.fromarray(macro_rgba, "RGBA").save(output, "PNG", optimize=False,
                                               compress_level=9)

    if labels_output is not None:
        raw_grass = _area_resize(grass.astype(np.float32), size)
        raw_rock = _area_resize(rock.astype(np.float32), size)
        raw_rejected = _area_resize(rejected.astype(np.float32), size)
        labels = np.stack((raw_rock, raw_grass, raw_rejected), axis=-1)
        labels = np.clip(np.rint(labels * 255.0), 0, 255).astype(np.uint8)
        labels_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels, "RGB").save(labels_output, "PNG", optimize=False,
                                               compress_level=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Alpine source image (default: assets/terrain_albedo.png)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output PNG (default: assets/terrain_macro.png)")
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS,
                        help="RGB diagnostic: rock, grass, rejected (default: assets/terrain_macro_labels.png)")
    parser.add_argument("--no-labels", action="store_true",
                        help="do not write the classification diagnostic")
    parser.add_argument("--size", type=int, default=64,
                        help="macro samples per side (default: 64)")
    parser.add_argument("--blur-radius", type=float, default=1.5,
                        help="normalized-convolution radius in macro texels (default: 1.5)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = None if args.no_labels else args.labels_output
    build_macro(args.source, args.output, args.size, args.blur_radius, labels)
    print(f"wrote {args.output} ({args.size}x{args.size}, blur={args.blur_radius:g}; "
          "RGB=macro colour, A=grass weight)")
    if labels is not None:
        print(f"wrote {labels} (R=rock, G=grass, B=rejected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
