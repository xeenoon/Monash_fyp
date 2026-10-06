#!/usr/bin/env python3
"""Find roads and buildings in raw orthophotos by colour and local form.

No vector data. Works at 0.5 m (SWISSIMAGE 10 cm, box-downsampled 5x).

The signal is GREEN DEFICIT: how much less green a pixel is than its
neighbourhood (excess green 2G-R-B against its 20 m mean). Every road, roof,
car park and rail line in the Alps is a neutral surface cut into pasture or
forest, so it is a sharp local drop in green; bare rock, scree and ice have
no green around them to drop from, so they never trigger. That makes the
detector independent of lighting, of the sensor, and of how bright the
surface itself is (white asphalt and dark slate both qualify).

Candidates are then sorted by form:

  roads      ribbons of near-constant width (the spread of the half-width
             along the medial ridge is small; natural pale bands vary)
  buildings  thick, smooth, box-filling blobs, found after an opening that
             erases anything as narrow as a road -- so a roof is separated
             from the track that runs up to it

The mask is dilated to take verges, walls and cast shadows with it.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes, binary_opening,
                           distance_transform_edt, find_objects, label, maximum_filter,
                           uniform_filter)

LUMA = np.array((0.299, 0.587, 0.114), np.float32)


def disk(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def local_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = uniform_filter(values, window)
    return np.sqrt(np.maximum(uniform_filter(values * values, window) - mean * mean, 0.0))


def component_form(mask: np.ndarray):
    """Per connected component: labels, count, area, mean width, implied length."""
    labels, count = label(mask, structure=np.ones((3, 3)))
    if not count:
        return labels, 0, np.zeros(1), np.zeros(1), np.zeros(1)
    area = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float32)
    edt = distance_transform_edt(mask)
    width = 2.0 * np.bincount(labels.ravel(), weights=edt.ravel(), minlength=count + 1) / np.maximum(area, 1)
    return labels, count, area, width, area / np.maximum(width, 1.0)


def component_mean(labels: np.ndarray, count: int, values: np.ndarray) -> np.ndarray:
    sums = np.bincount(labels.ravel(), weights=values.ravel(), minlength=count + 1)
    return sums / np.maximum(np.bincount(labels.ravel(), minlength=count + 1), 1)


def rectangularity(labels: np.ndarray, count: int, area: np.ndarray) -> np.ndarray:
    """Area over the area of the component's ORIENTED bounding box (principal
    axes): a roof fills it whatever way the building faces; a tree with its
    shadow or a ragged outcrop does not."""
    out = np.zeros(count + 1, np.float32)
    for i, box in enumerate(find_objects(labels), start=1):
        if box is None:
            continue
        ys, xs = np.nonzero(labels[box] == i)
        if len(ys) < 3:
            continue
        points = np.stack([ys, xs], 1).astype(np.float32)
        points -= points.mean(0)
        _, vectors = np.linalg.eigh(points.T @ points)
        projected = points @ vectors
        extent = (np.ptp(projected[:, 0]) + 1.0) * (np.ptp(projected[:, 1]) + 1.0)
        out[i] = area[i] / max(extent, 1.0)
    return out


def width_spread(mask: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """Coefficient of variation of the half-width along each component's
    medial ridge: ~0.2 for a road, 0.3-0.6 for a natural band."""
    edt = distance_transform_edt(mask)
    ridge = (edt > 0) & (edt >= maximum_filter(edt, 3))
    lab, w = labels[ridge], edt[ridge]
    n = np.maximum(np.bincount(lab, minlength=count + 1), 1)
    mean = np.bincount(lab, weights=w, minlength=count + 1) / n
    var = np.bincount(lab, weights=w * w, minlength=count + 1) / n - mean * mean
    return np.sqrt(np.maximum(var, 0.0)) / np.maximum(mean, 1e-6)


def infrastructure_mask(rgb: np.ndarray, *, metres_per_pixel: float = 0.5,
                        dilate_m: float = 1.5, return_layers: bool = False) -> dict[str, np.ndarray]:
    px = 0.5 / metres_per_pixel  # every threshold below is written for 0.5 m
    f = rgb.astype(np.float32) / 255.0
    lum = f @ LUMA
    chroma = f.max(axis=2) - f.min(axis=2)
    exg = 2.0 * f[..., 1] - f[..., 0] - f[..., 2]
    window = max(3, int(41 * px) | 1)  # 20 m
    wide = max(3, int(121 * px) | 1)   # 60 m
    wider = max(3, int(241 * px) | 1)  # 120 m
    # The greenest of three scales: inside a station complex nothing within
    # 20 or even 60 m is green, but the pasture beyond still is.
    green_around = np.maximum.reduce([uniform_filter(exg, w, mode="nearest")
                                      for w in (window, wide, wider)])
    lum_around = uniform_filter(lum, window, mode="nearest")
    deficit = green_around - exg

    # Neutral against green surroundings, and not a hole of deep shadow.
    neutral_cut = (deficit > 0.07) & (exg < 0.05) & (green_around > 0.05) & (lum > 0.16)
    # Bright or level surfaces: roads, pale roofs, yards, platforms. Very
    # bright near-neutral surfaces (concrete galleries, white roofs) qualify
    # even with a little green cast.
    light = (neutral_cut & (lum > lum_around - 0.02)) | \
        ((lum > lum_around + 0.12) & (chroma < 0.12) & (exg < 0.08) & (green_around > 0.05))
    # Darker than surroundings yet still grey, not black: slate and metal roofs.
    dim = neutral_cut & (chroma < 0.07) & (lum <= lum_around - 0.02)
    light = binary_opening(binary_closing(light, iterations=1), iterations=1)
    dim = binary_opening(binary_closing(dim, iterations=1), iterations=1)
    # Red and brown roofs (tile, painted metal, weathered timber): warmer than
    # anything natural here (R well above G, no excess green) and smooth.
    # Roof tiles are textured at 0.5 m, so no per-pixel smoothness test here;
    # the colour itself is rare enough in alpine terrain.
    warm_roof = ((f[..., 0] - f[..., 1]) > 0.06) & (exg < -0.03) & (lum > 0.10)
    warm_roof = binary_opening(binary_closing(warm_roof, iterations=1), iterations=1)

    # --- Roads: long ribbons of near-constant width --------------------------
    # Trees overhanging a road cut it into pieces; length is judged on a
    # bridged copy so the pieces count as the one road they are.
    bridged = binary_closing(light, structure=disk(max(1, int(round(3 * px)))))
    # Measure roads without the yards and roofs they lead to: those are thick
    # and would spoil the constant-width test for the whole network.
    bridged &= ~binary_opening(bridged, structure=disk(max(1, int(round(7 * px)))))
    labels, count, area, width, length = component_form(bridged)
    road = np.zeros_like(light)
    if count:
        # Roads form a network: either long, or running on past the tile
        # edge. Pale limestone bands are short and isolated, so a 40-140 m
        # ribbon in the middle of the tile is not taken as a road.
        border = np.zeros_like(bridged)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        at_edge = np.bincount(labels[border & bridged], minlength=count + 1) > 0
        network = (length >= 300 * px) | ((length >= 100 * px) & at_edge)
        spread = width_spread(bridged, labels, count)
        # A long network may carry junctions and lay-bys; give it more slack.
        even_width = (spread <= 0.34) | ((length >= 600 * px) & (spread <= 0.45))
        keep = network & (width >= 2 * px) & (width <= 30 * px) & even_width
        keep[0] = False
        road = keep[labels] & bridged

    # --- Faint tracks: narrow, low-contrast paths ------------------------------
    # Dirt tracks and footpaths are only slightly less green than the
    # pasture. Accept a weaker colour cut, but only as long, narrow,
    # even-width ribbons that join the road network or run off the tile.
    faint = (deficit > 0.035) & (exg < 0.07) & (green_around > 0.06) & (lum > lum_around - 0.01) & \
        (lum > 0.16) & ~light
    faint = binary_opening(binary_closing(faint | light, structure=disk(max(1, int(round(2 * px))))),
                           iterations=1)
    labels, count, area, width, length = component_form(faint)
    if count:
        border = np.zeros_like(faint)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        joins = np.bincount(labels[(binary_dilation(road, iterations=2) | border) & faint],
                            minlength=count + 1) > 0
        keep = joins & (length >= 160 * px) & (width >= 1.5 * px) & (width <= 9 * px) & \
            (width_spread(faint, labels, count) <= 0.36)
        keep[0] = False
        road |= keep[labels]

    # --- Buildings: thick, smooth, box-filling blobs --------------------------
    roofish = binary_closing(light | dim | warm_roof, iterations=2)
    thick = binary_opening(roofish, structure=disk(max(1, int(round(4 * px)))))
    building = np.zeros_like(light)
    labels, count, area, width, length = component_form(thick)
    if count:
        interior = thick & ~binary_dilation(~thick, iterations=1)
        texture = component_mean(labels, count, np.where(interior, local_std(lum, 3), 0.0)) / \
            np.maximum(component_mean(labels, count, interior.astype(np.float32)), 1e-6)
        # Shape and texture do not separate a hut from a pale limestone block
        # (both smooth and box-like at 0.5 m). Access does: buildings are
        # served by a road or track, measured 15-60 m away for every hut and
        # station on the calibration tiles, while loose limestone mostly lies
        # 70 m+ from one. A stray rock removed is harmless; a house left in
        # is not, so the test leans inclusive.
        redness = component_mean(labels, count, f[..., 0] - f[..., 1])
        texture = np.where(redness > 0.06, np.minimum(texture, 0.05), texture)  # tiled roofs
        road_distance = distance_transform_edt(~road) * metres_per_pixel
        nearest = np.full(count + 1, np.inf)
        np.minimum.at(nearest, labels.ravel(), road_distance.ravel())
        # A clearly red, box-shaped roof needs no road to vouch for it:
        # nothing natural here is that colour AND that shape.
        served = (nearest <= 60.0) | ((redness > 0.08) & (rectangularity(labels, count, area) > 0.6))
        keep = (area >= 160 * px * px) & (area <= 40000 * px * px) & (texture < 0.055) & \
            (rectangularity(labels, count, area) > 0.5) & served
        # Built-up ground (yards, platforms, car parks) right beside a road.
        # Built ground is laid out: either it fills its box (yard, platform)
        # or it is a constant-width strip (rail gallery). A limestone ridge
        # beside a path is neither.
        laid_out = (rectangularity(labels, count, area) > 0.35) | \
            (width_spread(thick, labels, count) <= 0.30)
        keep |= (area >= 1200 * px * px) & (texture < 0.04) & (nearest <= 15.0) & laid_out
        keep[0] = False
        building = binary_fill_holes(keep[labels])

    # --- Built-up areas: stations, villages, depots --------------------------
    # In a big complex the buildings, yards, tracks and car parks are each
    # odd shapes, but together they cover most of the ground and leave no
    # green. Take any area that dense in man-made candidates whole.
    candidates = (neutral_cut | light | dim | warm_roof | road | building).astype(np.float32)
    density = uniform_filter(candidates, max(3, int(61 * px) | 1), mode="nearest")
    local_green = uniform_filter(exg, max(3, int(41 * px) | 1), mode="nearest")
    built = (density > 0.30) & (local_green < 0.06)
    built = binary_opening(built, structure=disk(max(1, int(round(6 * px)))))
    labels, count, area, width, length = component_form(built)
    if count:
        # A built-up area must contain a detected building. A field of pale
        # limestone is just as dense and as bare -- and may well have a road
        # running past it -- but no roof stands in it.
        anchors = binary_dilation(building, iterations=max(1, int(round(4 * px))))
        anchored = np.bincount(labels[anchors & built], minlength=count + 1) >= 40 * px * px
        keep = (area >= 2000 * px * px) & anchored  # 500 m2: a station, not a road
        keep[0] = False
        built = binary_fill_holes(keep[labels])
        # Grow to the candidates it is made of, so its edge buildings come too.
        built = binary_dilation(built, iterations=max(1, int(round(4 * px)))) & \
            binary_dilation(candidates > 0, iterations=1) | built
        building |= built

    # --- Water: blue-dominant, smooth, mid-dark, and sizeable ---------------
    # Lakes read blue over grey (B-R ~ +0.07); grass, rock and gravel are
    # warm or neutral. Cliff shadow is blue too but darker and textured;
    # glacier ice is blue but bright. Beaches are the bare, non-green ring
    # around the water, taken up to ~8 m out.
    blue = f[..., 2] - f[..., 0]
    # Measured: lake B-R 0.065-0.085 with fine texture ~0.01; blue-grey
    # limestone B-R <= 0.04 and texture >= 0.04.
    water_px = (blue > 0.055) & (f[..., 2] >= f[..., 1] + 0.02) & (lum > 0.11) & \
        (lum < 0.42) & (local_std(lum, 5) < 0.022)
    water_px = binary_opening(binary_closing(water_px, iterations=2), iterations=2)
    labels, count, area, width, length = component_form(water_px)
    water = np.zeros_like(light)
    if count:
        # A water body is flat and blue THROUGHOUT; a shaded rock face only
        # in patches. Judge the whole component, not each pixel.
        flatness = component_mean(labels, count, local_std(lum, 5))
        blueness = component_mean(labels, count, blue)
        keep = (area >= 300 * px * px) & (width >= 6 * px) & (flatness < 0.013) & \
            (blueness > 0.065)
        keep[0] = False
        water = binary_fill_holes(keep[labels])
    beach = np.zeros_like(water)
    if water.any():
        # Shore, embankment and dam: up to ~15 m of bare ground round the water.
        reach = binary_dilation(water, iterations=max(1, int(round(30 * px))))
        # Bare shore, or the shallow teal margin (bluer than land).
        bare = (exg < 0.04) | (blue > 0.02)
        beach = reach & bare & ~water
        # Only shore that is connected to the water, not bare ground beyond.
        lab, n = label(beach | water)
        touching = np.unique(lab[water])
        beach &= np.isin(lab, touching[touching > 0])

    core = road | building | water | beach
    # Buffer: removal has to take the verge, the wall and the shadow with the
    # feature, or a dark outline of it survives into the fill. Roads take a
    # few metres; buildings more, and more again down-sun where their shadow
    # falls (the shadow side is found from the image itself: the darker side
    # just outside each building).
    steps = max(1, int(round(dilate_m / metres_per_pixel)))
    mask = binary_dilation(core, iterations=steps)
    mask |= binary_dilation(building, iterations=steps * 2)
    if building.any():
        ring = binary_dilation(building, iterations=steps * 3) & ~building
        ys, xs = np.nonzero(ring)
        cy, cx = np.nonzero(building)
        darkness = (lum_around[ys, xs] - lum[ys, xs])
        weights = np.maximum(darkness, 0)
        if weights.sum() > 0:
            # Mean offset of the darker ring pixels from the building centre
            # of mass gives the shadow direction for this tile.
            dy = float(((ys - cy.mean()) * weights).sum() / weights.sum())
            dx = float(((xs - cx.mean()) * weights).sum() / weights.sum())
            norm = max((dy * dy + dx * dx) ** 0.5, 1e-6)
            shift = int(round(4.0 / metres_per_pixel))  # 4 m of cast shadow
            sy, sx = int(round(dy / norm * shift)), int(round(dx / norm * shift))
            shadow = np.zeros_like(building)
            ys0, xs0 = np.nonzero(building)
            for k in range(1, shift + 1):
                ty = np.clip(ys0 + int(round(sy * k / shift)), 0, building.shape[0] - 1)
                tx = np.clip(xs0 + int(round(sx * k / shift)), 0, building.shape[1] - 1)
                shadow[ty, tx] = True
            mask |= binary_dilation(shadow, iterations=steps)
    result = {"mask": mask, "core": core, "road": road, "building": building,
              "donor_exclude": binary_dilation(water, iterations=max(1, int(round(80 * px)))) & (exg < 0.04),
              "water": water, "beach": beach}
    if return_layers:
        result.update(light=light, dim=dim, deficit=deficit)
    return result


def overlay(rgb: np.ndarray, result: dict[str, np.ndarray]) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = result["mask"]
    out[m] = out[m] * 0.35 + np.array((255, 0, 255)) * 0.65
    out[result["building"]] = out[result["building"]] * 0.35 + np.array((0, 255, 255)) * 0.65
    return out.astype(np.uint8)
