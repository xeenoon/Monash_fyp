#!/usr/bin/env python3
"""Replace masked roads/buildings with real terrain borrowed from nearby.

Exemplar ("mosaic") fill: no pixel is invented. The hole is filled from its
edge inwards; at each step the most-constrained patch on the fill front is
completed by copying the best-matching patch of untouched terrain from within
`radius_m` (sum of squared differences over the pixels already known).

Candidates are random local samples plus COHERENT continuations -- the source
offsets that already-filled neighbours used -- so once a good donor region is
found, the fill keeps reading from it and the copied texture runs on
continuously instead of becoming a patchwork. Because the donors are always
nearby, the fill takes on whatever the surroundings are: grass in pasture,
scree on scree.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, convolve


def exemplar_fill(rgb: np.ndarray, mask: np.ndarray, *, metres_per_pixel: float = 0.5,
                  patch: int = 9, radius_m: float = 40.0, samples: int = 240,
                  seed: int = 1, donor_exclude: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w, _ = rgb.shape
    half = patch // 2
    image = rgb.astype(np.float32).copy()
    known = ~mask.copy()
    # Donor patches must be wholly original terrain: never copy filled pixels
    # (that is how errors compound) and never copy what was masked.
    donor_ok = ~binary_dilation(mask, iterations=half)
    donor_ok[:half, :] = donor_ok[-half:, :] = False
    donor_ok[:, :half] = donor_ok[:, -half:] = False
    if donor_exclude is not None:
        # e.g. bare shore round a lake: real terrain, but the wrong material
        # to rebuild the lake bed from.
        donor_ok &= ~binary_dilation(donor_exclude, iterations=half)
    offset = np.zeros((h, w, 2), np.int32)  # source - target, per filled pixel
    has_offset = np.zeros((h, w), bool)
    # A wide hole (a lake) has only its own shore within 40 m of its middle;
    # reach further so its interior is built from the wider surroundings.
    from scipy.ndimage import distance_transform_edt
    deepest = float(distance_transform_edt(mask).max()) if mask.any() else 0.0
    radius = int(max(radius_m / metres_per_pixel, 2.5 * deepest))
    kernel = np.ones((patch, patch), np.float32)
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]

    remaining = int((~known).sum())
    while remaining:
        # Fill front: unknown pixels with known neighbours; priority by how
        # much of their patch is already known.
        support = convolve(known.astype(np.float32), kernel, mode="constant")
        front = ~known & binary_dilation(known, iterations=1)
        ys, xs = np.nonzero(front)
        if not len(ys):  # an island with no known border: seed from anywhere
            ys, xs = np.nonzero(~known)
        order = np.argsort(-support[ys, xs])
        # Take a batch of the most-constrained, mutually distant front pixels.
        taken = np.zeros((h, w), bool)
        batch = []
        for i in order[: max(64, len(order) // 4)]:
            y, x = ys[i], xs[i]
            if taken[max(y - half, 0):y + half + 1, max(x - half, 0):x + half + 1].any():
                continue
            taken[max(y - half, 0):y + half + 1, max(x - half, 0):x + half + 1] = True
            batch.append((y, x))
        for y, x in batch:
            y = int(np.clip(y, half, h - half - 1))
            x = int(np.clip(x, half, w - half - 1))
            ty, tx = y + dy, x + dx
            target = image[ty, tx]
            valid = known[ty, tx]
            # Candidates: coherent continuations first, then local random.
            cands = []
            neigh = has_offset[ty, tx]
            if neigh.any():
                cands.extend(offset[ty, tx][neigh].tolist())
            ry = rng.integers(-radius, radius + 1, samples)
            rx = rng.integers(-radius, radius + 1, samples)
            cands.extend(np.stack([ry, rx], 1).tolist())
            cands = np.unique(np.array(cands, np.int32), axis=0)
            sy, sx = y + cands[:, 0], x + cands[:, 1]
            inside = (sy >= half) & (sy < h - half) & (sx >= half) & (sx < w - half)
            cands, sy, sx = cands[inside], sy[inside], sx[inside]
            ok = donor_ok[sy, sx]
            cands, sy, sx = cands[ok], sy[ok], sx[ok]
            if not len(cands):
                # Nothing clean nearby: take the nearest clean pixel at all.
                dys, dxs = np.nonzero(donor_ok)
                k = np.argmin((dys - y) ** 2 + (dxs - x) ** 2)
                sy, sx = np.array([dys[k]]), np.array([dxs[k]])
                cands = np.array([[dys[k] - y, dxs[k] - x]])
            source = image[sy[:, None, None] + dy, sx[:, None, None] + dx]  # n,p,p,3
            diff = ((source - target[None]) ** 2).sum(-1)
            score = (diff * valid[None]).sum((1, 2)) / max(valid.sum(), 1)
            # Mild preference for nearby donors: same light, same ground.
            score += 0.02 * np.hypot(cands[:, 0], cands[:, 1]) * (40.0 / metres_per_pixel) / radius
            best = int(np.argmin(score))
            fill = ~valid
            image[ty[fill], tx[fill]] = source[best][fill]
            offset[ty[fill], tx[fill]] = cands[best]
            has_offset[ty[fill], tx[fill]] = True
            known[ty[fill], tx[fill]] = True
        remaining = int((~known).sum())
    return np.clip(image, 0, 255).astype(np.uint8), offset
