from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import motif_synthesis_bench as MOTIF  # noqa: E402


def exemplar(rgb: np.ndarray, labels: np.ndarray) -> MOTIF.MotifExemplar:
    orientation, coherence = MOTIF._tensor_descriptor(rgb)
    mix = np.bincount(labels.ravel(), minlength=MOTIF.CLASS_COUNT) / labels.size
    return MOTIF.MotifExemplar(
        rgb, labels, (3, 7), (0, 0), mix.astype(np.float32),
        MOTIF.patch_descriptor(rgb, labels), orientation, coherence,
        MOTIF.transition_pairs(labels))


class FrequencyCompositionTests(unittest.TestCase):
    def test_band_split_reconstructs_source(self) -> None:
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (MOTIF.PATCH, MOTIF.PATCH, 3), dtype=np.uint8)
        low, mid, high = MOTIF.split_bands(rgb)
        np.testing.assert_allclose(low + mid + high, rgb, atol=2e-5)

    def test_one_warp_is_shared_and_retains_residual_energy(self) -> None:
        y, x = np.indices((MOTIF.PATCH, MOTIF.PATCH), dtype=np.float32)
        wave = 110 + 45 * np.sin((x + 2 * y) / 4.0) + 28 * np.sin(y / 17.0)
        rgb = np.repeat(np.clip(wave, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
        labels = np.zeros((MOTIF.PATCH, MOTIF.PATCH), np.uint8)
        source = exemplar(rgb, labels)
        _low, mid, high, _labels, sx, sy = MOTIF.transform_exemplar(
            source, .38, 1.07, 1.5)
        self.assertEqual(sx.shape, labels.shape)
        self.assertEqual(sy.shape, labels.shape)
        source_mid, source_high = MOTIF.split_bands(rgb)[1:]
        self.assertGreater(np.std(mid), .9 * np.std(source_mid))
        self.assertGreater(np.std(high), .9 * np.std(source_high))


class CompositionTests(unittest.TestCase):
    def test_overlap_window_tiles_to_unity(self) -> None:
        window = MOTIF._overlap_window()[MOTIF.PATCH // 2]
        total = np.zeros(MOTIF.PATCH * 3, np.float32)
        for start in range(0, MOTIF.PATCH * 2 + 1, MOTIF.STEP):
            total[start:start + MOTIF.PATCH] += window
        interior = total[MOTIF.PATCH:MOTIF.PATCH * 2]
        np.testing.assert_allclose(interior, 1.0, atol=1e-6)

    def test_style_targets_follow_the_donor_distribution(self) -> None:
        # Four blank donors and sixteen structured ones: a field that spends its
        # area evenly over the range would sit halfway between the two modes.
        rows = np.zeros((20, 11), np.float32)
        rows[:4, 7] = .05
        rows[4:, 7] = .95
        field = MOTIF.continuous_style_field(96, 3, rows)
        median = float(np.median(field[..., 7]))
        self.assertGreater(median, .5)
        self.assertLess(abs(median - .95), abs(median - .5))

    def test_assembly_preserves_donor_residual_energy(self) -> None:
        rng = np.random.default_rng(5)
        y, x = np.indices((MOTIF.PATCH, MOTIF.PATCH), dtype=np.float32)
        bank = []
        for index in range(8):
            angle = rng.uniform(0, np.pi)
            values = (150 + 26 * np.sin((np.cos(angle) * x + np.sin(angle) * y) / 2.5) +
                      16 * np.sin(x / 55 + index))
            rgb = np.repeat(np.clip(values, 0, 255).astype(np.uint8)[..., None], 3, 2)
            labels = np.full((MOTIF.PATCH, MOTIF.PATCH), MOTIF.CLASS_SNOW, np.uint8)
            bank.append(exemplar(rgb, labels))
        donor = float(np.median([np.std(sum(MOTIF.split_bands(e.rgb)[1:])) for e in bank]))
        result = MOTIF.render_material(bank, MOTIF.CLASS_SNOW, 384, 5, np.deg2rad(40))
        # Hard ownership means the canvas cannot be blander than its donors; a
        # seam or a blend that averages two of them would show up here.
        self.assertGreater(float(np.std(result.mid + result.high)), .9 * donor)


class TransitionTests(unittest.TestCase):
    def test_thin_real_transition_is_retained(self) -> None:
        labels = np.zeros((MOTIF.PATCH, MOTIF.PATCH), np.uint8)
        labels[:, -4:] = 2
        rgb = np.repeat((labels * 100 + 30)[..., None], 3, axis=2).astype(np.uint8)
        bank = [exemplar(rgb, labels)]
        np.testing.assert_array_equal(MOTIF.transition_pool(bank, 0, 2), [0])

    def test_transition_warp_obeys_new_boundary_sign(self) -> None:
        labels = np.zeros((MOTIF.PATCH, MOTIF.PATCH), np.uint8)
        labels[:, MOTIF.PATCH // 2:] = 2
        rgb = np.zeros((MOTIF.PATCH, MOTIF.PATCH, 3), np.uint8)
        rgb[labels == 0] = (30, 40, 50)
        rgb[labels == 2] = (210, 220, 230)
        source = exemplar(rgb, labels)
        y, x = np.indices((MOTIF.PATCH, MOTIF.PATCH), dtype=np.float32)
        sdf = x - (MOTIF.PATCH / 2 + 18 * np.sin(y / 24))
        low, mid, high, _sx, _sy = MOTIF.transform_transition(
            source, 0, 2, sdf, (MOTIF.PATCH // 2, MOTIF.PATCH // 2),
            (0, 0), 1.0)
        reconstructed = low + mid + high
        self.assertLess(float(reconstructed[sdf < -24].mean()), 80)
        self.assertGreater(float(reconstructed[sdf > 24].mean()), 170)

    def test_high_frequency_ownership_is_not_alpha_blended(self) -> None:
        size = 128
        shape = (size, size, 3)

        def render(value: float) -> MOTIF.RenderResult:
            return MOTIF.RenderResult(
                np.full(shape, value, np.float32), np.full(shape, value, np.float32),
                np.full(shape, value, np.float32), np.zeros(shape, np.uint8),
                np.zeros((size, size), np.int16), np.zeros((size, size), np.int16),
                np.zeros((size, size), np.int16), np.zeros((size, size), np.int16),
                np.full((size, size), 3, np.uint8))

        first, second, transition = render(1), render(9), render(40)
        sdf = np.indices((size, size), dtype=np.float32)[1] - size / 2
        result = MOTIF.compose_transition(first, second, transition, sdf)
        # The central strip has one transition owner, while pixels outside it
        # keep one material's residual exactly; no weighted HIGH average exists.
        self.assertEqual(float(result.high[size // 2, size // 2, 0]), 40)
        self.assertEqual(float(result.high[size // 2, 0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
