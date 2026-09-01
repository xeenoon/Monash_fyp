#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import procedural_gap_demo as DEMO  # noqa: E402
import generate_material_combo_tiles as COMBOS  # noqa: E402
import terrain_shape_synth as SHAPES  # noqa: E402


class MacroMicroShapeTests(unittest.TestCase):
    def test_combo_pure_and_transition_probability_fields(self) -> None:
        height = np.linspace(0, 1, SHAPES.SIZE * SHAPES.SIZE,
                             dtype=np.float32).reshape(SHAPES.SIZE, SHAPES.SIZE)
        snow = COMBOS.preset_probabilities("full_snow", height, 1)
        self.assertTrue(np.all(snow[..., DEMO.CLASS_SNOW] == 1.0))
        transition = COMBOS.preset_probabilities("rock_grass", height, 2)
        np.testing.assert_allclose(transition.sum(2), 1.0, atol=1.0e-6)
        self.assertGreater(float(transition[:, :32, DEMO.CLASS_ROCK].mean()), 0.8)
        self.assertGreater(float(transition[:, -32:, DEMO.CLASS_GRASS].mean()), 0.8)

    def test_soft_choice_is_not_a_hidden_argmin(self) -> None:
        scores = np.linspace(0.0, 1.0, 24)
        selected = {SHAPES.soft_choice(scores, np.random.default_rng(seed),
                                       temperature=1.5, finalists=24)
                    for seed in range(40)}
        self.assertIn(0, selected)
        self.assertGreater(len(selected), 5)

    def test_exact_material_mass_has_zero_error(self) -> None:
        labels = np.zeros((SHAPES.SIZE, SHAPES.SIZE), np.uint8)
        labels[:, SHAPES.SIZE // 2:] = 1
        probabilities = np.eye(3, dtype=np.float32)[labels]
        self.assertAlmostEqual(SHAPES.mass_error(labels, probabilities), 0.0)
        collapsed = np.zeros_like(labels)
        self.assertGreater(SHAPES.mass_error(collapsed, probabilities), 0.1)

    def test_signed_distance_is_positive_inside_and_negative_outside(self) -> None:
        mask = np.zeros((32, 32), bool); mask[8:24, 10:22] = True
        sdf = SHAPES.signed_distance(mask)
        self.assertGreater(float(sdf[16, 16]), 0.0)
        self.assertLess(float(sdf[0, 0]), 0.0)

    def test_native_frequency_bands_are_zero_mean_and_independently_phased(self) -> None:
        rng = np.random.default_rng(9)
        source = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
        bands = [DEMO.native_frequency_band(source, level, 17) for level in range(3)]
        for band in bands:
            np.testing.assert_allclose(band.mean(axis=(0, 1)), 0.0, atol=1.0e-3)
        self.assertFalse(np.allclose(bands[0], bands[1]))
        self.assertFalse(np.allclose(bands[1], bands[2]))

    def test_real_minimum_cut_remains_active_for_categorical_overlap(self) -> None:
        existing = np.zeros((20, 20), np.uint8)
        candidate = np.ones((20, 20), np.uint8)
        known = np.ones((20, 20), bool)
        take = DEMO.quilt_take_mask(existing, candidate, known, overlap=8,
                                    has_left=True, has_top=False)
        self.assertFalse(take[:, 0].any())
        self.assertTrue(take[:, 8:].all())
        self.assertFalse(hasattr(DEMO, "source_ownership"))

    def test_residual_cut_preserves_one_native_detail_layer(self) -> None:
        existing = np.full((20, 20, 3), -12.0, np.float32)
        candidate = np.full((20, 20, 3), 12.0, np.float32)
        known = np.ones((20, 20), bool)
        take = DEMO.residual_quilt_take_mask(
            existing, candidate, known, overlap=8, has_left=True, has_top=False)
        composed = existing.copy(); composed[take] = candidate[take]
        # Seam ownership must choose actual signed donor detail, never the
        # zero-valued blur produced by averaging opposite-phase residuals.
        self.assertTrue(np.all(np.abs(composed) == 12.0))
        self.assertFalse(take[:, 0].any())
        self.assertTrue(take[:, 8:].all())

    def test_native_microtexture_relayer_replaces_blurry_surface(self) -> None:
        base = np.full((32, 32, 3), 120, np.uint8)
        checker = ((np.indices((32, 32)).sum(0) % 2) * 60 + 80).astype(np.uint8)
        rgb = np.repeat(checker[..., None], 3, axis=2)
        donor = type("Donor", (), {"rgb": rgb})()
        donor_map = np.zeros((32, 32), np.int16)
        yy, xx = np.indices((32, 32), dtype=np.int16)
        result, native, fine = DEMO.relayer_native_microtexture(
            base, [donor], donor_map, xx, yy, macro_sigma=3.0,
            detail_sigma=1.0, mid_gain=1.0, fine_gain=1.0)
        self.assertGreater(float(result.std()), 20.0)
        np.testing.assert_array_equal(native, rgb)
        np.testing.assert_array_equal(fine, rgb)


if __name__ == "__main__":
    unittest.main()
