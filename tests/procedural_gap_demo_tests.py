#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "procedural_gap_demo.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("procedural_gap_demo", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


class QuiltingSeamTests(unittest.TestCase):
    def test_vertical_seam_follows_low_cost_channel(self) -> None:
        cost = np.ones((12, 8), dtype=np.float32)
        cost[:, 3] = 0.0
        np.testing.assert_array_equal(TOOL.minimum_vertical_seam(cost), 3)

    def test_unwritten_pixels_are_always_taken(self) -> None:
        existing = np.zeros((8, 8), dtype=np.uint8)
        candidate = np.ones((8, 8), dtype=np.uint8)
        known = np.ones((8, 8), dtype=bool)
        known[:, 4:] = False
        take = TOOL.quilt_take_mask(existing, candidate, known, overlap=4,
                                    has_left=True, has_top=False)
        self.assertTrue(take[:, 4:].all())

    def test_left_overlap_preserves_pixels_before_seam(self) -> None:
        existing = np.zeros((8, 8), dtype=np.uint8)
        candidate = np.ones((8, 8), dtype=np.uint8)
        known = np.ones((8, 8), dtype=bool)
        take = TOOL.quilt_take_mask(existing, candidate, known, overlap=4,
                                    has_left=True, has_top=False)
        self.assertFalse(take[:, 0].any())
        self.assertTrue(take[:, 4:].all())

    def test_layout_marks_both_tile_seams_in_red(self) -> None:
        tile = np.zeros((TOOL.SIZE, TOOL.SIZE, 3), dtype=np.uint8)
        result = TOOL.layout(tile, tile, tile)
        expected = np.broadcast_to(TOOL.SEAM_GUIDE_COLOR, (TOOL.SIZE * 2, 3))
        np.testing.assert_array_equal(result[TOOL.SIZE], expected)
        np.testing.assert_array_equal(result[:, TOOL.SIZE], expected)

    def test_unmarked_layout_preserves_clean_comparison_pixels(self) -> None:
        tile = np.zeros((TOOL.SIZE, TOOL.SIZE, 3), dtype=np.uint8)
        result = TOOL.unmarked_layout(tile, tile, tile)
        self.assertFalse(result.any())

    def test_cropped_target_marks_north_and_east_seams(self) -> None:
        tile = np.zeros((TOOL.SIZE, TOOL.SIZE, 3), dtype=np.uint8)
        result = TOOL.mark_target_seams(tile)
        np.testing.assert_array_equal(
            result[0], np.broadcast_to(TOOL.SEAM_GUIDE_COLOR, (TOOL.SIZE, 3)))
        np.testing.assert_array_equal(
            result[:, -1], np.broadcast_to(TOOL.SEAM_GUIDE_COLOR, (TOOL.SIZE, 3)))
        self.assertFalse(result[1:, :-1].any())

    def test_edge_occupancy_is_a_bounded_per_class_field(self) -> None:
        labels = np.zeros((TOOL.SIZE, TOOL.SIZE), dtype=np.uint8)
        labels[:, TOOL.SIZE // 3:2 * TOOL.SIZE // 3] = 1
        labels[:, 2 * TOOL.SIZE // 3:] = 2
        guide = TOOL.extrapolate_edge_occupancy(labels, "south", guide_depth=12)
        self.assertEqual(guide.shape, (3, 12, TOOL.SIZE))
        self.assertGreaterEqual(float(guide.min()), 0.0)
        self.assertLessEqual(float(guide.max()), 1.0)
        np.testing.assert_allclose(guide.sum(axis=0), 1.0, atol=2.0e-3)

    def test_hard_corner_is_fixed_only_when_guides_agree(self) -> None:
        north_labels = np.zeros((TOOL.EDGE_GUIDE_DEPTH, TOOL.SIZE), dtype=np.uint8)
        east_labels = np.ones((TOOL.SIZE, TOOL.EDGE_GUIDE_DEPTH), dtype=np.uint8)
        donor_north = SimpleNamespace(
            colour_class=np.zeros((TOOL.SIZE, TOOL.SIZE), dtype=np.uint8))
        donor_east = SimpleNamespace(
            colour_class=np.ones((TOOL.SIZE, TOOL.SIZE), dtype=np.uint8))
        fixed, fixed_labels = TOOL.edge_hard_constraints(
            donor_north, donor_east, north_labels, east_labels)
        depth = TOOL.COHERENCE_HARD_EDGE_DEPTH
        self.assertFalse(fixed[1:depth, -depth:-1].any())
        self.assertTrue(fixed[depth, -2])
        self.assertEqual(int(fixed_labels[depth, -2]), 1)
        self.assertTrue(fixed[2, -depth - 1])
        self.assertEqual(int(fixed_labels[2, -depth - 1]), 0)

    def test_continuous_source_coordinate_field_has_no_internal_jumps(self) -> None:
        yy, xx = np.indices((TOOL.SIZE, TOOL.SIZE), dtype=np.int16)
        donors = np.zeros((TOOL.SIZE, TOOL.SIZE), dtype=np.int16)
        discontinuities = TOOL.source_discontinuities(donors, xx, yy)
        self.assertFalse(discontinuities.any())

    def test_source_donor_switch_is_reported_as_a_jump(self) -> None:
        yy, xx = np.indices((TOOL.SIZE, TOOL.SIZE), dtype=np.int16)
        donors = np.zeros((TOOL.SIZE, TOOL.SIZE), dtype=np.int16)
        donors[:, TOOL.SIZE // 2:] = 1
        discontinuities = TOOL.source_discontinuities(donors, xx, yy)
        self.assertTrue(discontinuities[:, TOOL.SIZE // 2].all())
        self.assertFalse(discontinuities[:, :TOOL.SIZE // 2].any())

    def test_semantic_layout_error_does_not_hide_rare_classes(self) -> None:
        target = np.zeros((10, 10), dtype=np.uint8)
        target[0, :5] = 1
        miss_rare = target.copy()
        miss_rare[0, 0] = 0
        miss_common = target.copy()
        miss_common[1, 0] = 1
        rare_error = float(TOOL.semantic_layout_error(miss_rare, target))
        common_error = float(TOOL.semantic_layout_error(miss_common, target))
        self.assertGreater(rare_error, common_error)

    def test_repetition_metric_detects_a_tiled_motif(self) -> None:
        rng = np.random.default_rng(5)
        motif = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        tiled = np.tile(motif, (4, 4, 1))
        noise = rng.integers(0, 256, tiled.shape, dtype=np.uint8)
        tiled_fraction = TOOL.distant_patch_repetition(tiled)[2]
        noise_fraction = TOOL.distant_patch_repetition(noise)[2]
        self.assertGreater(tiled_fraction, noise_fraction)

    def test_edge_snap_changes_only_target_and_leaves_interior_clean(self) -> None:
        rng = np.random.default_rng(7)
        target = rng.integers(0, 256, (TOOL.SIZE, TOOL.SIZE, 3), dtype=np.uint8)
        north = rng.integers(0, 256, target.shape, dtype=np.uint8)
        east = rng.integers(0, 256, target.shape, dtype=np.uint8)
        north_before, east_before = north.copy(), east.copy()
        blended, strength = TOOL.snap_generated_edges(target, north, east, band=12)
        np.testing.assert_array_equal(north, north_before)
        np.testing.assert_array_equal(east, east_before)
        np.testing.assert_array_equal(blended[0, 20], north[-1, 20])
        np.testing.assert_array_equal(blended[20, -1], east[20, 0])
        np.testing.assert_array_equal(blended[20:, :TOOL.SIZE - 20],
                                      target[20:, :TOOL.SIZE - 20])
        self.assertEqual(float(strength[0, 20]), 1.0)
        self.assertEqual(float(strength[20, 20]), 0.0)


    def test_grade_patch_moves_overlap_toward_neighbour_exposure(self) -> None:
        patch, overlap_width = 32, 8
        candidate = np.full((patch, patch, 3), 60, dtype=np.uint8)   # dark swatch
        existing = np.full((patch, patch, 3), 190, dtype=np.uint8)   # bright neighbour
        overlap = np.zeros((patch, patch), dtype=bool)
        overlap[:, :overlap_width] = True
        graded = TOOL.grade_patch_to_context(candidate, existing, overlap, None)
        self.assertEqual(graded.shape, candidate.shape)
        self.assertEqual(graded.dtype, np.uint8)
        # The whole swatch is lifted toward the bright neighbour it joins.
        self.assertGreater(float(graded.mean()), float(candidate.mean()) + 10.0)
        self.assertLess(float(graded.mean()), float(existing.mean()))

    def test_source_polygons_split_on_a_donor_switch(self) -> None:
        yy, xx = np.indices((TOOL.SIZE, TOOL.SIZE), dtype=np.int16)
        donors = np.zeros((TOOL.SIZE, TOOL.SIZE), dtype=np.int16)
        # One contiguous translation everywhere is a single polygon.
        labels, count = TOOL.source_polygon_labels(donors, xx.copy(), yy.copy())
        self.assertEqual(count, 1)
        self.assertFalse(TOOL.polygon_outline_mask(labels).any())
        # A donor switch down the middle carves exactly two polygons with an
        # outline along the boundary column.
        donors[:, TOOL.SIZE // 2:] = 1
        labels, count = TOOL.source_polygon_labels(donors, xx.copy(), yy.copy())
        self.assertEqual(count, 2)
        outline = TOOL.polygon_outline_mask(labels)
        self.assertTrue(outline[:, TOOL.SIZE // 2].all())

    def test_grade_patch_without_overlap_uses_fallback_mean(self) -> None:
        patch = 16
        candidate = np.full((patch, patch, 3), 60, dtype=np.uint8)
        overlap = np.zeros((patch, patch), dtype=bool)
        bright = TOOL.srgb_to_linear(np.full((1, 1, 3), 200, dtype=np.uint8)).reshape(3)
        graded = TOOL.grade_patch_to_context(candidate, candidate, overlap, bright)
        self.assertGreater(float(graded.mean()), float(candidate.mean()) + 10.0)
        # With neither overlap nor fallback the swatch is returned untouched.
        untouched = TOOL.grade_patch_to_context(candidate, candidate, overlap, None)
        np.testing.assert_array_equal(untouched, candidate)


if __name__ == "__main__":
    unittest.main()
