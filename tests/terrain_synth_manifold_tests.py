#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "terrain_synth.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("terrain_synth", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


class DonorManifoldTests(unittest.TestCase):
    def test_material_probabilities_are_continuous_and_normalized(self) -> None:
        rgb = np.array([[[45, 115, 35], [125, 120, 112], [242, 244, 245]]], np.uint8)
        probabilities = TOOL.material_probability_map(rgb)
        np.testing.assert_allclose(probabilities.sum(-1), 1.0, atol=1.0e-6)
        self.assertEqual(int(probabilities[0, 0].argmax()), 0)  # grass
        self.assertEqual(int(probabilities[0, 1].argmax()), 1)  # rock
        self.assertEqual(int(probabilities[0, 2].argmax()), 2)  # snow
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))

    def test_magenta_is_repaired_and_confidence_retained(self) -> None:
        rgb = np.full((48, 48, 3), (115, 112, 104), np.uint8)
        rgb[20:28, 20:28] = (230, 20, 220)
        cleaned, confidence, hard = TOOL.clean_source(rgb)
        self.assertTrue(hard[24, 24])
        self.assertLess(int(confidence[24, 24]), 64)
        self.assertLess(int(cleaned[24, 24, 0] - cleaned[24, 24, 1]), 40)

    def test_dark_non_green_rock_is_not_mistaken_for_tree(self) -> None:
        rgb = np.full((48, 48, 3), (45, 42, 40), np.uint8)
        repair, hard = TOOL.detect_nonterrain(rgb)
        self.assertFalse(repair.any())
        self.assertFalse(hard.any())

    def test_bounded_style_target_limits_one_step_and_normalizes_material(self) -> None:
        current = np.zeros(TOOL.STYLE_DIM, np.float32)
        current[TOOL.MATERIAL] = (0.9, 0.1, 0.0)
        target = np.ones(TOOL.STYLE_DIM, np.float32)
        target[TOOL.MATERIAL] = (0.0, 0.0, 1.0)
        result = TOOL.bounded_style_target(target, current, amount=0.30)
        self.assertLessEqual(float(np.linalg.norm((result - current) / TOOL.STYLE_SCALE)), 0.301)
        self.assertAlmostEqual(float(result[TOOL.MATERIAL].sum()), 1.0, places=5)
        self.assertGreater(float(result[8]), 0.0)
        self.assertLess(float(result[8]), 1.0)

    def test_graph_path_uses_real_intermediate_nodes(self) -> None:
        count = 5
        style = np.zeros((count, TOOL.STYLE_DIM), np.float32)
        style[:, 6] = np.linspace(1, 0, count)
        style[:, 7] = np.linspace(0, 1, count)
        normalized = style / TOOL.STYLE_SCALE
        chain = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]], np.int32)
        dummy = np.zeros(count, np.float32)
        tree = cKDTree(np.arange(count)[:, None])
        bank = TOOL.PatchBank(
            32, np.zeros(count, np.int32), np.zeros(count, np.int32), np.zeros(count, np.int32),
            np.column_stack((np.arange(count), np.zeros(count))).astype(np.float32),
            np.zeros((count, 7), np.float32), style, np.ones(count, np.float32), dummy, dummy,
            chain, chain, cKDTree(np.zeros((count, 7))), cKDTree(normalized), tree, None,
            np.empty(0, np.int32))
        path = TOOL.graph_path(bank, 0, 4)
        self.assertEqual(path[0], 0)
        self.assertIn(path[-1], (3, 4))
        self.assertGreater(len(path), 1)
        self.assertNotEqual(path[1], 4)

    def test_source_discontinuity_marks_only_true_coordinate_jumps(self) -> None:
        yy, xx = np.indices((16, 16), dtype=np.int32)
        donor = np.zeros((16, 16), np.int32)
        self.assertFalse(TOOL.source_discontinuities(donor, xx, yy).any())
        donor[:, 8:] = 1
        jump = TOOL.source_discontinuities(donor, xx, yy)
        self.assertTrue(jump[:, 8].all())

    def test_frequency_mix_keeps_base_mean_and_uses_detail_residual(self) -> None:
        yy, xx = np.indices((32, 32))
        base = np.full((32, 32, 3), 130, np.uint8)
        detail = np.where(((xx + yy) & 1)[..., None], 190, 70).astype(np.uint8)
        detail = np.broadcast_to(detail, base.shape).copy()
        mixed = TOOL.frequency_decoupled_patch(base, detail, strength=0.6)
        self.assertLess(abs(float(mixed.mean()) - float(base.mean())), 3.0)
        self.assertGreater(float(mixed.std()), 15.0)


if __name__ == "__main__":
    unittest.main()
