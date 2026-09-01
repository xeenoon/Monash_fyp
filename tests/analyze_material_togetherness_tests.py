#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "analyze_material_togetherness.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("analyze_material_togetherness", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


class TogethernessMetricTests(unittest.TestCase):
    def test_compact_region_scores_above_checkerboard(self) -> None:
        compact = np.zeros((128, 128), dtype=bool)
        compact[:, :64] = True
        yy, xx = np.indices((128, 128))
        checkerboard = (xx + yy) % 2 == 0
        self.assertGreater(TOOL.aggregation_curve(compact).mean(),
                           TOOL.aggregation_curve(checkerboard).mean())

    def test_thin_bridge_does_not_inflate_aggregation_like_raw_connectivity(self) -> None:
        separated = np.zeros((128, 128), dtype=bool)
        separated[20:52, 12:44] = True
        separated[76:108, 84:116] = True
        bridged = separated.copy()
        bridged[35, 43:85] = True
        bridged[35:77, 84] = True
        self.assertEqual(TOOL.component_metrics(bridged)["component_count"], 1)
        self.assertLess(TOOL.aggregation_curve(bridged).mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
