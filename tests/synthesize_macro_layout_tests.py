from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import diagnose_macro_structure as DIAG  # noqa: E402
import synthesize_macro_layout as LAYOUT  # noqa: E402


def fake_library(seed: int = 8) -> LAYOUT.StructureLibrary:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(600, 6)).astype(np.float32)
    features[:, 0] = rng.uniform(1.5, 3.5, len(features))
    materials = np.where(features[:, 0] > 2.8, 2,
                         np.where(features[:, 1] > 0.1, 0, 1)).astype(np.uint8)
    phrases = []
    yy, xx = np.indices((16, 16))
    for material in range(3):
        for index in range(4):
            mask = (((xx - 8) / (3 + index)) ** 2 + ((yy - 8) / (6 - index / 2)) ** 2 < 1)
            phrases.append(LAYOUT.ShapePhrase(material, mask.astype(np.float32),
                                               (index, material), 0.0, int(mask.sum())))
    coordinates = [(index, material) for material in range(3) for index in range(4)]
    return LAYOUT.StructureLibrary(coordinates, features, materials, phrases,
                                   cKDTree(features))


class MacroLayoutTests(unittest.TestCase):
    def test_generation_is_seed_deterministic_but_seed_sensitive(self) -> None:
        y, x = np.indices((64, 64), dtype=np.float32)
        height = 1800 + 12 * x + 5 * y + 80 * np.sin(y / 9)
        slope = np.hypot(*np.gradient(height / 100.0)).astype(np.float32)
        library = fake_library()
        first = LAYOUT.generate_layout(height, slope, library, seed=45,
                                       placements=18, streaks=5)
        second = LAYOUT.generate_layout(height, slope, library, seed=45,
                                        placements=18, streaks=5)
        third = LAYOUT.generate_layout(height, slope, library, seed=46,
                                       placements=18, streaks=5)
        np.testing.assert_array_equal(first.labels, second.labels)
        self.assertFalse(np.array_equal(first.labels, third.labels))

    def test_generation_has_no_imagery_dependency(self) -> None:
        height = np.linspace(1700, 3100, 48 * 48, dtype=np.float32).reshape(48, 48)
        slope = np.full_like(height, 0.7)
        with mock.patch.object(LAYOUT, "tile_rgb", side_effect=AssertionError("RGB forbidden")):
            result = LAYOUT.generate_layout(height, slope, fake_library(), seed=2,
                                            placements=8, streaks=3)
        self.assertEqual(result.labels.shape, height.shape)

    def test_shifted_correlation_detects_copy_but_not_new_phase(self) -> None:
        rng = np.random.default_rng(4)
        reference = rng.normal(size=(64, 64))
        copied = np.roll(reference, (13, -9), axis=(0, 1))
        unrelated = rng.normal(size=(64, 64))
        self.assertGreater(DIAG.circular_max_correlation(copied, reference), 0.999)
        self.assertLess(DIAG.circular_max_correlation(unrelated, reference), 0.15)

    def test_compare_images_separates_spectrum_from_phase(self) -> None:
        rng = np.random.default_rng(19)
        reference = rng.normal(size=(64, 64))
        spectrum = np.fft.fft2(reference)
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi, spectrum.shape))
        generated = np.fft.ifft2(np.abs(spectrum) * phase).real
        def image(values: np.ndarray) -> Image.Image:
            values = (values - values.min()) / (values.max() - values.min()) * 255
            return Image.fromarray(values.astype(np.uint8), "L")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated_path, reference_path = root / "generated.png", root / "reference.png"
            image(generated).save(generated_path); image(reference).save(reference_path)
            metrics = DIAG.compare_images(generated_path, reference_path)
        self.assertLess(metrics["summary"]["max_rgb_correlation"], 0.5)
        self.assertGreater(metrics["summary"]["structure_retention"], 0.25)


if __name__ == "__main__":
    unittest.main()
