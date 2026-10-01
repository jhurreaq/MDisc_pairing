"""Fast checks of the public numerical, state, and execution contracts.

The complete legacy/new comparisons live in scripts/validate_equivalence.py.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from mdisc.isotropic import compute_invariants, compute_invariant_derivatives, mooney_rivlin_terms
from mdisc.preprocessing import fit_standard_scaler, prepare_weighted_normalized_design, build_stress_magnitude_weights
from mdisc.regression import refit_support_nnls, refit_nnls_with_intercept, clip_nonnegative
from mdisc.runner import run_case
from mdisc.paths import positive_omp_path, unique_size_supports


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NumericalContracts(unittest.TestCase):
    def test_reference_configuration_invariants(self):
        self.assertEqual(compute_invariants(1.0, 1.0, 1.0), (3.0, 3.0))
        self.assertEqual(compute_invariant_derivatives(1.0, 1.0, 1.0), ((2.0, 2.0, 2.0), (4.0, 4.0, 4.0)))

    def test_mr_basis_order_and_power_limits(self):
        self.assertEqual(list(mooney_rivlin_terms(2)), [
            "MR: (I₂-3)", "MR: (I₁-3)", "MR: (I₂-3)^2",
            "MR: (I₁-3)(I₂-3)", "MR: (I₁-3)^2",
        ])
        self.assertEqual(list(mooney_rivlin_terms(3, max_i1=1, max_i2=1)), [
            "MR: (I₂-3)", "MR: (I₁-3)", "MR: (I₁-3)(I₂-3)",
        ])

    def test_standardization_retains_original_units_on_inverse_transform(self):
        values = np.array([[1.0, 10.0, 7.0], [3.0, 30.0, 7.0], [5.0, 50.0, 7.0]])
        scaler = fit_standard_scaler(values)
        scaled = scaler.transform(values)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-15)
        np.testing.assert_array_equal(scaled[:, 2], 0.0)
        np.testing.assert_allclose(scaler.inverse_transform(scaled), values, rtol=0, atol=1e-14)

    def test_cardiac_row_weighting_and_column_back_transform(self):
        X = np.array([[3.0, 0.0], [0.0, 4.0]])
        y = np.array([2.0, 6.0])
        normalized, weighted_y, scales = prepare_weighted_normalized_design(X, y, np.array([4.0, 9.0]))
        np.testing.assert_array_equal(normalized, np.eye(2))
        np.testing.assert_array_equal(weighted_y, [4.0, 18.0])
        np.testing.assert_array_equal(scales, [6.0, 12.0])
        beta = np.array([2.0, 3.0])
        np.testing.assert_array_equal(normalized @ beta, (X @ (beta / scales)) * np.array([2.0, 3.0]))

    def test_stress_weights_use_inverse_rms_instead_of_inverse_squared_rms(self):
        data = [{"test_name": "A", "target_stress": 1.0},
                {"test_name": "A", "target_stress": 1.0},
                {"test_name": "B", "target_stress": 2.0}]
        with redirect_stdout(StringIO()):
            weights = build_stress_magnitude_weights(data)
        self.assertEqual(weights[0], weights[1])
        self.assertEqual(weights[0] / weights[2], 2.0)

    def test_nonnegative_support_refit_preserves_a_free_intercept(self):
        X = np.random.default_rng(5).normal(size=(40, 3))
        true_beta = np.array([2.0, 0.0, 4.0])
        y = 7.0 + X @ true_beta
        beta, intercept, prediction = refit_support_nnls(X, y, [0, 2])
        np.testing.assert_allclose(beta, true_beta, rtol=0, atol=1e-14)
        self.assertAlmostEqual(intercept, 7.0, places=13)
        np.testing.assert_allclose(prediction, y, rtol=0, atol=2e-14)

    def test_treloar_nnls_fallbacks_recover_positive_coefficients(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        y = 3.0 + X @ np.array([2.0, 4.0])
        with patch("mdisc.regression.nnls", side_effect=RuntimeError("force fallback")), patch("scipy.optimize.nnls", side_effect=RuntimeError("force fallback")):
            for variant in ("lars", "omp"):
                beta, intercept = refit_nnls_with_intercept(X, y, variant=variant)
                np.testing.assert_allclose(beta, [2.0, 4.0], rtol=0, atol=1e-6)
                self.assertAlmostEqual(intercept, 3.0, places=6)

    def test_coefficient_clipping_preserves_the_original_threshold_boundary(self):
        np.testing.assert_array_equal(clip_nonnegative([-1.0, 1e-13, 1e-12, 0.5]), [0.0, 0.0, 1e-12, 0.5])

    def test_positive_omp_path_recovers_an_orthogonal_mixture(self):
        X = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
        y = 5.0 + X @ np.array([2.0, 0.5])
        with redirect_stdout(StringIO()):
            for variant in ("synthetic", "treloar"):
                coefficients = positive_omp_path(X, y, variant=variant)
                np.testing.assert_allclose(coefficients[:, -1], [2.0, 0.5], rtol=0, atol=1e-14)
                self.assertTrue(np.all(coefficients >= 0))

    def test_lars_compression_keeps_the_first_support_at_each_size(self):
        coefficients = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 3.0], [0.0, 0.0, 4.0]])
        supports, indices = unique_size_supports(coefficients)
        self.assertEqual(indices, [0, 1])
        np.testing.assert_array_equal(supports[0], [0])
        np.testing.assert_array_equal(supports[1], [0, 1])


class ExecutionContracts(unittest.TestCase):
    def test_case_state_is_fresh_per_invocation(self):
        from Isotropic.Benchmarks.configuration import create_context
        with warnings.catch_warnings():
            first, second = create_context(), create_context()
        first.OGDEN_EXPONENTS.append(10)
        first.INCLUDE_MOONEY_RIVLIN = False
        self.assertEqual(second.OGDEN_EXPONENTS, [])
        self.assertTrue(second.INCLUDE_MOONEY_RIVLIN)

    def test_all_workflows_import_without_data_or_output_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), MPLBACKEND="Agg")
            process = subprocess.run([sys.executable, "-c",
                "import Isotropic.Benchmarks.workflow, Isotropic.Treloar.workflow, Anisotropic.workflow"],
                cwd=directory, env=environment, text=True, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_runner_restores_working_directory_when_input_loading_fails(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "Anisotropic").mkdir()
            with self.assertRaises(SystemExit):
                run_case("cardiac", project_root=directory)
        self.assertEqual(Path.cwd(), previous)

    def test_centralized_scripts_support_help_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in ("synthetic", "treloar", "cardiac"):
                process = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / f"run_{case}.py"), "--help"],
                    cwd=directory, text=True, capture_output=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("--project-root", process.stdout)


if __name__ == "__main__":
    unittest.main()
