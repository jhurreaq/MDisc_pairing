"""Scientific display semantics and rendered text spacing."""
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

from mdisc.configuration import PlotSettings
from mdisc.mechanics import cardiac_library, isotropic_library
from mdisc.rendering import selection_figure, activation_figure, plot_style
from mdisc.reporting import json_value, replot_selection
from mdisc.types import Candidate, DiscoveryResult, ModelFit


def rendered_text_overlaps(figure):
    """Check actual glyph extents, including ticks, labels, titles and legends."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [artist for artist in figure.findobj(Text)
             if artist.get_visible() and artist.get_text().strip()]
    overlaps = []
    for left, right in combinations(texts, 2):
        a, b = left.get_window_extent(renderer), right.get_window_extent(renderer)
        if min(a.x1, b.x1)-max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1)-max(a.y0, b.y0) > 0.5:
            overlaps.append((left.get_text(), right.get_text()))
    return overlaps


def fixture(library, method='Lasso', zero=False):
    x = np.geomspace(1, 1e-6, 40) if method == 'Lasso' else np.linspace(0, 1, 21)
    candidates = []
    for index, parameter in enumerate(x):
        coefficients = np.zeros(len(library.active_terms))
        if not zero:
            coefficients[:min(index, len(coefficients))] = np.geomspace(1e-12, 1e3, min(index, len(coefficients)))
        fit = ModelFit(coefficients, np.zeros(5), float(np.count_nonzero(coefficients)))
        candidates.append(Candidate(float(parameter), (), fit,
            aic=-6000+100*index, bic=-5500+100*index,
            cv_mse=1.5/(index+1), cv_std=0.4/(index+1),
            path_coefficients=-np.ones(len(coefficients))))
    return DiscoveryResult(method, candidates, {'CV': 12, 'AIC': 10, 'BIC': 10}, [], {})


class Rendering(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_labels_do_not_overlap_at_original_sizes_and_fonts(self):
        for library in (cardiac_library(), isotropic_library(mr_order=4, ogden_exponents=(-10,-3,1,10))):
            for method in ('Lasso', 'LARS', 'OMP'):
                result = fixture(library, method)
                for factory in (selection_figure, activation_figure):
                    with self.subTest(method=method, library=library.terms[0].family, figure=factory.__name__):
                        fig = (factory(result, library, style=plot_style()) if factory == activation_figure
                               else factory(result, style=plot_style()))
                        self.assertEqual(rendered_text_overlaps(fig), [])
                        if factory == selection_figure:
                            np.testing.assert_array_equal(fig.get_size_inches(), (3, 4.5))
                            self.assertEqual(fig.axes[0].yaxis.label.get_fontsize(), 14)
                        else:
                            self.assertEqual(fig.get_size_inches()[0], 7)
                            self.assertEqual(fig.axes[0].get_yticklabels()[0].get_fontsize(), 9)
                        plt.close(fig)

    def test_heatmap_is_binary_unit_invariant_and_uses_refitted_support(self):
        library = cardiac_library()
        result = fixture(library)
        fig = activation_figure(result, library, style=plot_style())
        image = np.array(fig.axes[0].images[0].get_array())
        self.assertEqual(set(np.unique(image)), {False, True})
        expected = np.column_stack([c.fit.coefficients > 0 for c in result.candidates])
        np.testing.assert_array_equal(image, expected)
        labels = [label.get_text() for label in fig.axes[0].get_yticklabels()]
        for term, label in zip(library.active_terms, labels):
            self.assertIn(f'c_{{{term.term_id}}}', label)
        # The discovery proposal is deliberately negative everywhere in this fixture.
        # It must not override the physically refitted support, or reintroduce exclusions.
        for c in result.candidates:
            c.fit.coefficients *= 1e-6
        second = activation_figure(result, library, style=plot_style())
        np.testing.assert_array_equal(image, second.axes[0].images[0].get_array())
        self.assertEqual(len(fig.axes[0].lines), 2)  # coincident AIC/BIC grouped
        self.assertIn('AIC/BIC', fig.legends[1].get_texts()[1].get_text())

    def test_criteria_display_unmodified_values_and_correct_parameter(self):
        for method in ('Lasso', 'LARS', 'OMP'):
            result = fixture(cardiac_library(), method)
            fig = selection_figure(result, style=plot_style())
            for ax, metric in zip(fig.axes, ('cv_mse', 'aic', 'bic')):
                np.testing.assert_array_equal(ax.lines[0].get_ydata(),
                                              [getattr(c, metric) for c in result.candidates])
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), [c.parameter for c in result.candidates])
            self.assertGreater(fig.axes[0].get_ylim()[1], 1.5)  # no clipping CV to [0,1]
            self.assertEqual(fig.axes[-1].get_xscale(), 'log' if method == 'Lasso' else 'linear')
            self.assertEqual(fig.axes[-1].xaxis_inverted(), method == 'Lasso')

    def test_zero_response_still_has_a_heatmap_file_and_clear_explanation(self):
        fig = activation_figure(fixture(cardiac_library(), zero=True), cardiac_library(), style=plot_style())
        self.assertIn('No active terms', fig.axes[0].texts[0].get_text())
        self.assertEqual(rendered_text_overlaps(fig), [])

    def test_replay_preserves_numerical_report_and_log_without_discovery(self):
        library = cardiac_library()
        report = {'provenance': {'configuration': {'plot': asdict(PlotSettings())}},
                  'scenarios': {'example': {'library': [asdict(t) for t in library.terms],
                                          'paths': {'Lasso': fixture(library)}}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'results.json'
            path.write_text(json.dumps(json_value(report)))
            log = Path(folder)/'log.txt'; log.write_text('Original models and metrics\n')
            original = path.read_bytes(), log.read_bytes()
            with patch('mdisc.reporting.discover', side_effect=AssertionError('Replay must never fit')):
                replot_selection(path)
            self.assertEqual(original, (path.read_bytes(), log.read_bytes()))
            self.assertEqual(len(list(Path(folder).rglob('*.pdf'))), 2)


if __name__ == '__main__':
    unittest.main()
