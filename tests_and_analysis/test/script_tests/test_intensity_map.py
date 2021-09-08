import os
import json
from unittest.mock import patch
from platform import platform

import pytest
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot
from packaging import version
from scipy import __version__ as scipy_ver

from tests_and_analysis.test.utils import get_data_path, get_castep_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_image_data, args_to_key)
import euphonic.cli.intensity_map


quartz_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'quartz',
    'quartz_bandstructure_qpoint_phonon_modes.json')
lzo_phonon_file = get_castep_path('LZO', 'La2Zr2O7.phonon')
graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
intensity_map_output_file = os.path.join(get_script_test_data_path(),
                                         'intensity-map.json')
intensity_map_params_macos_segfault = [
    [graphite_fc_file, '--weights=coherent', '--cmap=bone'],
    [graphite_fc_file, '--weighting=coherent', '--cmap=bone'],
    [graphite_fc_file, '--weighting=coherent', '--temperature=800']]

@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using
        # gcf()
        mocker.patch('matplotlib.pyplot.show')
        mocker.resetall()

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    def run_intensity_map_and_test_result(
            self, intensity_map_args):
        euphonic.cli.intensity_map.main(intensity_map_args)

        image_data = get_current_plot_image_data()

        with open(intensity_map_output_file, 'r') as expected_data_file:
            # Test deprecated --weights until it is removed
            key = args_to_key(intensity_map_args).replace(
                'weights', 'weighting')
            expected_image_data = json.load(expected_data_file)[key]
        for key, value in image_data.items():
            if key == 'extent':
                # Lower bound of y-data (energy) varies by up to ~2e-6 on
                # different systems when --asr is used, compared to
                # the upper bound of 100s of meV this is effectively zero,
                # so increase tolerance to allow for this
                npt.assert_allclose(value, expected_image_data[key], atol=2e-6)
            elif isinstance(value, list) and isinstance(value[0], float):
                # Errors of 2-4 epsilon seem to be common when using
                # broadening, so slightly increase tolerance
                npt.assert_allclose(value, expected_image_data[key],
                                    atol=1e-14)
            else:
                assert value == expected_image_data[key]

    @pytest.mark.parametrize(
        'intensity_map_args', intensity_map_params_macos_segfault)
    def test_intensity_map_image_data_macos_segfault(
            self, inject_mocks, intensity_map_args):
        self.run_intensity_map_and_test_result(intensity_map_args)
