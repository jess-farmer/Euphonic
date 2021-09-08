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

from tests_and_analysis.test.utils import get_data_path, get_castep_path, get_phonopy_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_image_data, args_to_key)
import euphonic.cli.powder_map

graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
nacl_prim_fc_file = get_phonopy_path('NaCl_prim', 'phonopy_nacl.yaml')
powder_map_output_file = os.path.join(get_script_test_data_path(),
                                      'powder-map.json')

quick_calc_params = ['--npts=10', '--npts-min=10', '--q-spacing=1']
powder_map_params_macos_segfault = [
    [nacl_prim_fc_file, '--temperature=1000', '--weights=coherent',
     *quick_calc_params],
    [nacl_prim_fc_file, '--temperature=1000', '--weighting=coherent',
     *quick_calc_params]]


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

    def run_powder_map_and_test_result(self, powder_map_args):
        euphonic.cli.powder_map.main(powder_map_args)

        image_data = get_current_plot_image_data()

        with open(powder_map_output_file, 'r') as expected_data_file:
            # Test deprecated --weights until it is removed
            key = args_to_key(powder_map_args).replace(
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

    @pytest.mark.parametrize('powder_map_args', powder_map_params_macos_segfault)
    def test_powder_map_plot_image_macos_segfault(
            self, inject_mocks, powder_map_args):
        self.run_powder_map_and_test_result(powder_map_args)
