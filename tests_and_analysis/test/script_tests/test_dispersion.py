import sys
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

from tests_and_analysis.test.utils import (
    get_data_path, get_castep_path, get_phonopy_path)
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_line_data, args_to_key)
import euphonic.cli.dispersion


cahgo2_fc_file = get_phonopy_path('CaHgO2', 'mp-7041-20180417.yaml')
lzo_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'LZO_force_constants.json')
nacl_fc_file = get_phonopy_path('NaCl_cli_test', 'force_constants.hdf5')
nacl_phonon_file = os.path.join(
    get_phonopy_path('NaCl', 'band'), 'band.yaml')
nacl_phonon_hdf5_file = os.path.join(
    get_phonopy_path('NaCl', 'band'), 'band.hdf5')
quartz_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'quartz',
    'quartz_bandstructure_qpoint_phonon_modes.json')
disp_output_file = os.path.join(get_script_test_data_path(), 'dispersion.json')

disp_params_macos_segfault =  [[cahgo2_fc_file, '--reorder']]


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

    def run_dispersion_and_test_result(self, dispersion_args):
        euphonic.cli.dispersion.main(dispersion_args)

        line_data = get_current_plot_line_data()

        with open(disp_output_file, 'r') as f:
            expected_line_data = json.load(f)[args_to_key(dispersion_args)]
        # Increase tolerance if asr present - can give slightly
        # different results with different libs
        if any(['--asr' in arg for arg in dispersion_args]):
            atol = 5e-6
        else:
            atol = sys.float_info.epsilon
        for key, value in line_data.items():
            if key == 'xy_data':
                # numpy can only auto convert 2D lists - xy_data has
                # dimensions (n_lines, 2, n_points) so check in a loop
                for idx, line in enumerate(value):
                    npt.assert_allclose(
                        line, expected_line_data[key][idx],
                        atol=atol)
            else:
                assert value == expected_line_data[key]


    @pytest.mark.parametrize('dispersion_args', disp_params_macos_segfault)
    def test_dispersion_plot_data_macos_segfault(
            self, inject_mocks, dispersion_args):
        self.run_dispersion_and_test_result(dispersion_args)
