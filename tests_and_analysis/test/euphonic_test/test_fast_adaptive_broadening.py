import os
import json
import pytest
import numpy as np
from scipy.integrate import simps
from scipy.stats import norm

from euphonic.fast_adaptive_broadening import gaussian, find_coeffs
from euphonic import ureg, Crystal, QpointFrequencies, Spectrum1D
from tests_and_analysis.test.euphonic_test.test_force_constants import get_fc_dir
from tests_and_analysis.test.utils import get_data_path


def get_qpt_freqs_dir(material):
    return os.path.join(get_data_path(), 'qpoint_frequencies', material)

def get_qpt_freqs(material, file):
    return QpointFrequencies.from_json_file(
        os.path.join(get_qpt_freqs_dir(material), file))

@pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json, ebins'), [
            ('quartz', 'quartz_554_full_qpoint_frequencies.json',
             'quartz_554_full_mode_widths.json',
             np.arange(0, 155, 0.1)*ureg('meV')),
            ('LZO', 'lzo_222_full_qpoint_frequencies.json',
             'lzo_222_full_mode_widths.json',
             np.arange(0, 100, 0.1)*ureg('meV'))])
def test_area_unchanged_for_broadened_dos(material, qpt_freqs_json, mode_widths_json, ebins):
    """Test that the area is approximately equal for unbroadened and broadened dos"""
    qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
    with open(os.path.join(get_fc_dir(), mode_widths_json), 'r') as fp:
        modw_dict = json.load(fp)
    mode_widths = modw_dict['mode_widths']*ureg(
        modw_dict['mode_widths_unit'])
    dos = qpt_freqs._calculate_dos(ebins)
    adaptively_broadened_dos = qpt_freqs._calculate_dos(
        ebins, mode_widths=mode_widths, adaptive_method='fast')
    ebins_centres = Spectrum1D._bin_edges_to_centres(ebins.to('hartree').magnitude)
    dos_area = simps(dos, ebins_centres)
    adaptively_broadened_dos_area = simps(adaptively_broadened_dos, ebins_centres)
    assert adaptively_broadened_dos_area == pytest.approx(dos_area, abs=1e-4)

def test_gaussian():
    """Test gaussian function against scipy.norm.pdf"""
    xvals = np.linspace(-5,5,101)
    sigma = 2
    centre = 0
    scipy_norm = norm.pdf(xvals, loc=centre, scale=sigma)
    gaussian_eval = gaussian(xvals, sigma, centre)
    assert gaussian_eval == pytest.approx(scipy_norm)

@pytest.mark.parametrize(('spacing','expected_coeffs'),
                         [(2, [-0.14792773,  1.19816561, -3.55951241,  3.50998823]),
                          (np.sqrt(2), [-0.39948008,  2.97409516, -7.83100669,  6.25652513])])
def test_find_coeffs(spacing, expected_coeffs):
    """Test find_coeffs against coefficients calculated on a fine grid"""
    coeffs = find_coeffs(spacing)
    assert coeffs == pytest.approx(expected_coeffs, abs=0.015)