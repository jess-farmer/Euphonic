import pytest
import numpy as np
from scipy.integrate import simps
from scipy.stats import norm

from euphonic.fast_adaptive_broadening import find_coeffs, gaussian
from euphonic import ureg, Spectrum1D
from tests_and_analysis.test.euphonic_test.test_force_constants\
    import get_fc_path
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies\
    import get_qpt_freqs
from tests_and_analysis.test.utils import get_mode_widths
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_expected_spectrum1d, check_spectrum1d)


@pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json, ebins'), [
            ('quartz', 'quartz_554_full_qpoint_frequencies.json',
             'quartz_554_full_mode_widths.json',
             np.arange(0, 155, 0.1)*ureg('meV')),
            ('LZO', 'lzo_222_full_qpoint_frequencies.json',
             'lzo_222_full_mode_widths.json',
             np.arange(0, 100, 0.1)*ureg('meV'))])
def test_area_unchanged_for_broadened_dos(material, qpt_freqs_json,
                                          mode_widths_json, ebins):
    """
    Test that the area is approximately equal for unbroadened
    and broadened dos
    """
    qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
    mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
    dos = qpt_freqs._calculate_dos(ebins)
    adaptively_broadened_dos = qpt_freqs._calculate_dos(
        ebins, mode_widths=mode_widths, adaptive_method='fast')
    ebins_centres = \
        Spectrum1D._bin_edges_to_centres(ebins.to('hartree').magnitude)
    dos_area = simps(dos, ebins_centres)
    adaptively_broadened_dos_area = simps(adaptively_broadened_dos,
                                          ebins_centres)
    assert adaptively_broadened_dos_area == pytest.approx(dos_area, abs=1e-3)

@pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json,'
         'expected_dos_json, ebins'), [
            ('quartz', 'toy_quartz_qpoint_frequencies.json',
             'toy_quartz_mode_widths.json',
             'toy_quartz_adaptive_dos.json',
             np.arange(0, 40, 0.1)*ureg('meV'))])
def test_lower_bound_widths_broadened(material, qpt_freqs_json,
                                      mode_widths_json,
                                      expected_dos_json, ebins):
    qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
    mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
    dos = qpt_freqs.calculate_dos(ebins, mode_widths=mode_widths,
                                   adaptive_method='fast')
    expected_dos = get_expected_spectrum1d(expected_dos_json)  
    check_spectrum1d(dos, expected_dos, tol=1e-13)                         

def test_gaussian():
    """Test gaussian function against scipy.norm.pdf"""
    xvals = np.linspace(-5,5,101)
    sigma = 2
    scipy_norm = norm.pdf(xvals, scale=sigma)
    gaussian_eval = gaussian(xvals, sigma)
    assert gaussian_eval == pytest.approx(scipy_norm)

@pytest.mark.parametrize(('spacing','expected_coeffs'),
    [(2, [-0.1883858, 1.46930932, -4.0893793, 3.80872458]),
    (np.sqrt(2), [-0.60874207, 4.10599029, -9.63986481, 7.14267836])])
def test_find_coeffs(spacing, expected_coeffs):
    """Test find_coeffs against expected coefficients"""
    coeffs = find_coeffs(spacing)
    assert coeffs == pytest.approx(expected_coeffs)
