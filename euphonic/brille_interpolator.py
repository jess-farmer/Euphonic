import spglib as spg
import numpy as np
import brille as br

from euphonic import ureg, QpointPhononModes, DebyeWaller, Quantity
from euphonic.debye_waller import _calculate_debye_waller

class BrilleInterpolator(object):
    """
    A class to perform linear interpolation of eigenvectors and
    frequencies at arbitrary q-points using the Brille library

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    """
    def __init__(self, force_constants, grid_type='mesh', grid_kwargs={},
                 interp_kwargs={}):

        crystal = force_constants.crystal
        cell_vectors = crystal._cell_vectors
        cell = crystal.to_spglib_cell()

        dataset = spg.get_symmetry_dataset(cell)
        rotations = dataset['rotations'] # in fractional
        translations = dataset['translations'] # in fractional

        symmetry = br.Symmetry(rotations, translations)
        direct = br.Direct(*cell)
        direct.spacegroup = symmetry
        bz = br.BrillouinZone(direct.star)

        if grid_type == 'mesh':
            grid = br.BZMeshQdc(bz, **grid_kwargs)
        elif grid_type == 'nest':
            grid = br.BZNestQdc(bz, **grid_kwargs)
        elif grid_type == 'trellis':
            grid = br.BZTrellisQdc(bz, **grid_kwargs)
        else:
            raise ValueError(f'Grid type "{grid_type}" not recognised')

        print(f'Grid generated with {len(grid.rlu)} q-points')
        interp_kwargs['insert_gamma'] = False
        interp_kwargs['reduce_qpts'] = False
        phonons = force_constants.calculate_qpoint_phonon_modes(
            grid.rlu, **interp_kwargs)
        print((f'Frequencies/eigenvectors calculated for '
               f'{len(grid.rlu)} q-points'))
        evecs_basis = np.einsum('ba,ijkb->ijka', np.linalg.inv(cell_vectors),
                                phonons.eigenvectors)
        n_atoms = crystal.n_atoms
        frequencies = np.reshape(phonons._frequencies,
                                 phonons._frequencies.shape + (1,))
        freq_el = (1,)
        freq_weight = (1., 0., 0.)
        evecs = np.reshape(evecs_basis,
                           (evecs_basis.shape[0], 3*n_atoms, 3*n_atoms))
        cost_function = 0
        evecs_el = (0, 3*n_atoms, 0, 3, 0, cost_function)
        evecs_weight = (0., 1., 0.)
        grid.fill(frequencies, freq_el, freq_weight, evecs, evecs_el,
                  evecs_weight)
        self._grid = grid
        self.crystal = crystal

    def calculate_qpoint_phonon_modes(self, qpts, **kwargs):
        vals, vecs = self._grid.ir_interpolate_at(qpts, **kwargs)
        vecs_cart = self._br_evec_to_eu(
            vecs, cell_vectors=self.crystal._cell_vectors)
        frequencies = vals.squeeze()*ureg('hartree').to('meV')
        return QpointPhononModes(
            self.crystal, qpts, frequencies, vecs_cart)

    def calculate_debye_waller(self,
                               temperature: Quantity,
                               frequency_min: Quantity = Quantity(0.01, 'meV'),
                               symmetrise: bool = True) -> DebyeWaller:

        grid = self._grid
        dw = _calculate_debye_waller(
            grid.rlu,
            grid.values.squeeze(),
            self._br_evec_to_eu(grid.vectors),
            temperature.to('K').magnitude,
            self.crystal,
            frequency_min=frequency_min,
            symmetrise=symmetrise)
        # Convert Debye-Waller from basis to Cartesian
        dw = np.einsum(
            'ab,ija->ijb', self.crystal._cell_vectors, dw)
        dw = np.einsum(
            'ab,iak->ibk', self.crystal._cell_vectors, dw)
        dw = dw*ureg('bohr**2').to(self.crystal.cell_vectors_unit + '**2')
        return DebyeWaller(self.crystal, dw, temperature)

    @staticmethod
    def _br_evec_to_eu(br_evecs, cell_vectors=None):
        n_branches = len(br_evecs[1])
        eu_evecs = br_evecs.view().reshape(-1, n_branches, n_branches//3, 3)
        if cell_vectors is not None:
            # Convert Brille evecs (stored in basis coordinates) to
            # Cartesian coordinates
            eu_evecs = np.einsum('ab,ijka->ijkb', cell_vectors, eu_evecs)
        return eu_evecs
