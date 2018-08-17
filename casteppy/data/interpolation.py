import math
import struct
import sys
import os
import numpy as np
from casteppy import ureg

class InterpolationData:
    """
    A class to read the data required for a supercell phonon interpolation
    calculation from a .castep_bin file, and store any calculated
    frequencies/eigenvectors

    Attributes
    ----------
    n_ions : int
        Number of ions in the unit cell
    n_branches : int
        Number of phonon dispersion branches
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors. Default units Angstroms.
    ion_r : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_r
    ion_mass : list of floats
        n_ions length list of the mass of each ion in the unit cell in atomic
        units
    n_cells_in_sc : int
        Number of cells in the supercell
    sc_matrix : list of floats
        3 x 3 list of the unit cell matrix
    cell_origins : list of floats
        K x 3 list of the locations of the unit cells within the supercell,
        where K = number of cells in the supercell
    force_constants : list of floats
        K x 3*L x 3*L list of the force constants, where K = number of cells
        in the supercell and L = number of ions. Default units atomic units
    n_qpts : int
        Number of q-points used in the most recent interpolation calculation.
        Default value is 0
    qpts : list of floats
        M x 3 list of coordinates of the q-points used for the most recent
        interpolation calculation, where M = number of q-points. Is empty by
        default
    freqs: list of floats
        M x N list of phonon frequencies from the most recent interpolation
        calculation, where M = number of q-points and N = number of branches.
        Default units eV. Is empty by default
    eigenvecs: list of complex floats
        M x N x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors) from the most recent interpolation calculation, where
        M = number of q-points, N = number of branches and L = number of ions.
        Is empty by default
    """


    def __init__(self, seedname, path='', qpts=np.array([])):
        """"
        Reads .castep_bin file, sets attributes, and calculates
        frequencies/eigenvectors at specific q-points if requested

        Parameters
        ----------
        seedname : str
            Name of .castep_bin file to read
        path : str, optional
            Path to dir containing the .castep_bin file, if it is in another 
            directory
        qpts : list of floats, optional
            M x 3 list of q-point coordinates to use for an initial
            interpolation calculation
        """
        self._get_data(seedname, path)

        self.qpts = qpts
        self.n_qpts = len(qpts)
        self.eigenvecs = np.array([])
        energy_units = '{}'.format(self.force_constants.units).split('/')[0]
        self.freqs = np.array([])*ureg[energy_units]

        if self.n_qpts > 0:
            self.calculate_fine_phonons(qpts)


    def _get_data(self, seedname, path):
        """"
        Opens .castep_bin file for reading

        Parameters
        ----------
        seedname : str
            Name of .castep_bin file to read
        path : str
            Path to dir containing the .castep_bin file, if it is in another 
            directory
        """
        file = os.path.join(path, seedname + '.castep_bin')
        with open(file, 'rb') as f:
            self._read_interpolation_data(f)


    def _read_interpolation_data(self, file_obj):
        """
        Reads data from .castep_bin file and sets attributes

        Parameters
        ----------
        f : file object
            File object in read mode for the .castep_bin file containing the
            data
        """

        def read_entry(file_obj, dtype=''):
            """
            Read a record from a Fortran binary file, including the beginning
            and end record markers and the data inbetween
            """
            def record_mark_read(file_obj):
                # Read 4 byte Fortran record marker
                return struct.unpack('>i', file_obj.read(4))[0]

            begin = record_mark_read(file_obj)
            if dtype:
                n_bytes = int(dtype[-1])
                n_elems = int(begin/n_bytes)
                if n_elems > 1:
                    data = np.fromfile(file_obj, dtype=dtype, count=n_elems)
                else:
                    if 'i' in dtype:
                        data = struct.unpack('>i', file_obj.read(begin))[0]
                    elif 'f' in dtype:
                        data = struct.unpack('>d', file_obj.read(begin))[0]
            else:
                data = file_obj.read(begin)
            end = record_mark_read(file_obj)
            if begin != end:
                sys.exit("""Problem reading binary file: beginning and end
                            record markers do not match""")

            return data

        int_type = '>i4'
        float_type = '>f8'

        header = ''
        while header.strip() != b'END':
            header = read_entry(file_obj)
            if header.strip() == b'CELL%NUM_IONS':
                n_ions = read_entry(file_obj, int_type)
            elif header.strip() == b'CELL%REAL_LATTICE':
                cell_vec = np.transpose(np.reshape(
                    read_entry(file_obj, float_type), (3, 3)))
            elif header.strip() == b'CELL%NUM_SPECIES':
                n_species= read_entry(file_obj, int_type)
            elif header.strip() == b'CELL%NUM_IONS_IN_SPECIES':
                n_ions_in_species = read_entry(file_obj, int_type)
            elif header.strip() == b'CELL%IONIC_POSITIONS':
                max_ions_in_species = max(n_ions_in_species)
                ion_r_tmp = np.reshape(read_entry(file_obj, float_type),
                                  (n_species, max_ions_in_species, 3))
            elif header.strip() == b'CELL%SPECIES_MASS':
                ion_mass_tmp = read_entry(file_obj, float_type)
            elif header.strip() == b'CELL%SPECIES_SYMBOL':
                ion_type_tmp = [x.strip() for x in read_entry(file_obj, 'S8')]
            elif header.strip() == b'FORCE_CON':
                sc_matrix = np.reshape(
                    read_entry(file_obj, int_type), (3, 3))
                n_cells_in_sc = int(np.absolute(np.linalg.det(sc_matrix)))
                force_constants = np.reshape(
                    read_entry(file_obj, float_type),
                    (n_cells_in_sc, 3*n_ions, 3*n_ions))
                cell_origins = np.reshape(
                    read_entry(file_obj, int_type), (n_cells_in_sc, 3))
                fc_row = read_entry(file_obj, int_type)

        # Get ion_r in correct form
        # CASTEP stores ion positions as 3D array (3,
        # max_ions_in_species, n_species) so need to slice data to get
        # correct information
        ion_begin = np.insert(np.cumsum(n_ions_in_species[:-1]), 0, 0)
        ion_end = np.cumsum(n_ions_in_species)
        ion_r = np.zeros((n_ions, 3))
        for i in range(n_species):
                ion_r[ion_begin[i]:ion_end[i], :] = ion_r_tmp[
                    i,:n_ions_in_species[i], :]
        # Get ion_type in correct form
        ion_type = []
        ion_mass = []
        for ion in range(n_species):
            ion_type.extend([ion_type_tmp[ion] for i in range(n_ions_in_species[ion])])
            ion_mass.extend([ion_mass_tmp[ion] for i in range(n_ions_in_species[ion])])

        cell_vec = cell_vec*ureg.bohr
        cell_vec.ito('angstrom')
        ion_mass = ion_mass*ureg.e_mass
        ion_mass.ito('amu')
        force_constants = force_constants*ureg.hartree/(ureg.bohr**2)

        self.n_ions = n_ions
        self.n_branches = 3*n_ions
        self.cell_vec = cell_vec
        self.ion_r = ion_r
        self.ion_type = ion_type
        self.ion_mass = ion_mass
        self.sc_matrix = sc_matrix
        self.n_cells_in_sc = n_cells_in_sc
        self.force_constants = force_constants
        self.cell_origins = cell_origins


    def _calculate_supercell_images(self, lim):
        n_ions = self.n_ions
        cell_vec = self.cell_vec.to(ureg.bohr).magnitude
        ion_r = self.ion_r
        cell_origins = self.cell_origins
        n_cells_in_sc = self.n_cells_in_sc
        sc_matrix = self.sc_matrix

        # List of points defining Wigner-Seitz cell
        ws_frac = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                            [0, 1, -1], [1, 0, 0], [1, 0, 1], [1, 0, -1],
                            [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0],
                            [1, -1, 1], [1, -1, -1]])
        cutoff_scale = 1.0

        # Calculate points of WS cell for this supercell
        sc_vecs = np.matmul(sc_matrix, cell_vec)
        ws_list = np.matmul(ws_frac, sc_vecs)
        inv_ws_sq = 1.0/np.sum(np.square(ws_list[1:]), axis=1)

        # Get Cartesian coords of supercell images and ions in supercell
        sc_image_r = self._calculate_supercell_image_r(lim)
        sc_image_cart = np.matmul(sc_image_r, np.transpose(sc_vecs))
        sc_ion_r = np.matmul(np.tile(ion_r, (n_cells_in_sc, 1)) + np.repeat(cell_origins, n_ions, axis=0), np.linalg.inv(sc_matrix))
        sc_ion_cart = np.zeros((len(sc_image_r), 3))
        for i in range(n_ions*n_cells_in_sc):
            sc_ion_cart[i, :] = np.matmul(sc_ion_r[i, :], sc_vecs)

        sc_image_i = np.zeros((self.n_ions, self.n_ions*self.n_cells_in_sc, (2*lim + 1)**3), dtype=np.int8)
        n_sc_images = np.zeros((self.n_ions, self.n_ions*self.n_cells_in_sc), dtype=np.int8)
        for i in range(n_ions):
            for j in range(n_ions*n_cells_in_sc):
                # Get distance between ions in every supercell
                dist = sc_ion_cart[i] - sc_ion_cart[j] - sc_image_cart
                for k in range(len(sc_image_r)):
                    dist_ws_point = np.absolute(np.dot(ws_list[1:], dist[k, :])*inv_ws_sq)
                    if np.max(dist_ws_point) <= (0.5*cutoff_scale + 0.001):
                        sc_image_i[i, j, n_sc_images[i, j]] = k
                        n_sc_images[i, j] += 1

        self.sc_image_i = sc_image_i
        self.n_sc_images = n_sc_images


    def _calculate_supercell_image_r(self, lim):
        irange = range(-lim, lim + 1)
        inum = 2*lim + 1
        scx = np.repeat(irange, inum**2)
        scy = np.tile(np.repeat(irange, inum), inum)
        scz = np.tile(irange, inum**2)
        return np.column_stack((scx, scy, scz))


    def calculate_fine_phonons(self, qpts):
        force_constants = self.force_constants.magnitude
        sc_matrix = self.sc_matrix
        cell_origins = self.cell_origins
        n_cells_in_sc = self.n_cells_in_sc
        n_ions = self.n_ions
        n_branches = self.n_branches
        n_qpts = qpts.shape[0]
        freqs = np.zeros((n_qpts, n_branches))
        eigenvecs = np.zeros((n_qpts, n_branches, n_ions, 3), dtype=np.complex128)

        # Build list of all possible supercell image coordinates
        lim = 2 # Supercell image limit
        sc_image_r = self._calculate_supercell_image_r(lim)
        phases = np.zeros((n_cells_in_sc, (2*lim + 1)**3), dtype=np.complex128)

        if not hasattr(self, 'sc_image_i'):
            self._calculate_supercell_images(lim)
        n_sc_images = self.n_sc_images
        sc_image_i = self.sc_image_i

        # Calculate phase lookup
        lookup_i = range(2*lim + 1)
        for q in range(n_qpts):
            qpt = qpts[q, :]
            dyn_mat = np.zeros((n_ions*3, n_ions*3), dtype=np.complex128)
            for i in range(len(sc_image_r)):
                sc_offset = np.dot(np.transpose(sc_matrix), sc_image_r[i, :])
                for nc in range(n_cells_in_sc):
                    cell_r = sc_offset + cell_origins[nc, :]
                    phase = 2*math.pi*np.dot(qpt, cell_r)
                    phases[nc, i] = np.complex(math.cos(phase), math.sin(phase))

            for i in range(n_ions):
                for scj in range(n_ions*n_cells_in_sc):
                    j = int(scj%n_ions)
                    nc = int(scj/n_ions)
                    # Cumulant method: sum phases for all supercell images and divide by number of images
                    term = 0.
                    for img_i in range(n_sc_images[i, scj]):
                        term = term + phases[nc, sc_image_i[i, scj, img_i]]
                    dyn_mat[3*i:(3*i + 3), 3*j:(3*j + 3)] = dyn_mat[3*i:(3*i + 3), 3*j:(3*j + 3)] + (term*force_constants[nc, 3*i:(3*i + 3), 3*j:(3*j + 3)])/n_sc_images[i, scj]

            evals, evecs = np.linalg.eigh(dyn_mat)
            eigenvecs[q, :] = np.reshape(evecs, (n_branches, n_ions, 3))
            freqs[q, :] = np.sqrt(evals)

        self.n_qpts = n_qpts
        self.qpts = qpts
        self.freqs = freqs*self.freqs.units
        self.eigenvecs = eigenvecs

        return self.freqs, self.eigenvecs

    def convert_e_units(self, units):
        super(InterpolationData, self).convert_e_units(units)

        if hasattr(self, 'freqs'):
            self.freqs.ito(units, 'spectroscopy')
