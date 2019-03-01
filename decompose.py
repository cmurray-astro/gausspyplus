# @Author: riener
# @Date:   2019-02-08T15:40:10+01:00
# @Filename: decompose.py
# @Last modified by:   riener
# @Last modified time: 2019-03-01T14:43:57+01:00

from __future__ import print_function

import logging
import os
import pickle
import sys
import warnings

import numpy as np

from datetime import datetime
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from gausspyplus.shared_functions import gaussian, area_of_gaussian, combined_gaussian
from gausspyplus.spectral_cube_functions import change_header, save_fits, correct_header, update_header

# if (sys.version_info >= (3, 0)):
#     raise Exception('Script has to be run in Python 2 environment.')


class GaussPyDecompose(object):
    """Wrapper for decomposition with GaussPy+.

    Parameters
    ----------
    pathToPickleFile : str
        Filepath to the pickled dictionary produced by GaussPyPrepare.

    Attributes
    ----------
    dirname : str
        Path to the directory containing the pickled dictionary produced by GaussPyPrepare.
    file : str
        Filename and extension of the pickled dictionary produced by GaussPyPrepare.
    filename : str
        Filename of the pickled dictionary produced by GaussPyPrepare.
    fileExtension : str
        Extension of the pickled dictionary produced by GaussPyPrepare.
    decompDirname : str
        Path to directory in which decomposition results are saved.
    parentDirname : str
        Parent directory of 'gpy_prepared'.
    gaussPyDecomposition : bool
        'True' if data should be decomposed. 'False' if decomposition results are loaded.
    twoPhaseDecomposition : bool
        'True' (default) uses two smoothing parameters (alpha1, alpha2) for the decomposition. 'False' uses only the alpha1 smoothing parameter.
    trainingSet : bool
        Default is 'False'. Set to 'True' if training set is decomposed.
    saveInitialGuesses : bool
        Default is 'False'. Set to 'True' if initial GaussPy fitting guesses should be saved.
    alpha1 : float
        First smoothing parameter.
    alpha2 : float
        Second smoothing parameter. Only used if twoPhaseDecomposition is set to 'True'
    snrThresh : float
        S/N threshold used for the original spectrum.
    snr2Thresh : float
        S/N threshold used for the second derivate of the smoothed spectrum.
    useCpus : int
        Number of CPUs used in the decomposition. By default 75% of all CPUs on the machine are used.
    fitting : dct
        Description of attribute `fitting`.
    main_beam_efficiency : float
        Default is 'None'. Specify if intensity values should be corrected by the main beam efficiency.
    vel_unit : astropy.units
        Default is 'u.km/u.s'. Unit to which velocity values will be converted.
    testing : bool
        Default is 'False'. Set to 'True' if in testing mode.
    verbose : bool
        Default is 'True'. Set to 'False' if descriptive statements should not be printed in the terminal.
    suffix : str
        Suffix for filename of the decomposition results.
    removeHeaderKeywords : list
        Specify keywords that should be removed from the header.
    headerComments : list
        Specify comments that should be included in the header.
    restoreNans : bool
        Default is 'True'. Set to 'False' if NaN values should not be restored in the FITS files created from the decomposition results.
    log_output : bool
        Default is 'True'. Set to 'False' if terminal output should not be logged.
    """
    def __init__(self, pathToPickleFile):
        self.pathToPickleFile = pathToPickleFile
        self.dirname = os.path.dirname(pathToPickleFile)
        self.file = os.path.basename(pathToPickleFile)
        self.filename, self.fileExtension = os.path.splitext(self.file)
        self.decompDirname = None
        self.parentDirname = os.path.dirname(self.dirname)

        self.gaussPyDecomposition = True
        self.twoPhaseDecomposition = True
        self.trainingSet = False
        self.saveInitialGuesses = False
        self.alpha1, self.alpha2, self.snrThresh, self.snr2Thresh, self.useCpus = (
                None for i in range(5))
        self.fitting = {
            'improve_fitting': False, 'min_fwhm': 1., 'max_fwhm': None,
            'min_offset': 2., 'snr': 3., 'snr_fit': None, 'significance': 5.,
            'snr_negative': None, 'rchi2_limit': 1.5, 'max_amp_factor': 1.1,
            'negative_residual': True, 'broad': True, 'blended': True, 'fwhm_factor': 1.5}
        self.main_beam_efficiency = None
        self.vel_unit = u.km / u.s
        self.testing = False
        self.verbose = True
        self.suffix = ''
        self.removeHeaderKeywords = []
        self.headerComments = None
        self.restoreNans = True
        self.log_output = True

    def set_up_logger(self):
        #  setting up logger
        now = datetime.now()
        date_string = "{}{}{}-{}{}{}".format(
            now.year,
            str(now.month).zfill(2),
            str(now.day).zfill(2),
            str(now.hour).zfill(2),
            str(now.minute).zfill(2),
            str(now.second).zfill(2))

        dirname = os.path.join(self.parentDirname, 'gpy_log')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.splitext(os.path.basename(self.filename))[0]

        logname = os.path.join(dirname, '{}_decompose_{}.log'.format(
            date_string, filename))
        logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        self.logger = logging.getLogger(__name__)

    def getting_ready(self):
        if self.log_output:
            self.set_up_logger()

        string = 'GaussPy decomposition'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        self.say(heading)

    def say(self, message):
        """Diagnostic messages."""
        if self.log_output:
            self.logger.info(message)
        if self.verbose:
            print(message)

    def initialize_data(self):
        self.check_settings()
        self.getting_ready()

        self.say("\npickle load '{}'...".format(self.file))

        with open(self.pathToPickleFile, "rb") as pickle_file:
            self.pickledData = pickle.load(pickle_file, encoding='latin1')

        # TODO: check what consequences it has if one of those keywords is missing
        if 'header' in self.pickledData.keys():
            self.header = self.pickledData['header']
        if 'location' in self.pickledData.keys():
            self.location = self.pickledData['location']
        if 'nan_mask' in self.pickledData.keys():
            self.nan_mask = self.pickledData['nan_mask']
        if 'testing' in self.pickledData.keys():
            self.testing = self.pickledData['testing']

        self.data = self.pickledData['data_list']
        self.channels = self.pickledData['x_values']
        self.errors = self.pickledData['error']

        self.header = correct_header(self.header)
        self.wcs = WCS(self.header)

        self.velocity_increment = (
            self.wcs.wcs.cdelt[2] * self.wcs.wcs.cunit[2]).to(
                self.vel_unit).value

        if self.decompDirname is None:
            decompDirname = os.path.normpath(
                    self.dirname + os.sep + os.pardir)
            self.decompDirname = os.path.join(decompDirname, 'gpy_decomposed')
            if not os.path.exists(self.decompDirname):
                os.makedirs(self.decompDirname)

    def check_settings(self):
        if self.gaussPyDecomposition and (self.alpha1 is None or
                                          self.snrThresh is None or
                                          self.snr2Thresh is None):
            errorMessage = \
                """gaussPyDecomposition needs alpha1, snrThresh and
                snr2Thresh values"""
            raise Exception(errorMessage)

        if self.gaussPyDecomposition and self.twoPhaseDecomposition and \
                (self.alpha2 is None):
                    errorMessage = \
                        """twoPhaseDecomposition needs alpha2 value"""
                    raise Exception(errorMessage)

        if self.fitting['snr_negative'] is None:
            self.fitting['snr_negative'] = self.fitting['snr']

        if self.fitting['snr_fit'] is None:
            self.fitting['snr_fit'] = self.fitting['snr'] / 2.

        if self.main_beam_efficiency is None:
            warnings.warn('assuming intensities are already corrected for  main beam efficiency')

        warnings.warn("converting velocity values to {}".format(
            self.vel_unit))

    def decompose(self):
        self.initialize_data()

        if self.gaussPyDecomposition:
            self.gaussPy_decomposition()

    def decomposition_settings(self):
        string_gausspy = str(
            '\ndecomposition settings:'
            '\nGaussPy:'
            '\nTwo phase decomposition: {a}'
            '\nalpha1: {b}'
            '\nalpha2: {c}'
            '\nSNR1: {d}'
            '\nSNR2: {e}').format(
                a=self.twoPhaseDecomposition,
                b=self.alpha1,
                c=self.alpha2,
                d=self.snrThresh,
                e=self.snr2Thresh)
        self.say(string_gausspy)

        string_gausspy_plus = ''
        if self.fitting['improve_fitting']:
            #  TODO: change to items() in Python 3
            for key, value in self.fitting.iteritems():
                string_gausspy_plus += str('\n{}: {}').format(key, value)
        else:
            string_gausspy_plus += str(
                '\nimprove_fitting: {}').format(
                    self.fitting['improve_fitting'])
        self.say(string_gausspy_plus)

    def gaussPy_decomposition(self):
        self.decomposition_settings()
        self.say('\ndecomposing data...')

        from .gausspy_py3 import gp as gp
        g = gp.GaussianDecomposer()  # Load GaussPy
        g.set('useCpus', self.useCpus)
        g.set('SNR_thresh', self.snrThresh)
        g.set('SNR2_thresh', self.snr2Thresh)
        g.set('improve_fitting_dict', self.fitting)
        g.set('alpha1', self.alpha1)

        if not self.trainingSet:
            if self.testing:
                g.set('verbose', True)
                g.set('plot', True)

        if self.twoPhaseDecomposition:
            g.set('phase', 'two')
            g.set('alpha2', self.alpha2)
        else:
            g.set('phase', 'one')

        self.decomposition = g.batch_decomposition(self.pathToPickleFile)

        self.save_final_results()

        if self.saveInitialGuesses:
            self.save_initial_guesses()

    def save_initial_guesses(self):
        self.say('\npickle dump GaussPy initial guesses...')

        filename = '{}{}_fit_ini.pickle'.format(self.filename, self.suffix)
        pathname = os.path.join(self.decompDirname, filename)

        dct_initial_guesses = {}

        for key in ["index_initial", "amplitudes_initial",
                    "fwhms_initial", "means_initial"]:
            dct_initial_guesses[key] = self.decomposition[key]

        pickle.dump(dct_initial_guesses, open(pathname, 'w'))
        self.say(">> saved as '{}' in {}".format(filename, self.decompDirname))

    def save_final_results(self):
        self.say('\npickle dump GaussPy final results...')

        dct_gausspy_settings = {"two_phase": self.twoPhaseDecomposition,
                                "alpha1": self.alpha1,
                                "snr1_thresh": self.snrThresh,
                                "snr2_thresh": self.snr2Thresh}

        if self.twoPhaseDecomposition:
            dct_gausspy_settings["alpha2"] = self.alpha2

        dct_final_guesses = {}

        for key in ["index_fit", "best_fit_rchi2", "best_fit_aicc",
                    "amplitudes_fit", "amplitudes_fit_err", "fwhms_fit",
                    "fwhms_fit_err", "means_fit", "means_fit_err", "log_gplus",
                    "N_negative_residuals", "N_blended", "N_components"]:
            dct_final_guesses[key] = self.decomposition[key]

        dct_final_guesses["gausspy_settings"] = dct_gausspy_settings

        if not self.trainingSet:
            dct_final_guesses["improve_fit_settings"] = self.fitting
        else:
            dct_final_guesses["header"] = self.header
            dct_final_guesses["location"] = self.location
            dct_final_guesses["data_list"] = self.data
            dct_final_guesses["error"] = self.errors

        filename = '{}{}_fit_fin.pickle'.format(self.filename, self.suffix)
        pathname = os.path.join(self.decompDirname, filename)
        pickle.dump(dct_final_guesses, open(pathname, 'w'))
        self.say(">> saved as '{}' in {}".format(filename, self.decompDirname))

    def load_final_results(self, pathToDecomp):
        self.initialize_data()
        self.say('\npickle load final GaussPy results...')

        self.decompDirname = os.path.dirname(pathToDecomp)
        with open(pathToDecomp, "rb") as pickle_file:
            self.decomposition = pickle.load(pickle_file, encoding='latin1')

        self.file = os.path.basename(pathToDecomp)
        self.filename, self.fileExtension = os.path.splitext(self.file)

        if 'header' in self.decomposition.keys():
            self.header = self.decomposition['header']
        if 'channels' in self.decomposition.keys():
            self.channels = self.decomposition['channels']
        if 'nan_mask' in self.pickledData.keys():
            self.nan_mask = self.pickledData['nan_mask']
        if 'location' in self.pickledData.keys():
            self.location = self.pickledData['location']

    # def change_fits_header(self):
    #     import getpass
    #     import socket
    #
    #     if 'COMMENT' in self.header.keys():
    #         for i in range(len(self.header['COMMENT'])):
    #             self.header.remove('COMMENT')
    #
    #     for keyword in self.removeHeaderKeywords:
    #         if keyword in self.header.keys():
    #             self.header.remove(keyword)
    #
    #     self.header['AUTHOR'] = getpass.getuser()
    #     self.header['ORIGIN'] = socket.gethostname()
    #     self.header['DATE'] = (datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'), '(GMT)')
    #
    #     if self.headerComments is not None:
    #         self.header['COMMENT'] = self.headerComments
    #
    #     if self.gaussPyDecomposition:
    #         card = fits.Card('hierarch SNR_1', self.snrThresh,
    #                          'threshold for GaussPy decomposition')
    #         self.header.append(card)
    #
    #         card = fits.Card('hierarch SNR_2', self.snr2Thresh,
    #                          'threshold for GaussPy decomposition')
    #         self.header.append(card)
    #
    #         text = str('GaussPy parameter for decomposition')
    #         card = fits.Card('hierarch ALPHA1', self.alpha1, text)
    #         self.header.append(card)
    #
    #         if self.twoPhaseDecomposition:
    #             text = str('2nd parameter for 2-phase decomposition')
    #             card = fits.Card('hierarch ALPHA2', self.alpha2, text)
    #             self.header.append(card)

    def make_cube(self, mode='full_decomposition'):
        """Create FITS cube of the decomposition results.

        Parameters
        ----------
        mode : str
            'full_decomposition' recreates the whole FITS cube, 'integrated_intensity' creates a cube with the integrated intensity values of the Gaussian components placed at their mean positions, 'main_component' only retains the fitted component with the largest amplitude value
        """
        self.say('\ncreate {} cube...'.format(mode))

        x = self.header['NAXIS1']
        y = self.header['NAXIS2']
        z = self.header['NAXIS3']

        array = np.zeros([z, y, x], dtype=np.float32)
        nSpectra = len(self.decomposition['N_components'])

        for idx in xrange(nSpectra):
            ncomps = self.decomposition['N_components'][idx]
            if ncomps is None:
                continue

            yi = self.location[idx][0]
            xi = self.location[idx][1]

            amps = self.decomposition['amplitudes_fit'][idx]
            fwhms = self.decomposition['fwhms_fit'][idx]
            means = self.decomposition['means_fit'][idx]

            if self.main_beam_efficiency is not None:
                amps = [amp / self.main_beam_efficiency for amp in amps]

            if mode == 'main_component' and ncomps > 0:
                j = amps.index(max(amps))
                array[:, yi, xi] = gaussian(
                    amps[j], fwhms[j], means[j], self.channels)
            elif mode == 'integrated_intensity' and ncomps > 0:
                for j in xrange(ncomps):
                    integrated_intensity = area_of_gaussian(
                        amps[j], fwhms[j] * self.velocity_increment)
                    channel = int(round(means[j]))
                    if self.channels[0] <= channel <= self.channels[-1]:
                        array[channel, yi, xi] += integrated_intensity
            elif mode == 'full_decomposition':
                array[:, yi, xi] = combined_gaussian(
                    amps, fwhms, means, self.channels)

            nans = self.nan_mask[:, yi, xi]
            array[:, yi, xi][nans] = np.NAN

        if mode == 'main_component':
            comment = str('Fitted Gaussians from GaussPy decomposition, '
                          'per spectrum only Gaussian with highest '
                          'amplitude is included')
            filename = "{}{}_main.fits".format(self.filename, self.suffix)
        elif mode == 'integrated_intensity':
            comment = str('integrated intensity of Gaussian components '
                          'from GaussPy decomposition at VLSR positions')
            filename = "{}{}_wco.fits".format(self.filename, self.suffix)
        elif mode == 'full_decomposition':
            comment = 'Fitted Gaussians of GaussPy decomposition'
            filename = "{}{}_decomp.fits".format(self.filename, self.suffix)

        array[self.nan_mask] = np.nan

        comments = [comment]
        if self.gaussPyDecomposition:
            for name, value in zip(
                    ['SNR_1', 'SNR_2', 'ALPHA1'],
                    [self.snrThresh, self.snr2Thresh, self.alpha1]):
                comments.append('GaussPy+ parameter {}={}'.format(name, value))

            if self.twoPhaseDecomposition:
                comments.append('GaussPy+ parameter {}={}'.format(
                    'ALPHA2', self.alpha2))

        self.header = update_header(
            self.header, comments=comments, write_meta=True)

        pathToFile = os.path.join(self.decompDirname, 'FITS', filename)
        save_fits(array, self.header, pathToFile, verbose=False)
        self.say('>> saved {} in {}'.format(
            filename, os.path.dirname(pathToFile)))

    def create_input_table(self, ncomps_max=None):
        """Create a table of the decomposition results.

        The table contains the following columns:
        {0}: Pixel position in X direction
        {1}: Pixel position in Y direction
        {2}: Pixel position in Z direction
        {3}: Amplitude value of fitted Gaussian component
        {4}: Root-mean-square noise of the spectrum
        {5}: Velocity dispersion value of fitted Gaussian component
        {6}: Integrated intensity value of fitted Gaussian component
        {7}: Coordinate position in X direction
        {8}: Coordinate position in Y direction
        {9}: Mean position (VLSR) of fitted Gaussian component
        {10}: Error of amplitude value
        {11}: Error of velocity dispersion value
        {12}: Error of velocity value
        {13}: Error of integrated intensity value

        Amplitude and RMS values get corrected by the main_beam_efficiency parameter in case it was supplied.

        The table is saved in the 'gpy_tables' directory.

        Parameters
        ----------
        ncomps_max : int
            All spectra whose number of fitted components exceeds this value will be neglected.
        """
        self.say('\ncreate input table...')

        length = len(self.decomposition['amplitudes_fit'])

        x_pos, y_pos, z_pos, amp, rms, vel_disp, int_tot, x_coord, y_coord,\
            velocity, e_amp, e_vel_disp, e_velocity, e_int_tot = (
                [] for i in range(14))

        for idx in xrange(length):
            ncomps = self.decomposition['N_components'][idx]

            #  do not continue if spectrum was masked out, was not fitted,
            #  or was fitted by too many components
            if ncomps is None:
                continue
            elif ncomps == 0:
                continue
            elif ncomps_max is not None:
                if ncomps > ncomps_max:
                    continue

            yi, xi = self.location[idx]
            fit_amps = self.decomposition['amplitudes_fit'][idx]
            fit_fwhms = self.decomposition['fwhms_fit'][idx]
            fit_means = self.decomposition['means_fit'][idx]
            fit_e_amps = self.decomposition['amplitudes_fit_err'][idx]
            fit_e_fwhms = self.decomposition['fwhms_fit_err'][idx]
            fit_e_means = self.decomposition['means_fit_err'][idx]
            error = self.errors[idx][0]

            if self.main_beam_efficiency is not None:
                fit_amps = [
                    amp / self.main_beam_efficiency for amp in fit_amps]
                fit_e_amps = [
                    e_amp / self.main_beam_efficiency for e_amp in fit_e_amps]
                error /= self.main_beam_efficiency

            for j in xrange(ncomps):
                amp_value = fit_amps[j]
                e_amp_value = fit_e_amps[j]
                fwhm_value = fit_fwhms[j] * self.velocity_increment
                e_fwhm_value = fit_e_fwhms[j] * self.velocity_increment
                mean_value = fit_means[j]
                e_mean_value = fit_e_means[j]

                channel = int(round(mean_value))
                if channel < self.channels[0] or channel > self.channels[-1]:
                    continue

                x_wcs, y_wcs, z_wcs = self.wcs.wcs_pix2world(
                    xi, yi, mean_value, 0)

                x_pos.append(xi)
                y_pos.append(yi)
                z_pos.append(channel)
                rms.append(error)

                amp.append(amp_value)
                e_amp.append(e_amp_value)

                velocity.append(
                    (z_wcs * self.wcs.wcs.cunit[2]).to(self.vel_unit).value)
                e_velocity.append(
                    e_mean_value * self.velocity_increment)

                vel_disp.append(fwhm_value / 2.354820045)
                e_vel_disp.append(e_fwhm_value / 2.354820045)

                integrated_intensity = area_of_gaussian(amp_value, fwhm_value)
                e_integrated_intensity = area_of_gaussian(
                    amp_value + e_amp_value, fwhm_value + e_fwhm_value) -\
                    integrated_intensity
                int_tot.append(integrated_intensity)
                e_int_tot.append(e_integrated_intensity)
                x_coord.append(x_wcs)
                y_coord.append(y_wcs)

        names = ['x_pos', 'y_pos', 'z_pos', 'amp', 'rms', 'vel_disp',
                 'int_tot', self.wcs.wcs.lngtyp, self.wcs.wcs.lattyp, 'VLSR',
                 'e_amp', 'e_vel_disp', 'e_VLSR', 'e_int_tot']

        dtype = tuple(3*['i4'] + (len(names) - 3)*['f4'])

        table = Table([
            x_pos, y_pos, z_pos, amp, rms, vel_disp, int_tot, x_coord,
            y_coord, velocity, e_amp, e_vel_disp, e_velocity, e_int_tot],
            names=names, dtype=dtype)

        for key in names[3:]:
            table[key].format = "{0:.4f}"

        tableDirname = os.path.join(os.path.dirname(self.dirname), 'gpy_tables')
        if not os.path.exists(tableDirname):
            os.makedirs(tableDirname)

        filename = '{}{}_wco.dat'.format(self.filename, self.suffix)
        pathToTable = os.path.join(tableDirname, filename)
        table.write(pathToTable, format='ascii', overwrite=True)
        self.say(">> saved table '{}' in {}".format(filename, tableDirname))

    def produce_component_map(self):
        """Create FITS map showing the number of fitted components.

        The FITS file in saved in the gpy_maps directory.
        """
        self.say("\nmaking component map...")
        data = np.empty((self.header['NAXIS2'],
                         self.header['NAXIS1']))
        data.fill(np.nan)

        for idx, ((y, x), components) in enumerate(zip(
                self.location, self.decomposition['N_components'])):
            if components is not None:
                data[y, x] = components

        comments = ['Number of fitted GaussPy components']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_component_map.fits".format(
            self.filename, self.suffix)
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data, header, pathToFile, verbose=True)

    def produce_rchi2_map(self):
        """Create FITS map showing the reduced chi-square values of the decomposition.

        The FITS file in saved in the gpy_maps directory.
        """
        self.say("\nmaking reduced chi2 map...")

        data = np.empty((self.header['NAXIS2'], self.header['NAXIS1']))
        data.fill(np.nan)

        for idx, ((y, x), components, rchi2) in enumerate(zip(
                self.location, self.decomposition['N_components'],
                self.decomposition['best_fit_rchi2'])):
            if components is not None:
                if rchi2 is None:
                    data[y, x] = 0.
                else:
                    data[y, x] = rchi2

        comments = ['Reduced chi2 values of GaussPy fits']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_rchi2_map.fits".format(
            self.filename, self.suffix)
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data, header, pathToFile, verbose=True)

    def produce_velocity_dispersion_map(self, mode='average'):
        """Produce map showing the maximum velocity dispersions."""
        self.say("\nmaking map of maximum velocity dispersions...")

        data = np.empty((self.header['NAXIS2'], self.header['NAXIS1']))
        data.fill(np.nan)

        # TODO: rewrite this in terms of wcs and CUNIT
        factor_kms = self.header['CDELT3'] / 1e3

        for idx, ((y, x), fwhms) in enumerate(zip(
                self.location, self.decomposition['fwhms_fit'])):
            if fwhms is not None:
                if len(fwhms) > 0:
                    if mode == 'average':
                        data[y, x] = np.mean(fwhms) * factor_kms / 2.354820045
                    elif mode == 'maximum':
                        data[y, x] = max(fwhms) * factor_kms / 2.354820045
                else:
                    data[y, x] = 0

        if mode == 'average':
            comments = ['Average velocity dispersion values of GaussPy fits']
        elif mode == 'maximum':
            comments = ['Maximum velocity dispersion values of GaussPy fits']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_{}_veldisp_map.fits".format(
            self.filename, self.suffix, mode)
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data, header, pathToFile, verbose=False)
        self.say(">> saved {} velocity dispersion map '{}' in {}".format(
            mode, filename, os.path.dirname(pathToFile)))
