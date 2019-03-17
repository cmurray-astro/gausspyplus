# @Author: riener
# @Date:   2019-02-18T16:27:12+01:00
# @Filename: training_set.py
# @Last modified by:   riener
# @Last modified time: 2019-03-17T15:04:19+01:00

import ast
import configparser
import itertools
import multiprocessing
import os
import pickle
import random
import signal
import sys

import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.modeling import models, fitting, optimizers

from tqdm import tqdm

from gausspyplus.shared_functions import gaussian, determine_significance, max_consecutive_channels, get_noise_spike_ranges, get_signal_ranges, mask_channels, goodness_of_fit

from gausspyplus.spectral_cube_functions import determine_noise, remove_additional_axes

if (sys.version_info < (3, 0)):
    raise Exception('Script has to be run in Python 3 environment.')


def mp_init_worker():
    """Worker initializer to ignore Keyboard interrupt."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def mp_init(lst):
    global mp_ilist, mp_indices, mp_gpy_object
    mp_gpy_object, mp_indices = lst
    mp_ilist = np.arange(len(mp_indices))


def mp_decompose_one(i):
    result = GaussPyTrainingSet.decompose(mp_gpy_object, mp_indices[i], i)
    return result


def mp_func(total, use_ncpus=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    if use_ncpus is None:
        use_ncpus = int(0.75 * ncpus)
    print('using {} out of {} cpus'.format(use_ncpus, ncpus))
    p = multiprocessing.Pool(use_ncpus, mp_init_worker)

    try:
        results_list = []
        counter = 0
        pbar = tqdm(total=total)
        for i, result in enumerate(p.imap_unordered(mp_decompose_one, mp_ilist)):
            if result is not None:
                counter += 1
                pbar.update(1)
                results_list.append(result)
            if counter == total:
                break
        pbar.close()

    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        p.terminate()
        quit()
    p.close()
    del p
    return results_list


class GaussPyTrainingSet(object):
    def __init__(self, pathToFile, configFile=''):
        self.pathToFile = pathToFile
        self.path_to_training_set = None

        self.training_set = True
        self.n_spectra = 5
        self.order = 6
        self.snr = 3
        self.significance = 5
        self.min_fwhm = 1.
        self.max_fwhm = None
        self.p_limit = 0.025
        self.signal_mask = True
        self.pad_channels = 5
        self.min_channels = 100
        self.snr_noise_spike = 4.
        # TODO: also define lower limit for rchi2 to prevent overfitting?
        self.rchi2_limit = 1.5
        self.use_all = False
        self.mask_out_ranges = []

        self.verbose = True
        self.suffix = ''
        self.use_ncpus = None
        self.random_seed = 111

        if configFile:
            self.get_values_from_config_file(configFile)

    def get_values_from_config_file(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)

        for key, value in config['training'].items():
            try:
                setattr(self, key, ast.literal_eval(value))
            except ValueError:
                if key == 'vel_unit':
                    value = u.Unit(value)
                    setattr(self, key, value)
                else:
                    raise Exception('Could not parse parameter {} from config file'.format(key))

    def initialize(self):
        self.minStddev = None
        if self.min_fwhm is not None:
            self.minStddev = self.min_fwhm/2.355

        self.maxStddev = None
        if self.max_fwhm is not None:
            self.maxStddev = self.max_fwhm/2.355

        self.dirname = os.path.dirname(self.pathToFile)
        self.file = os.path.basename(self.pathToFile)
        self.filename, self.fileExtension = os.path.splitext(self.file)

        self.header = None

        if self.fileExtension == '.fits':
            hdu = fits.open(self.pathToFile)[0]
            self.data = hdu.data
            self.header = hdu.header

            self.data, self.header = remove_additional_axes(
                self.data, self.header)
            self.n_channels = self.data.shape[0]
        else:
            with open(os.path.join(self.pathToFile), "rb") as pickle_file:
                dctData = pickle.load(pickle_file, encoding='latin1')
            self.data = dctData['data_list']
            self.n_channels = len(self.data[0])

        self.channels = np.arange(self.n_channels)

    def decompose_spectra(self):
        self.initialize()
        if self.verbose:
            print("decompose {} spectra ...".format(self.n_spectra))

        if self.random_seed is not None:
            random.seed(self.random_seed)

        if self.training_set:
            data = {}

        self.mask_omit = mask_channels(self.n_channels, self.mask_out_ranges)

        self.maxConsecutiveChannels = max_consecutive_channels(self.n_channels, self.p_limit)

        if self.header:
            yValues = np.arange(self.data.shape[1])
            xValues = np.arange(self.data.shape[2])
            nSpectra = yValues.size * xValues.size
            self.locations = list(itertools.product(yValues, xValues))
            indices = random.sample(list(range(nSpectra)), nSpectra)
        else:
            nSpectra = len(self.data)
            indices = random.sample(list(range(nSpectra)), nSpectra)
            # indices = np.array([4506])  # for testing

        if self.use_all:
            self.n_spectra = nSpectra

        #  start multiprocessing
        mp_init([self, indices])
        results_list = mp_func(self.n_spectra, use_ncpus=self.use_ncpus)
        print('SUCCESS\n')
        for result in results_list:
            if result is not None:
                fit_values, spectrum, location, signal_ranges, rms, rchi2, index, i = result
                # the next four lines are added to deal with the use_all=True feature
                if rchi2 is None:
                    continue
                if rchi2 > self.rchi2_limit:
                    continue
                amps, fwhms, means = ([] for i in range(3))
                if fit_values is not None:
                    for item in fit_values:
                        amps.append(item[0])
                        means.append(item[1])
                        fwhms.append(item[2]*2.355)

                # data['data_list'] = data.get('data_list', []) + [self.data[:, location[0], location[1]]]
                data['data_list'] = data.get('data_list', []) + [spectrum]
                if self.header:
                    data['location'] = data.get('location', []) + [location]
                data['index'] = data.get('index', []) + [index]
                data['error'] = data.get('error', []) + [[rms]]
                data['best_fit_rchi2'] = data.get('best_fit_rchi2', []) + [rchi2]
                data['amplitudes'] = data.get('amplitudes', []) + [amps]
                data['fwhms'] = data.get('fwhms', []) + [fwhms]
                data['means'] = data.get('means', []) + [means]
                data['signal_ranges'] = data.get('signal_ranges', []) + [signal_ranges]
                # data['x_values'] = data.get('x_values', []) + [self.channels]
        data['x_values'] = self.channels
        if self.header:
            data['header'] = self.header

        if self.training_set:
            if not os.path.exists(self.path_to_training_set):
                os.makedirs(self.path_to_training_set)
            filename = '{}{}.pickle'.format(self.filename, self.suffix)
            pathToFile = os.path.join(self.path_to_training_set, filename)
            pickle.dump(data, open(pathToFile, 'wb'), protocol=2)

    def decompose(self, index, i):
        if self.header:
            location = self.locations[index]
            spectrum = self.data[:, location[0], location[1]].copy()
        else:
            location = None
            spectrum = self.data[index].copy()

        if self.mask_out_ranges:
            nan_mask = mask_channels(self.n_channels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan

        rms = determine_noise(
            spectrum, maxConsecutiveChannels=self.maxConsecutiveChannels,
            pad_channels=self.pad_channels, idx=index, averageRms=None)

        if np.isnan(rms):
            return None

        noise_spike_ranges = get_noise_spike_ranges(
            spectrum, rms, snr_noise_spike=self.snr_noise_spike)
        if self.mask_out_ranges:
            noise_spike_ranges += self.mask_out_ranges

        signal_ranges = get_signal_ranges(
            spectrum, rms, snr=self.snr, significance=self.significance,
            maxConsecutiveChannels=self.maxConsecutiveChannels,
            pad_channels=self.pad_channels, min_channels=self.min_channels,
            remove_intervals=noise_spike_ranges)

        if signal_ranges:
            mask_signal = mask_channels(self.n_channels, signal_ranges)
        else:
            mask_signal = None

        maxima = self.get_maxima(spectrum, rms)
        fit_values, rchi2 = self.gaussian_fitting(
            spectrum, maxima, rms, mask_signal=mask_signal)
        # TODO: change the rchi2_limit value??
        if ((fit_values is not None) and (rchi2 < self.rchi2_limit)) or self.use_all:
            return [fit_values, spectrum, location, signal_ranges, rms,
                    rchi2, index, i]
        else:
            return None

    def get_maxima(self, spectrum, rms):
        from scipy.signal import argrelextrema

        array = spectrum.copy()
        #  set all spectral data points below threshold to zero
        low_values = array < self.snr*rms
        array[low_values] = 0
        #  find local maxima (order of x considers x neighboring data points)
        maxima = argrelextrema(array, np.greater, order=self.order)
        return maxima

    def gaussian_fitting(self, spectrum, maxima, rms, mask_signal=None):
        # TODO: don't hardcode the value of stddev_ini
        stddev_ini = 2  # in channels

        gaussians = []
        # loop through spectral channels of the local maxima, fit Gaussians
        sortedAmps = np.argsort(spectrum[maxima])[::-1]

        for idx in sortedAmps:
            mean, amp = maxima[0][idx], spectrum[maxima][idx]
            gauss = models.Gaussian1D(amp, mean, stddev_ini)
            gauss.bounds['amplitude'] = (None, 1.1*amp)
            gaussians.append(gauss)

        improve = True
        while improve is True:
            fit_values = self.determine_gaussian_fit_models(
                gaussians, spectrum)
            if fit_values is not None:
                improve, gaussians = self.check_fit_parameters(
                        fit_values, gaussians, rms)
            else:
                improve = False

        if fit_values is not None:
            comps = len(fit_values)
        else:
            comps = 0

        channels = np.arange(len(spectrum))
        if comps > 0:
            for j in range(len(fit_values)):
                gauss = gaussian(
                    fit_values[j][0], fit_values[j][2]*2.355, fit_values[j][1], channels)
                if j == 0:
                    combined_gauss = gauss
                else:
                    combined_gauss += gauss
        else:
            combined_gauss = np.zeros(len(channels))
        if comps > 0:
            rchi2 = goodness_of_fit(spectrum, combined_gauss, rms, comps, mask=mask_signal)
        else:
            rchi2 = None
        return fit_values, rchi2

    def check_fit_parameters(self, fit_values, gaussians, rms):
        improve = False
        revised_gaussians = gaussians.copy()
        for initial_guess, final_fit in zip(gaussians, fit_values):
            if (final_fit[0] < self.snr*rms):
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if final_fit[2] <= 0:
                print('negative!')
                # TODO: remove this negative Gaussian
            significance = determine_significance(
                final_fit[0], final_fit[2]*2.35482, rms)
            if significance < self.significance:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if self.maxStddev is not None:
                if final_fit[2] > self.maxStddev:
                    revised_gaussians.remove(initial_guess)
                    improve = True
                    break

            if self.minStddev is not None:
                if final_fit[2] < self.minStddev:
                    revised_gaussians.remove(initial_guess)
                    improve = True
                    break

        if improve:
            gaussians = revised_gaussians
        return improve, gaussians

    def determine_gaussian_fit_models(self, gaussians, spectrum):
        fit_values = None
        optimizers.DEFAULT_MAXITER = 1000
        channels = np.arange(self.n_channels)

        # To fit the data create a new superposition with initial
        # guesses for the parameters:
        if len(gaussians) > 0:
            gg_init = gaussians[0]

            if len(gaussians) > 1:
                for i in range(1, len(gaussians)):
                    gg_init += gaussians[i]

            fitter = fitting.SLSQPLSQFitter()
            gg_fit = fitter(gg_init, channels, spectrum, disp=False)

            fit_values = []
            if len(gg_fit.param_sets) > 3:
                for i in range(len(gg_fit.submodel_names)):
                    fit_values.append([gg_fit[i].amplitude.value,
                                       gg_fit[i].mean.value,
                                       abs(gg_fit[i].stddev.value)])
            else:
                fit_values.append([gg_fit.amplitude.value,
                                   gg_fit.mean.value,
                                   abs(gg_fit.stddev.value)])
        return fit_values
