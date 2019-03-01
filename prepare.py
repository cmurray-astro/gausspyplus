# @Author: riener
# @Date:   2019-02-26T16:38:04+01:00
# @Filename: prepare.py
# @Last modified by:   riener
# @Last modified time: 2019-03-01T14:45:48+01:00

import logging
import os
import pickle
import itertools
import sys

import numpy as np

from datetime import datetime
from astropy.io import fits
from tqdm import tqdm

from gausspyplus.shared_functions import max_consecutive_channels, mask_channels,\
    get_signal_ranges, get_noise_spike_ranges, add_subtracted_nan_ranges

from gausspyplus.spectral_cube_functions import determine_noise, remove_additional_axes,\
    calculate_average_rms_noise, add_noise, change_header, save_fits

if (sys.version_info < (3, 0)):
    raise Exception('Script has to be run in Python 3 environment.')


class GaussPyPrepare(object):
    def __init__(self, pathToFile, gpyDirname=None):
        self.pathToFile = pathToFile
        self.parentDirname = os.path.dirname(pathToFile)
        self.file = os.path.basename(pathToFile)
        self.filename, self.fileExtension = os.path.splitext(self.file)
        if gpyDirname is not None:
            self.parentDirname = gpyDirname
        self.pickleDirname = os.path.join(self.parentDirname, 'gpy_prepared')

        self.gausspyPickle = True
        self.testing = False
        self.dataLocation = None
        self.simulation = False

        self.rmsFromData = True
        self.average_rms = None
        #  TODO: check if this is always calculated from different spectra for
        #  different runs
        self.numberRmsSpectra = 1000
        self.pLimit = 0.025
        self.padChannels = 5
        self.signalMask = True
        self.minChannels = 100
        self.mask_out_ranges = []

        self.snr = 3.
        self.significance = 5.
        self.noiseSpikeSnr = 4.

        self.suffix = ''
        self.useCpus = None
        self.log_output = True
        self.verbose = True
        self.overwrite = True

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

        logname = os.path.join(dirname, '{}_prepare_{}.log'.format(
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

        string = 'GaussPy preparation'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        self.say(heading)

    def say(self, message):
        """Diagnostic messages."""
        if self.log_output:
            self.logger.info(message)
        if self.verbose:
            print(message)

    def prepare_cube(self):
        self.check_settings()

        hdu = fits.open(self.pathToFile)[0]

        if self.simulation:
            self.rmsFromData = False
            hdu = add_noise(self.average_rms, hdu=hdu, get_hdu=True)

        self.data = hdu.data
        self.header = hdu.header

        self.data, self.header = remove_additional_axes(
            self.data, self.header, verbose=self.verbose)

        self.errors = np.empty((self.data.shape[1], self.data.shape[2]))

        if self.testing:
            ypos = self.dataLocation[0]
            xpos = self.dataLocation[1]
            self.say('\nTesting: using only pixel at location ({}, {})'.format(
                ypos, xpos))
            self.data = self.data[:, ypos, xpos]
            self.data = self.data[:, np.newaxis, np.newaxis]
            self.rmsFromData = False

        self.nChannels = self.data.shape[0]
        if self.nChannels < self.minChannels:
            self.signalMask = False

        self.maxConsecutiveChannels = max_consecutive_channels(
            self.nChannels, self.pLimit)

        if self.rmsFromData:
            self.calculate_average_rms_from_data()

        if self.gausspyPickle:
            self.prepare_gausspy_pickle()

    def check_settings(self):

        if self.testing and (self.dataLocation is None):
            errorMessage = \
                """specify 'dataLocation' as (y, x) for 'testing'"""
            raise Exception(errorMessage)

        if self.simulation and (self.average_rms is None):
            errorMessage = \
                """specify 'average_rms' for 'simulation'"""
            raise Exception(errorMessage)

        if self.pickleDirname is None:
            errorMessage = """pickleDirname is not defined"""
            raise Exception(errorMessage)
        if not os.path.exists(self.pickleDirname):
            os.makedirs(self.pickleDirname)

        self.getting_ready()

    def calculate_average_rms_from_data(self):
        self.say('\ncalculating average rms from data...')

        self.average_rms = calculate_average_rms_noise(
            self.data.copy(), self.numberRmsSpectra,
            padChannels=self.padChannels,
            maxConsecutiveChannels=self.maxConsecutiveChannels)

        self.say('>> calculated rms value of {:.3f} from data'.format(
                self.average_rms))

    def prepare_gausspy_pickle(self):
        self.say('\npreparing GaussPy cube...')

        data = {}
        channels = np.arange(self.data.shape[0])

        if self.testing:
            locations = [(0, 0)]
        else:
            yMax = self.data.shape[1]
            xMax = self.data.shape[2]
            locations = list(itertools.product(range(yMax), range(xMax)))

        data['header'] = self.header
        data['nan_mask'] = np.isnan(self.data)
        data['x_values'] = channels
        data['data_list'], data['error'], data['index'], data['location'] = (
            [] for _ in range(4))

        if self.signalMask:
            data['signal_ranges'], data['noise_spike_ranges'] = ([] for _ in range(2))

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([locations, [self]])

        results_list = gausspyplus.parallel_processing.func(usecpus=self.useCpus, function='gpy_noise')

        print('SUCCESS\n')

        for i, item in tqdm(enumerate(results_list)):
            if not isinstance(item, list):
                print(i, item)
                continue
            idx, spectrum, (ypos, xpos), error, signal_ranges, noise_spike_ranges =\
                item
            self.errors[ypos, xpos] = error
            data['index'].append(idx)
            data['location'].append((ypos, xpos))

            if not np.isnan(error):
                # TODO: return spectrum = None if spectrum wasn't part nans
                # and changes with randomly rms sampled values
                # then add condition if spectrum is None for next line
                # data['data_list'].append(self.data[:, ypos, xpos])
                data['data_list'].append(spectrum)
                data['error'].append([error])
                if self.signalMask:
                    # TODO: make noise_spike_ranges independent from signal_ranges??
                    data['signal_ranges'].append(signal_ranges)
                    data['noise_spike_ranges'].append(noise_spike_ranges)
            else:
                # TODO: rework that so that list is initialized with None values
                # and this condition is obsolete?
                data['data_list'].append(None)
                data['error'].append([None])
                data['signal_ranges'].append(None)
                data['noise_spike_ranges'].append(None)

        self.say("\npickle dump dictionary...")

        if self.testing:
            suffix = '_test'
            data['testing'] = self.testing
        else:
            suffix = self.suffix

        pathToFile = os.path.join(
            self.pickleDirname, '{}{}.pickle'.format(self.filename, suffix))
        pickle.dump(data, open(pathToFile, 'wb'), protocol=2)
        print(">> for GaussPyDecompose: pathToPickleFile = '{}'".format(
            pathToFile))

    def calculate_rms_noise(self, location, idx):
        ypos, xpos = location
        spectrum = self.data[:, ypos, xpos].copy()

        signal_ranges, noise_spike_ranges = (None for _ in range(2))

        if self.mask_out_ranges:
            nan_mask = mask_channels(self.nChannels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan

        #  if spectrum contains nans they will be replaced by noise values
        #  randomly sampled from the calculated rms value
        rms = determine_noise(
            spectrum, maxConsecutiveChannels=self.maxConsecutiveChannels,
            padChannels=self.padChannels, idx=idx, averageRms=self.average_rms)

        if self.signalMask and not np.isnan(rms):
            noise_spike_ranges = get_noise_spike_ranges(
                spectrum, rms, noiseSpikeSnr=self.noiseSpikeSnr)
            if self.mask_out_ranges:
                noise_spike_ranges += self.mask_out_ranges
            signal_ranges = get_signal_ranges(
                spectrum, rms, snr=self.snr, significance=self.significance, maxConsecutiveChannels=self.maxConsecutiveChannels,
                padChannels=self.padChannels, minChannels=self.minChannels,
                remove_intervals=noise_spike_ranges)

        return [idx, spectrum, location, rms, signal_ranges, noise_spike_ranges]

    def produce_noise_map(self):
        comments = ['noise map']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_noise_map.fits".format(self.filename, self.suffix)
        pathToFile = os.path.join(
            os.path.dirname(self.pickleDirname), 'gpy_maps', filename)

        save_fits(self.errors, header, pathToFile, verbose=False)
        self.say("\n>> saved noise map '{}' to {}".format(
            filename, os.path.dirname(pathToFile)))
