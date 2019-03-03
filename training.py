# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: training.py
# @Last modified by:   riener
# @Last modified time: 2019-03-01T15:09:01+01:00

import os
import pickle
import numpy as np
import sys

# if (sys.version_info >= (3, 0)):
#     raise Exception('Script has to be run in Python 2 environment.')


class GaussPyTraining(object):
    def __init__(self, pathToTrainingSet):
        self.pathToTrainingSet = pathToTrainingSet
        self.dirname = os.path.dirname(pathToTrainingSet)
        self.file = os.path.basename(pathToTrainingSet)
        self.filename, self.fileExtension = os.path.splitext(self.file)

        self.createTrainingSet = True
        self.paramsFromData = True
        self.twoPhaseDecomposition = True
        self.alpha1, self.alpha2, self.snrThresh, self.snr2Thresh = (
                None for i in range(4))
        self.nChannels, self.nSpectra, self.nCompsLims, self.ampLims,\
            self.fwhmLims, self.meanLims, self.rms = (None for i in range(7))

        self.tryDecomposition = False
        self.numberRmsSpectra = 5000
        self.meanEdgeChans = 10

        self.verbose = True
        self.suffix = ''

    def training(self):
        if self.createTrainingSet:
            self.check_settings()
            self.create_training_set()

        self.gausspy_train_alpha()

    def check_settings(self):
        if self.nCompsLims is None:
            errorMessage = str("specify 'nCompsLims' as [minComps, maxComps]")
            raise Exception(errorMessage)

        if self.ampLims is None:
            errorMessage = str("specify 'ampLims' as [minAmp, maxAmp]")
            raise Exception(errorMessage)

        if self.meanLims is None:
            errorMessage = str("specify 'meanLims' in channels as "
                               "[minMean, maxMean]")
            raise Exception(errorMessage)

        if self.rms is None:
            errorMessage = str("specify 'rms'")
            raise Exception(errorMessage)

        if self.fwhmLims is None:
            errorMessage = str("specify 'fwhmLims' in channels as "
                               "[minFwhm, maxFwhm]")
            raise Exception(errorMessage)

        if self.nChannels is None:
            errorMessage = str("specify 'nChannels'")
            raise Exception(errorMessage)

        if self.nSpectra is None:
            errorMessage = str("specify 'nSepctra'")
            raise Exception(errorMessage)

    def get_parameters_from_data(self, pathToFile):
        import itertools
        import random

        from astropy.io import fits

        if self.verbose:
            print("determine parameters from data ...")

        hdu = fits.open(pathToFile)[0]
        data = hdu.data

        self.nChannels = data.shape[0]

        yValues = np.arange(data.shape[1])
        xValues = np.arange(data.shape[2])
        locations = list(itertools.product(yValues, xValues))
        if len(locations) > self.numberRmsSpectra:
            locations = random.sample(locations, self.numberRmsSpectra)
        rmsList, maxAmps = ([] for i in range(2))
        for y, x in locations:
            spectrum = data[:, y, x]
            if not np.isnan(spectrum).any():
                maxAmps.append(max(spectrum))
                rms = np.std(spectrum[spectrum < abs(np.min(spectrum))])
                rmsList.append(rms)
                if self.tryDecomposition:
                    self.try_to_decompose(spectrum, rms)

        self.rms = np.median(rmsList)
        self.ampLims = [3*self.rms, 0.8*max(maxAmps)]
        self.meanLims = [0 + self.meanEdgeChans,
                         data.shape[0] - self.meanEdgeChans]

        if self.verbose:
            print("nChannels = {}".format(self.nChannels))
            print("rms = {}".format(self.rms))
            print("ampLims = {}".format(self.ampLims))
            print("meanLims = {}".format(self.meanLims))

    def gaussian(self, amp, fwhm, mean):
        return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)

    def create_training_set(self, training_set=True):
        print('create training set ...')

        # Initialize
        data = {}
        channels = np.arange(self.nChannels)
        error = self.rms

        # Begin populating data
        for i in range(self.nSpectra):
            amps, fwhms, means = ([] for i in range(3))
            spectrum = np.random.randn(self.nChannels) * self.rms

            ncomps = np.random.choice(
                np.arange(self.nCompsLims[0], self.nCompsLims[1] + 1))

            for comp in range(ncomps):
                # Select random values for components within specified ranges
                amp = np.random.uniform(self.ampLims[0], self.ampLims[1])
                fwhm = np.random.uniform(self.fwhmLims[0], self.fwhmLims[1])
                mean = np.random.uniform(self.meanLims[0], self.meanLims[1])

                # Add Gaussian with random parameters from above to spectrum
                spectrum += self.gaussian(amp, fwhm, mean)(channels)

                # Append the parameters to initialized lists for storing
                amps.append(amp)
                fwhms.append(fwhm)
                means.append(mean)

            # Enter results into AGD dataset
            data['data_list'] = data.get('data_list', []) + [spectrum]
            data['x_values'] = data.get('x_values', []) + [channels]
            data['error'] = data.get('error', []) + [error]

            # If training data, keep answers
            if training_set:
                data['amplitudes'] = data.get('amplitudes', []) + [amps]
                data['fwhms'] = data.get('fwhms', []) + [fwhms]
                data['means'] = data.get('means', []) + [means]

        # Dump synthetic data into specified filename
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        pickle.dump(data, open(self.pathToTrainingSet, 'w'))

    def gausspy_train_alpha(self):
        from .gausspy_py3 import gp as gp

        g = gp.GaussianDecomposer()

        g.load_training_data(self.pathToTrainingSet)
        g.set('SNR_thresh', self.snrThresh)
        g.set('SNR2_thresh', self.snr2Thresh)

        if self.twoPhaseDecomposition:
            g.set('phase', 'two')  # Set GaussPy parameters
            # Train AGD starting with initial guess for alpha
            g.train(alpha1_initial=self.alpha1, alpha2_initial=self.alpha2)
        else:
            g.set('phase', 'one')
            g.train(alpha1_initial=self.alpha1)
