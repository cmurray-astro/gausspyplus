# @Author: riener
# @Date:   2019-01-09T12:27:55+01:00
# @Filename: moment_masking.py
# @Last modified by:   riener
# @Last modified time: 2019-03-04T12:36:57+01:00

"""Moment masking procedure from Dame (2011)."""

import os
import itertools
import warnings

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from tqdm import tqdm

from gausspyplus.shared_functions import max_consecutive_channels
from gausspyplus.spectral_cube_functions import remove_additional_axes, spatial_smoothing, spectral_smoothing, open_fits_file, calculate_average_rms_noise, moment_map, pv_map, correct_header
from gausspyplus.miscellaneous_functions import get_neighbors


def say(message, verbose=False):
    """Diagnostic messages."""
    if verbose is True:
        print(message)


class MomentMask(object):
    """Moment masking procedure from Dame (2011)."""

    def __init__(self, pathToFile, outputDir=None):
        self.pathToFile = pathToFile
        self.dirname = os.path.dirname(pathToFile)
        self.file = os.path.basename(pathToFile)
        self.filename, self.fileExtension = os.path.splitext(self.file)
        self.outputDir = None
        if outputDir is not None:
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            self.outputDir = outputDir

        self.sliceParams = None
        self.p_limit = 0.025
        self.pad_channels = 5
        self.use_nCpus = None
        self.pathToNoiseMap = None
        self.maskingCube = None
        self.targetResolutionSpatial = None
        self.targetResolutionSpectral = None
        self.currentResolutionSpatial = None
        self.currentResolutionSpectral = None
        self.numberRmsSpectra = 1000
        self.clippingLevel = 5
        self.verbose = True
        self.random_seed = 111

    def prepare_cube(self):
        # self.check_settings()

        hdu = fits.open(self.pathToFile)[0]
        self.data = hdu.data
        self.header = hdu.header
        self.wcs = WCS(self.header)

        self.header = correct_header(self.header)

        self.data, self.header = remove_additional_axes(self.data, self.header)

        yMax = self.data.shape[1]
        xMax = self.data.shape[2]
        self.locations = list(itertools.product(range(yMax), range(xMax)))

        self.n_channels = self.data.shape[0]
        self.maxConsecutiveChannels = max_consecutive_channels(self.n_channels, self.p_limit)

        self.nanMask = np.isnan(self.data)
        self.nanMask2D = np.zeros((yMax, xMax))
        for ypos, xpos in self.locations:
            self.nanMask2D[ypos, xpos] = self.nanMask[:, ypos, xpos].all()
        self.nanMask2D = self.nanMask2D.astype('bool')

        if self.pathToNoiseMap is not None:
            self.noiseMap = open_fits_file(self.pathToNoiseMap, get_header=False)

        if self.currentResolutionSpatial is None:
            self.currentResolutionSpatial = abs(
                self.wcs.wcs.cdelt[0]) * self.wcs.wcs.cunit[0]

        if self.currentResolutionSpectral is None:
            self.currentResolutionSpectral = abs(
                self.wcs.wcs.cdelt[2]) * self.wcs.wcs.cunit[2]

        if self.targetResolutionSpatial is None:
            self.targetResolutionSpatial = 2*self.currentResolutionSpatial
            warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(self.targetResolutionSpatial))

        if self.targetResolutionSpectral is None:
            self.targetResolutionSpectral = 2*self.currentResolutionSpectral
            warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(self.targetResolutionSpectral))

        # TODO: errorMessage if targetResolution* < currentResolution*

        # TODO: change round to int?
        self.n_s = round(0.5*self.targetResolutionSpatial.value /
                         self.currentResolutionSpatial.value)
        self.n_v = round(0.5*self.targetResolutionSpectral.value /
                         self.currentResolutionSpectral.value)

    def moment_masking(self):
        say('Preparing cube ...', verbose=self.verbose)
        self.prepare_cube()

        if self.maskingCube is None:
            self.moment_masking_first_steps()

        self.moment_masking_final_step()

    def moment_masking_first_steps(self):
        # 1) Determine the rms noise in T(v,x,y) (if noise map is not supplied)

        # 2) Generate a smoothed version of the data cube T_S(v,x,y) by degrading the resolution spatially by a factor of ~2 and in velocity to the width of the narrowest spectral lines generally observed.

        say('Smoothing cube spatially to a resolution of {} ...'.format(self.targetResolutionSpatial), verbose=self.verbose)

        self.dataSmoothed, self.headerSmoothed = spatial_smoothing(
            self.data.copy(), self.header, target_resolution=self.targetResolutionSpatial,
            current_resolution=self.currentResolutionSpatial)

        say('Smoothing cube spectrally to a resolution of {} ...'.format(self.targetResolutionSpectral), verbose=self.verbose)

        self.dataSmoothed, self.headerSmoothed = spectral_smoothing(
            self.dataSmoothed, self.headerSmoothed, target_resolution=self.targetResolutionSpectral,
            current_resolution=self.currentResolutionSpectral)

        self.dataSmoothedWithNans = self.dataSmoothed.copy()
        for ypos, xpos in self.locations:
            nan_mask = self.nanMask[:, ypos, xpos]
            self.dataSmoothedWithNans[:, ypos, xpos][nan_mask] = np.nan

        # 3) Determine the rms noise in T_S(v,x,y)
        # TODO: Take care that your smoothing algorithm does not zero (rather than blank) edge pixels since this would artificially lower the rms. Likewise be aware of under-sampled regions that were filled by linear interpolation, since these will have higher rms in the smoothed cube.

        if self.pathToNoiseMap is None:
            self.calculate_rms_noise()
        else:
            say('Using rms values from {} for the thresholding step for the smoothed cube...'.format(os.path.basename(self.pathToNoiseMap)), verbose=self.verbose)

        # 4) Generate a masking cube M(v,x,y) initially filled with zeros with the same dimensions as T and TS. The moment masked cube TM(v,x,y) will be calculated as M*T.

        say('Moment masking ...', verbose=self.verbose)
        self.maskingCube = np.zeros(self.data.shape)

        # 5) For each pixel T_S(vi, xj, yk) > Tc, unmask (set to 1) the corresponding pixel in M. Also unmask all pixels in M within the smoothing kernel of T_S(vi, xj, yk), since all of these pixels weigh into the value of T_S(vi, xj, yk). That is, unmask within n_v pixels in velocity and within n_s pixels spatially, where n_v = 0.5*fwhm_v / dv and n_s = 0.5*fwhm_s / ds

        pbar = tqdm(total=len(self.locations))

        for ypos, xpos in self.locations:
            pbar.update()
            spectrum_smoothed = self.dataSmoothed[:, ypos, xpos]
            nan_mask = self.nanMask[:, ypos, xpos]
            spectrum_smoothed[nan_mask] = 0
            rms_smoothed = self.noiseSmoothedCube[ypos, xpos]

            #  do not unmask anything if rms could not be calculated
            if np.isnan(rms_smoothed):
                continue

            if np.isnan(spectrum_smoothed).any():
                print("Nans", ypos, xpos)
            mask_v = spectrum_smoothed > self.clippingLevel*rms_smoothed
            mask_v = self.mask_pixels_in_velocity(mask_v)

            position_of_spectra_within_n_s = get_neighbors(
                (ypos, xpos), exclude_p=False, shape=self.data.shape[1:], nNeighbors=self.n_s)
            for pos in position_of_spectra_within_n_s:
                self.maskingCube[:, pos[0], pos[1]][mask_v] = 1
        pbar.close()

    def moment_masking_final_step(self):
        for ypos, xpos in self.locations:
            nan_mask = self.nanMask[:, ypos, xpos]
            mask = self.maskingCube[:, ypos, xpos]
            mask[nan_mask] = 0
            mask = mask.astype('bool')
            self.data[:, ypos, xpos][~mask] = 0

    def calculate_rms_noise(self):
        say('Determining average rms noise from {} spectra ...'.format(self.numberRmsSpectra), verbose=self.verbose)
        averageRms = calculate_average_rms_noise(
            self.dataSmoothedWithNans, self.numberRmsSpectra,
            maxConsecutiveChannels=self.maxConsecutiveChannels,
            pad_channels=self.pad_channels, random_seed=self.random_seed)
        say('Determined average rms value of {}'.format(averageRms), verbose=self.verbose)

        say('Determining noise of smoothed cube ...', verbose=self.verbose)

        self.noiseSmoothedCube = np.empty(
            (self.data.shape[1], self.data.shape[2]))

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([self.locations, [self.dataSmoothedWithNans, self.maxConsecutiveChannels, self.pad_channels, averageRms]])

        results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='noise')

        for i, rms in tqdm(enumerate(results_list)):
            if not isinstance(rms, np.float):
                warnings.warn('Problems with entry {} from resulting parallel_processing list, skipping entry'.format(i))
                continue
            else:
                ypos, xpos = self.locations[i]
                self.noiseSmoothedCube[ypos, xpos] = rms

    def mask_pixels_in_velocity(self, mask):
        mask = mask.astype('float')
        mask_new = mask.copy()
        for i in range(1, self.n_v + 1):
            # unmask element to the left
            mask_new += np.append(mask[i:], np.zeros(i))
            # unmask element to the right
            mask_new += np.append(np.zeros(i), mask[:-i])
        mask_new = mask_new.astype('bool')
        return mask_new

    def make_moment_map(self, order=0, linewidth='sigma', save=True, get_hdu=False,
                        vel_unit=u.km/u.s, restoreNans=True):
        pathToOutputFile = self.get_path_to_output_file(
            suffix='_mom_{}_map'.format(order))
        hdu = fits.PrimaryHDU(
            data=self.data.copy(), header=self.header.copy())
        moment_map(hdu=hdu, sliceParams=self.sliceParams, save=save,
                   order=order, linewidth=linewidth,
                   pathToOutputFile=pathToOutputFile,
                   vel_unit=vel_unit, applyNoiseThreshold=False,
                   get_hdu=get_hdu, restoreNans=restoreNans,
                   nanMask=self.nanMask2D)

    def make_pv_map(self, save=True, get_hdu=False, vel_unit=u.km/u.s,
                    sumOverLatitude=True, suffix='_pv_map'):
        pathToOutputFile = self.get_path_to_output_file(
            suffix=suffix)
        hdu = fits.PrimaryHDU(
            data=self.data.copy(), header=self.header.copy())
        pv_map(hdu=hdu, sliceParams=self.sliceParams, get_hdu=False, save=True,
               pathToOutputFile=pathToOutputFile, applyNoiseThreshold=False,
               sumOverLatitude=sumOverLatitude)

    def get_path_to_output_file(self, suffix=''):
        if self.outputDir is not None:
            filename = '{}{}.fits'.format(self.filename, suffix)
            pathToOutputFile = os.path.join(self.outputDir, filename)
        else:
            pathToOutputFile = None
        return pathToOutputFile
use_ncpus
