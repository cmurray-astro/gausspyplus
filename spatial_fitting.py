# @Author: riener
# @Date:   2019-01-22T08:00:18+01:00
# @Filename: spatial_fitting.py
# @Last modified by:   riener
# @Last modified time: 2019-03-11T17:30:17+01:00

import ast
import collections
import configparser
# import datetime
# import logging
import os
import pickle

from astropy import units as u
from scipy.stats import normaltest
from tqdm import tqdm

from networkx.algorithms.components.connected import connected_components
import numpy as np

from gausspyplus.shared_functions import goodness_of_fit, mask_channels, mask_covering_gaussians
from gausspyplus.miscellaneous_functions import to_graph, get_neighbors, set_up_logger
from gausspyplus.gausspy_py3.gp_plus import split_params, get_fully_blended_gaussians, check_for_peaks_in_residual, get_best_fit, check_for_negative_residual, remove_components_from_list, combined_gaussian


class SpatialFitting(object):
    def __init__(self, pathToPickleFile, pathToDecompFile, finFilename,
                 configFile=''):
        self.pathToPickleFile = pathToPickleFile
        self.pathToDecompFile = pathToDecompFile
        self.finFilename = finFilename

        self.exclude_flagged = False
        self.max_fwhm = None
        self.rchi2_limit = 1.5
        self.rchi2_limit_refit = None
        self.max_diff_comps = 2
        self.max_jump_comps = 2
        self.n_max_jump_comps = 2
        self.max_refitting_iteration = 30

        self.flag_blended = False
        self.flag_residual = False
        self.flag_rchi2 = False
        self.flag_broad = False
        self.flag_ncomps = False
        self.refit_blended = False
        self.refit_residual = False
        self.refit_rchi2 = False
        self.refit_broad = False
        self.refit_ncomps = False

        self.mean_separation = 2.  # minimum distance between peaks in channels
        self.fwhm_separation = 4.
        self.snr = 3.
        self.fwhm_factor = 1.5
        self.fwhm_factor_refit = None
        self.broad_neighbor_fraction = 0.5
        self.min_weight = 0.5
        self.weight_factor = 2
        self.min_pvalue = 0.01
        self.use_ncpus = None
        self.verbose = True
        self.log_output = True
        self.only_print_flags = False

        if configFile:
            self.get_values_from_config_file(configFile)

    def get_values_from_config_file(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)

        for key, value in config['spatial fitting'].items():
            try:
                setattr(self, key, ast.literal_eval(value))
            except ValueError:
                if key == 'vel_unit':
                    value = u.Unit(value)
                    setattr(self, key, value)
                else:
                    raise Exception('Could not parse parameter {} from config file'.format(key))

    def initialize(self):
        self.dirname = os.path.dirname(self.pathToDecompFile)
        self.file = os.path.basename(self.pathToDecompFile)
        self.filename, self.fileExtension = os.path.splitext(self.file)
        self.parentDirname = os.path.dirname(self.dirname)

        with open(self.pathToPickleFile, "rb") as pickle_file:
            pickledData = pickle.load(pickle_file, encoding='latin1')

        self.indexList = pickledData['index']
        self.data = pickledData['data_list']
        self.errors = pickledData['error']
        if 'header' in pickledData.keys():
            self.header = pickledData['header']
            self.shape = (self.header['NAXIS2'], self.header['NAXIS1'])
            self.length = self.header['NAXIS2'] * self.header['NAXIS1']
            self.location = pickledData['location']
            self.n_channels = self.header['NAXIS3']
        else:
            self.length = len(self.data)
            self.n_channels = len(self.data[0])
        self.channels = np.arange(self.n_channels)

        self.signalRanges = pickledData['signal_ranges']
        self.noiseSpikeRanges = pickledData['noise_spike_ranges']

        with open(self.pathToDecompFile, "rb") as pickle_file:
            self.decomposition = pickle.load(pickle_file, encoding='latin1')

        self.nIndices = len(self.decomposition['index_fit'])

        self.decomposition['refit_iteration'] = [0] * self.nIndices
        self.decomposition['gaussians_rchi2'] = [None] * self.nIndices
        self.decomposition['gaussians_aicc'] = [None] * self.nIndices

        self.neighbor_indices = np.array([None]*self.nIndices)
        self.neighbor_indices_all = np.array([None]*self.nIndices)

        self.nanMask = np.isnan([np.nan if i is None else i
                                 for i in self.decomposition['N_components']])
        self.nanIndices = np.array(
            self.decomposition['index_fit'])[self.nanMask]

        self.signal_mask = [None for _ in range(self.nIndices)]
        for i, (noiseSpikeRanges, signalRanges) in enumerate(
                zip(self.noiseSpikeRanges, self.signalRanges)):
            if signalRanges is not None:
                self.signal_mask[i] = mask_channels(
                    self.n_channels, signalRanges,
                    remove_intervals=noiseSpikeRanges)

        #  starting condition so that refitting iteration can start
        # self.mask_refitted = np.ones(1)
        self.mask_refitted = np.array([1]*self.nIndices)
        self.list_n_refit = []
        self.refitting_iteration = 0
        self.min_p = 5/6

        if self.rchi2_limit_refit is None:
            self.rchi2_limit_refit = self.rchi2_limit
        if self.fwhm_factor_refit is None:
            self.fwhm_factor_refit = self.fwhm_factor
        if self.max_fwhm is None:
            self.max_fwhm = int(self.n_channels / 2)

    def getting_ready(self):
        if self.log_output:
            self.logger = set_up_logger(
                self.parentDirname, self.filename, method='g+_spatial_refitting')

        phase = 1
        if self.phase_two:
            phase = 2

        string = 'Spatial refitting - Phase {}'.format(phase)
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        self.say(heading)

        string = str(
            '\nFlagging:'
            '\n - Blended components: {a}'
            '\n - Negative residual features: {b}'
            '\n - Broad components: {c}'
            '\n   flagged if FWHM of broadest component in spectrum is:'
            '\n   >= {d} times the FWHM of second broadest component'
            '\n   or'
            '\n   >= {d} times any FWHM in >= {e:.0%} of its neigbors'
            '\n - High reduced chi2 values (> {f}): {g}'
            '\n - Differing number of components: {h}').format(
                a=self.flag_blended,
                b=self.flag_residual,
                c=self.flag_broad,
                d=self.fwhm_factor,
                e=self.broad_neighbor_fraction,
                f=self.rchi2_limit,
                g=self.flag_rchi2,
                h=self.flag_ncomps)
        self.say(string)

        string = str(
            '\nExclude flagged spectra as possible refit solutions: {}'.format(
                self.exclude_flagged))
        if not self.phase_two:
            self.say(string)

        string = str(
            '\nRefitting:'
            '\n - Blended components: {a}'
            '\n - Negative residual features: {b}'
            '\n - Broad components: {c}'
            '\n   try to refit if FWHM of broadest component in spectrum is:'
            '\n   >= {d} times the FWHM of second broadest component'
            '\n   or'
            '\n   >= {d} times any FWHM in >= {e:.0%} of its neigbors'
            '\n - High reduced chi2 values (> {f}): {g}'
            '\n - Differing number of components: {h}').format(
                a=self.refit_blended,
                b=self.refit_residual,
                c=self.refit_broad,
                d=self.fwhm_factor_refit,
                e=self.broad_neighbor_fraction,
                f=self.rchi2_limit_refit,
                g=self.refit_rchi2,
                h=self.refit_ncomps)
        if not self.phase_two:
            self.say(string)

    def spatial_fitting(self, continuity=False):
        self.phase_two = continuity
        self.initialize()
        self.getting_ready()
        if self.phase_two:
            self.list_n_refit.append([self.length])
            self.check_continuity()
        else:
            self.determine_spectra_for_refitting()

    def define_mask(self, key, limit, flag):
        if not flag:
            return np.zeros(self.length).astype('bool')

        array = np.array(self.decomposition[key])
        array[self.nanMask] = 0
        mask = array > limit
        return mask

    def define_mask_broad_limit(self, flag):
        n_broad = np.zeros(self.length)

        if not flag:
            return n_broad.astype('bool'), n_broad

        for i, fwhms in enumerate(self.decomposition['fwhms_fit']):
            if fwhms is None:
                continue
            n_broad[i] = np.count_nonzero(np.array(fwhms) > self.max_fwhm)
        mask = n_broad > 0
        return mask, n_broad

    def define_mask_broad(self, max_fwhm_factor, flag):
        import scipy.ndimage as ndimage

        def broad_components(values):
            central_value = values[4]
            if np.isnan(central_value):
                return 0
            values = np.delete(values, 4)
            values = values[~np.isnan(values)]
            if values.size == 0:
                return 0
            counter = 0
            for value in values:
                if np.isnan(value):
                    continue
                if central_value > value * max_fwhm_factor and\
                        (central_value - value) > self.fwhm_separation:
                    counter += 1
            if counter > values.size * self.broad_neighbor_fraction:
                return central_value
            return 0

        if not flag:
            return np.zeros(self.length).astype('bool'), np.zeros(self.length)

        broad_1d = np.empty(self.length)
        broad_1d.fill(np.nan)
        mask_broad = np.zeros(self.length)

        for i, fwhms in enumerate(self.decomposition['fwhms_fit']):
            if fwhms is None:
                continue
            if len(fwhms) == 0:
                continue
            broad_1d[i] = max(fwhms)
            if len(fwhms) == 1:
                continue
            fwhms = sorted(fwhms)
            if (fwhms[-1] > max_fwhm_factor * fwhms[-2]) and\
                    (fwhms[-1] - fwhms[-2]) > self.fwhm_separation:
                mask_broad[i] = 1

        broad_2d = broad_1d.astype('float').reshape(self.shape)

        footprint = np.ones((3, 3))

        broad_fwhm_values = ndimage.generic_filter(
            broad_2d, broad_components, footprint=footprint,
            mode='reflect').flatten()
        mask_broad = mask_broad.astype('bool')
        mask_broad += broad_fwhm_values.astype('bool')

        return mask_broad

    def number_of_component_jumps(self, values):
        central_value = values[4]
        if np.isnan(central_value):
            return 0
        values = np.delete(values, 4)
        counter = 0
        for value in values:
            if np.isnan(value):
                continue
            if np.abs(central_value - value) > self.max_jump_comps:
                counter += 1
        return counter

    def define_mask_neighbor_ncomps(self, nanmask_1d, flag):
        import scipy.ndimage as ndimage

        # def expected_components(values):
        #     values = np.delete(values, 4)
        #     denominator = np.count_nonzero(~np.isnan(values))
        #     if denominator > 0:
        #         return np.int(np.round(np.nansum(values) / denominator))
        #     else:
        #         return np.int(0)
        #
        # def number_of_component_jumps(values):
        #     central_value = values[4]
        #     if np.isnan(central_value):
        #         return 0
        #     values = np.delete(values, 4)
        #     counter = 0
        #     for value in values:
        #         if np.isnan(value):
        #             continue
        #         if np.abs(central_value - value) > self.max_jump_comps:
        #             counter += 1
        #     return counter

        if not flag:
            return np.zeros(self.length).astype('bool'), False

        nanmask_1d += self.nanMask  # not really necessary
        nanmask_2d = nanmask_1d.reshape(self.shape)
        ncomps_1d = np.empty(self.length)
        ncomps_1d.fill(np.nan)
        ncomps_1d[~self.nanMask] = np.array(
            self.decomposition['N_components'])[~self.nanMask]
        ncomps_2d = ncomps_1d.astype('float').reshape(self.shape)
        ncomps_2d[nanmask_2d] = np.nan

        footprint = np.ones((3, 3))

        # ncomps_expected = ndimage.generic_filter(
        #     ncomps_2d, expected_components, footprint=footprint,
        #     mode='reflect').flatten()

        ncomps_jumps = ndimage.generic_filter(
            ncomps_2d, self.number_of_component_jumps, footprint=footprint,
            mode='reflect').flatten()

        mask_neighbor = np.zeros(self.length)
        # mask_neighbor[~self.nanMask] = np.abs(
        #     ncomps_expected[~self.nanMask] - ncomps_1d[~self.nanMask]) > self.max_diff_comps
        # #  TODO: rework this?
        # mask_neighbor[~self.nanMask] += ncomps_jumps[~self.nanMask] > self.n_max_jump_comps

        mask_neighbor[~self.nanMask] = ncomps_jumps[~self.nanMask] > self.n_max_jump_comps
        mask_neighbor = mask_neighbor.astype('bool')
        # return mask_neighbor, ncomps_expected
        return mask_neighbor, ncomps_jumps, ncomps_1d

    def determine_spectra_for_flagging(self):
        self.mask_blended = self.define_mask(
            'N_blended', 0, self.flag_blended)
        self.mask_residual = self.define_mask(
            'N_negative_residuals', 0, self.flag_residual)
        self.mask_rchi2_flagged = self.define_mask(
            'best_fit_rchi2', self.rchi2_limit, self.flag_rchi2)
        self.mask_broad_flagged = self.define_mask_broad(
            self.fwhm_factor, self.flag_broad)
        self.mask_broad_limit, self.n_broad = self.define_mask_broad_limit(
            self.flag_broad)

        mask_flagged = self.mask_blended + self.mask_residual\
            + self.mask_broad_flagged + self.mask_rchi2_flagged

        # self.mask_ncomps, self.ncomps_expected =\
        self.mask_ncomps, self.ncomps_jumps, self.ncomps =\
            self.define_mask_neighbor_ncomps(
                mask_flagged.copy(), self.flag_ncomps)

        mask_flagged += self.mask_ncomps
        self.indices_flagged = np.array(
            self.decomposition['index_fit'])[mask_flagged]

        if self.phase_two:
            n_flagged_blended = np.count_nonzero(self.mask_blended)
            n_flagged_residual = np.count_nonzero(self.mask_residual)
            n_flagged_broad = np.count_nonzero(self.mask_broad_flagged)
            n_flagged_rchi2 = np.count_nonzero(self.mask_rchi2_flagged)
            n_flagged_ncomps = np.count_nonzero(self.mask_ncomps)

            text = str(
                "\n Flags:"
                "\n - {a} spectra w/ blended components"
                "\n - {b} spectra w/ negative residual feature"
                "\n - {c} spectra w/ broad feature"
                "\n   (info: {d} spectra w/ a FWHM > {e} channels)"
                "\n - {f} spectra w/ high rchi2 value"
                "\n - {g} spectra w/ differing number of components").format(
                    a=n_flagged_blended,
                    b=n_flagged_residual,
                    c=n_flagged_broad,
                    d=np.count_nonzero(self.mask_broad_limit),
                    e=int(self.max_fwhm),
                    f=n_flagged_rchi2,
                    g=n_flagged_ncomps
                )

            self.say(text)

    def define_mask_refit(self):
        mask_refit = np.zeros(self.length).astype('bool')
        if self.refit_blended:
            mask_refit += self.mask_blended
        if self.refit_residual:
            mask_refit += self.mask_residual
        if self.refit_broad:
            mask_refit += self.mask_broad_refit
        if self.refit_rchi2:
            mask_refit += self.mask_rchi2_refit
        if self.refit_ncomps:
            mask_refit += self.mask_ncomps

        self.indices_refit = np.array(
            self.decomposition['index_fit'])[mask_refit]
        # self.indices_refit = self.indices_refit[10495:10500]  # for debugging
        self.locations_refit = np.take(
            np.array(self.location), self.indices_refit, axis=0)

    def determine_spectra_for_refitting(self):
        self.say('\ndetermine spectra that need refitting...')

        self.determine_spectra_for_flagging()

        self.mask_broad_refit = self.define_mask_broad(
            self.fwhm_factor_refit, self.refit_broad)

        self.mask_rchi2_refit = self.define_mask(
            'best_fit_rchi2', self.rchi2_limit_refit, self.refit_rchi2)

        self.define_mask_refit()

        n_spectra = sum([1 for x in self.decomposition['N_components']
                         if x is not None])
        n_indices_refit = len(self.indices_refit)
        n_flagged_blended = np.count_nonzero(self.mask_blended)
        n_flagged_residual = np.count_nonzero(self.mask_residual)
        n_flagged_broad = np.count_nonzero(self.mask_broad_flagged)
        n_flagged_rchi2 = np.count_nonzero(self.mask_rchi2_flagged)
        n_flagged_ncomps = np.count_nonzero(self.mask_ncomps)

        n_refit_blended, n_refit_residual, n_refit_ncomps = (
            0 for _ in range(3))
        if self.refit_blended:
            n_refit_blended = n_flagged_blended
        if self.refit_residual:
            n_refit_residual = n_flagged_residual
        n_refit_broad = np.count_nonzero(self.mask_broad_refit)
        n_refit_rchi2 = np.count_nonzero(self.mask_rchi2_refit)
        if self.refit_ncomps:
            n_refit_ncomps = n_flagged_ncomps

        n_refit_list = [
            n_refit_blended, n_refit_residual, n_refit_broad,
            n_refit_rchi2, n_refit_ncomps]

        text = str(
            "\n{a} out of {b} spectra ({c:.2%}) selected for refitting:"
            "\n - {d} spectra w/ blended components ({e} flagged)"
            "\n - {f} spectra w/ negative residual feature ({g} flagged)"
            "\n - {h} spectra w/ broad feature ({i} flagged)"
            "\n   (info: {j} spectra w/ a FWHM > {k} channels)"
            "\n - {m} spectra w/ high rchi2 value ({n} flagged)"
            "\n - {o} spectra w/ differing number of components ({p} flagged)").format(
                a=n_indices_refit,
                b=n_spectra,
                c=n_indices_refit/n_spectra,
                d=n_refit_blended,
                e=n_flagged_blended,
                f=n_refit_residual,
                g=n_flagged_residual,
                h=n_refit_broad,
                i=n_flagged_broad,
                j=np.count_nonzero(self.mask_broad_limit),
                k=int(self.max_fwhm),
                m=n_refit_rchi2,
                n=n_flagged_rchi2,
                o=n_refit_ncomps,
                p=n_flagged_ncomps
            )

        self.say(text)

        if self.only_print_flags:
            return
        elif self.stopping_criterion(n_refit_list):
            self.save_final_results()
        else:
            self.list_n_refit.append(n_refit_list)
            self.refitting_iteration += 1
            self.refitting()

    def stopping_criterion(self, n_refit_list):
        """Check if spatial refitting iterations should be stopped."""
        if self.refitting_iteration > self.max_refitting_iteration:
            return True
        if n_refit_list in self.list_n_refit:
            return True
        if self.refitting_iteration > 0:
            stop = True
            for i in range(len(n_refit_list)):
                if n_refit_list[i] < min([n[i] for n in self.list_n_refit]):
                    stop = False
            return stop

    def refitting(self):
        """Refit spectra with multiprocessing routine."""
        self.say('\nstart refit iteration #{}...'.format(
            self.refitting_iteration))

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([self.indices_refit, [self]])

        if self.phase_two:
            results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='refit_phase_2')
        else:
            results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='refit_phase_1')

        print('SUCCESS')

        self.mask_refitted = np.array([0]*self.nIndices)

        keys = ['amplitudes_fit', 'fwhms_fit', 'means_fit',
                'amplitudes_fit_err', 'fwhms_fit_err', 'means_fit_err',
                'best_fit_rchi2', 'best_fit_aicc', 'N_components',
                'gaussians_rchi2', 'gaussians_aicc',
                'N_negative_residuals', 'N_blended']

        count_selected, count_refitted = 0, 0

        for i, item in enumerate(results_list):
            if not isinstance(item, list):
                print("error:", i, item)
                continue

            index, result, indices_neighbors, refit = item
            if refit:
                count_selected += 1
            self.neighbor_indices[index] = indices_neighbors
            if result is not None:
                count_refitted += 1
                self.decomposition['refit_iteration'][index] += 1
                self.mask_refitted[index] = 1
                for key in keys:
                    self.decomposition[key][index] = result[key]
                #  TODO: not sure about this one, check if it causes eternal loop
                # self.neighbor_indices[index] = None

        if count_selected == 0:
            refit_percent = 0
        else:
            refit_percent = count_refitted/count_selected

        text = str(
            "\nResults of the refit iteration:"
            "\nTried to refit {a} spectra"
            "\nSuccessfully refitted {b} spectra ({c:.2%})"
            "\n\n***").format(
                a=count_selected,
                b=count_refitted,
                c=refit_percent)

        self.say(text)

        # TODO: check best procedure for self.iterative_rchi2
        # if self.iterative_rchi2 > self.max_rchi2:
        #     self.iterative_rchi2 -= 0.1

        # if self.broad_neighbor_fraction > 0.625:
        #     self.broad_neighbor_fraction -= 0.125

        if self.phase_two:
            if self.stopping_criterion([count_refitted]):
                self.min_p -= 1/6
                self.list_n_refit = [[self.length]]
                self.mask_refitted = np.array([1]*self.nIndices)
            else:
                self.list_n_refit.append([count_refitted])

            if self.min_p < self.min_weight:
                self.save_final_results()
            else:
                self.check_continuity()
        else:
            self.determine_spectra_for_refitting()

    def determine_neighbor_indices(self, neighbors):
        """Determine indices of all valid neighboring pixels."""
        indices_neighbors = np.array([])
        for neighbor in neighbors:
            indices_neighbors = np.append(
                indices_neighbors, np.ravel_multi_index(neighbor, self.shape)).astype('int')

        #  check if neighboring pixels were selected for refitting, are masked out, or contain no fits and thus cannot be used

        #  whether to exclude all flagged neighboring spectra as well that
        #  were not selected for refitting
        if self.exclude_flagged:
            indices_bad = self.indices_flagged
        else:
            indices_bad = self.indices_refit

        for idx in indices_neighbors:
            if (idx in indices_bad) or (idx in self.nanIndices) or\
                    (self.decomposition['N_components'][idx] == 0):
                indices_neighbors = np.delete(
                    indices_neighbors, np.where(indices_neighbors == idx))
        return indices_neighbors

    def refit_spectrum_phase_1(self, index, i):
        """Refit a spectrum based on neighboring pixels with a good fit.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        i : int
            List index of the entry in the list that is handed over to the
            multiprocessing routine

        Returns
        -------
        list
            A list including the 'index', 'dictResults', and 'indices_neighbors'
            in case of a successful refit (contains 'None' instead of
            'dictResults' if refitting did not work)

        """
        refit = False
        dictResults = None
        flag = 'none'

        loc = self.locations_refit[i]
        spectrum = self.data[index]
        rms = self.errors[index][0]
        signal_ranges = self.signalRanges[index]
        noise_spike_ranges = self.noiseSpikeRanges[index]
        signal_mask = self.signal_mask[index]

        neighbors = get_neighbors(loc, shape=self.shape)
        indices_neighbors = self.determine_neighbor_indices(neighbors)

        if indices_neighbors.size == 0:
            return [index, None, indices_neighbors, refit]

        # skip refitting if there were no changes to the last iteration
        if np.array_equal(indices_neighbors, self.neighbor_indices[index]):
            if self.mask_refitted[indices_neighbors].sum() < 1:
                return [index, None, indices_neighbors, refit]

        if self.refit_broad and self.mask_broad_refit[index]:
            flag = 'broad'
        elif self.refit_blended and self.mask_blended[index]:
            flag = 'blended'
        elif self.refit_residual and self.mask_residual[index]:
            flag = 'residual'

        dictResults, refit = self.try_refit_with_individual_neighbors(
            index, spectrum, rms, indices_neighbors, signal_ranges,
            noise_spike_ranges, signal_mask, flag=flag)

        if dictResults is not None:
            return [index, dictResults, indices_neighbors, refit]

        if indices_neighbors.size > 1:
            dictResults, refit = self.try_refit_with_clustering(
                index, spectrum, rms, indices_neighbors, signal_ranges,
                noise_spike_ranges, signal_mask)

        #
        # dictResults = self.try_refit_with_individual_neighbors(
        #     index, spectrum, rms, indices_neighbors, signal_ranges,
        #     noise_spike_ranges, signal_mask)

        return [index, dictResults, indices_neighbors, refit]

    def try_refit_with_clustering(self, index, spectrum, rms,
                                  indices_neighbors, signal_ranges,
                                  noise_spike_ranges, signal_mask):
        #  try getting intital guesses by clustering
        amps, means, fwhms = self.get_initial_values(indices_neighbors)
        refit = False
        # #  TODO: check if <= 1 is correct; if amps == 1 skip to fit with individual neighbors
        # if len(amps) <= 1:
        #     return [index, None, indices_neighbors]

        for split_fwhm in [False, True]:
            dictComps = self.clustering(
                amps, means, fwhms, split_fwhm=split_fwhm)
            dictComps = self.determine_average_values(
                spectrum, rms, dictComps)

            if len(dictComps.keys()) > 0:
                dictResults = self.gaussian_fitting(
                    spectrum, rms, dictComps, signal_ranges, noise_spike_ranges, signal_mask)
                refit = True
                if dictResults is None:
                    continue
                if self.choose_new_fit(dictResults, index):
                    return dictResults, refit

        return None, refit

    def skip_index_for_refitting(self, index, index_neighbor):
        #  do not refit with neighboring pixel initial guesses if
        #  neighboring pixel was already used but not refit in previous
        #  iteration
        if self.refitting_iteration > 1:
            #  check if spectrum was selected for refitting in any of the #  previous iterations
            if self.neighbor_indices[index] is not None:
                #  check if neighbor was used in that refitting iteration
                if index_neighbor in self.neighbor_indices[index]:
                    #  check if neighbor was refit in previous iteration
                    if not self.mask_refitted[index_neighbor]:
                        return True
        return False

    def try_refit_with_individual_neighbors(self, index, spectrum, rms,
                                            indices_neighbors, signal_ranges,
                                            noise_spike_ranges, signal_mask,
                                            interval=None, n_centroids=None, flag='none', dct_new_fit=None):
        dictComps = None
        refit = False
        sort = np.argsort(
            np.array(self.decomposition['best_fit_rchi2'])[indices_neighbors])

        for index_neighbor in indices_neighbors[sort]:
            if self.skip_index_for_refitting(index, index_neighbor):
                continue

            if flag in ['broad', 'blended', 'residual']:
                dictComps = self.replace_flagged_interval(
                    index, index_neighbor, spectrum, rms, flag=flag)
            elif interval is not None:
                dictComps = self.replace_flagged_interval(
                    index, index_neighbor, spectrum, rms, interval=interval,
                    dct_new_fit=dct_new_fit)
            # if (dictComps is None) and (interval is None):
            else:
                dictComps = self.get_initial_values_from_neighbor(
                    index_neighbor, spectrum, rms)

            if dictComps is None:
                continue

            # dictComps = self.get_initial_values_from_neighbor(
            #     index_neighbor, spectrum, rms)
            dictResults = self.gaussian_fitting(
                spectrum, rms, dictComps, signal_ranges, noise_spike_ranges,
                signal_mask)
            refit = True
            if dictResults is None:
                continue
            if self.choose_new_fit(
                    dictResults, index, dct_new_fit=dct_new_fit,
                    interval=interval, n_centroids=n_centroids):
                return dictResults, refit

        return None, refit

    def get_refit_interval(self, spectrum, rms, amps, fwhms, means, flag):
        if flag == 'broad':
            idx = np.argmax(np.array(fwhms))  # idx of broadest component
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]
        elif flag == 'blended':
            params = amps + fwhms + means
            indices = get_fully_blended_gaussians(params)
            lower = max(0, min(
                np.array(means)[indices] - np.array(fwhms)[indices]))
            upper = max(
                np.array(means)[indices] + np.array(fwhms)[indices])
        elif flag == 'residual':
            dct = self.decomposition['improve_fit_settings'].copy()
            # dct['min_data_snr'] = dct['peak_search_snr']

            best_fit_list = [None for _ in range(10)]
            best_fit_list[0] = amps + fwhms + means
            best_fit_list[2] = len(amps)
            residual = spectrum - combined_gaussian(
                amps, fwhms, means, self.channels)
            best_fit_list[4] = residual

            idx = check_for_negative_residual(
                self.channels, spectrum, rms, best_fit_list, dct, get_idx=True)
            if idx is None:
                return [self.channels[0], self.channels[-1]]
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]

        return [lower, upper]

    def replace_flagged_interval(self, index, index_neighbor, spectrum, rms,
                                 flag='none', interval=None, dct_new_fit=None):
        #  if parts of spectrum were already refit in this iteration (stage 2)
        if dct_new_fit is not None:
            amps = dct_new_fit['amplitudes_fit']
            fwhms = dct_new_fit['fwhms_fit']
            means = dct_new_fit['means_fit']

            amps_err = dct_new_fit['amplitudes_fit_err']
            fwhms_err = dct_new_fit['fwhms_fit_err']
            means_err = dct_new_fit['means_fit_err']
        else:
            amps = self.decomposition['amplitudes_fit'][index]
            fwhms = self.decomposition['fwhms_fit'][index]
            means = self.decomposition['means_fit'][index]

            amps_err = self.decomposition['amplitudes_fit_err'][index]
            fwhms_err = self.decomposition['fwhms_fit_err'][index]
            means_err = self.decomposition['means_fit_err'][index]

        #  remove the broadest component and all components overlapping with it

        if interval is None:
            interval = self.get_refit_interval(
                spectrum, rms, amps, fwhms, means, flag=flag)
        indices, interval = self.components_in_interval(
            fwhms, means, interval)

        amps, fwhms, means = remove_components_from_list(
            [amps, fwhms, means], indices)
        amps_err, fwhms_err, means_err = remove_components_from_list(
            [amps_err, fwhms_err, means_err], indices)

        amps_new = self.decomposition['amplitudes_fit'][index_neighbor]
        fwhms_new = self.decomposition['fwhms_fit'][index_neighbor]
        means_new = self.decomposition['means_fit'][index_neighbor]

        amps_err_new = self.decomposition['amplitudes_fit'][index_neighbor]
        fwhms_err_new = self.decomposition['fwhms_fit'][index_neighbor]
        means_err_new = self.decomposition['means_fit'][index_neighbor]

        #  get new initial guesses for removed components from neighbor

        indices, interval = self.components_in_interval(
            fwhms_new, means_new, interval)

        if len(indices) == 0:
            return None

        remove_indices = np.delete(np.arange(len(amps_new)), indices)
        amps_new, fwhms_new, means_new = remove_components_from_list(
            [amps_new, fwhms_new, means_new], remove_indices)
        amps_err_new, fwhms_err_new, means_err_new =\
            remove_components_from_list(
                [amps_err_new, fwhms_err_new, means_err_new], remove_indices)

        if len(amps_new) == 0:
            return None

        #  get best fit for the interval that contained the removed components
        idx_lower = int(interval[0])
        idx_upper = int(interval[1]) + 2

        #  add key to dictComps
        dictComps = {}
        for amp, fwhm, mean, mean_err in zip(
                amps_new, fwhms_new, means_new, means_err_new):
            dictComps = self.add_initial_value_to_dict(
                dictComps, spectrum[idx_lower:idx_upper], rms, amp, fwhm,
                mean - idx_lower, mean_err)

        #  get best fit for subsection of spectrum
        channels = np.arange(len(spectrum[idx_lower:idx_upper]))

        dictFit = self.gaussian_fitting(
            spectrum[idx_lower:idx_upper], rms, dictComps, None, None, None,
            params_only=True, channels=channels)

        if dictFit is None:
            return None

        dictComps = {}
        for amp, fwhm, mean, mean_err in zip(
                dictFit['amplitudes_fit'], dictFit['fwhms_fit'],
                dictFit['means_fit'], dictFit['means_fit_err']):
            dictComps = self.add_initial_value_to_dict(
                dictComps, spectrum, rms, amp, fwhm, mean + idx_lower, mean_err)

        for amp, fwhm, mean, mean_err in zip(amps, fwhms, means, means_err):
            dictComps = self.add_initial_value_to_dict(
                dictComps, spectrum, rms, amp, fwhm, mean, mean_err)

        return dictComps

    def components_in_interval(self, fwhms, means, interval):
        """Find indices of components overlapping with interval.

        Component i is selected if means[i] +/- fwhms[i] overlaps with the
        interval.
        """
        lower_interval, upper_interval = interval.copy()
        lower_interval_new, upper_interval_new = interval.copy()
        indices = []

        for i, (mean, fwhm) in enumerate(zip(means, fwhms)):
            lower = max(0, mean - fwhm)
            upper = mean + fwhm
            if (lower_interval <= lower <= upper_interval) or\
                    (lower_interval <= upper <= upper_interval):
                lower_interval_new = min(lower_interval_new, lower)
                upper_interval_new = max(upper_interval_new, upper)
                indices.append(i)
        return indices, [lower_interval_new, upper_interval_new]

    def add_initial_value_to_dict(self, dictComps, spectrum, rms,
                                  amp, fwhm, mean, mean_err):
        stddev = fwhm / 2.354820045

        idx_low = max(0, int(mean - stddev))
        idx_upp = int(mean + stddev) + 2
        amp_max = np.max(spectrum[idx_low:idx_upp])

        #  TODO: add here also mean +/- stddev??
        mean_min = min(mean - self.mean_separation, mean - mean_err)
        mean_min = max(0, mean_min)  # prevent negative values
        mean_max = max(mean + self.mean_separation, mean + mean_err)

        # fwhm_min = max(0., fwhm - self.fwhm_separation)
        # fwhm_max = fwhm + self.fwhm_separation

        keyname = str(len(dictComps) + 1)
        dictComps[keyname] = {}
        dictComps[keyname]['amp_ini'] = amp
        dictComps[keyname]['mean_ini'] = mean
        dictComps[keyname]['fwhm_ini'] = fwhm

        dictComps[keyname]['amp_bounds'] = [0., 1.1*amp_max]
        dictComps[keyname]['mean_bounds'] = [mean_min, mean_max]
        dictComps[keyname]['fwhm_bounds'] = [0., None]
        # dictComps[keyname]['fwhm_bounds'] = [fwhm_min, fwhm_max]

        return dictComps

    def get_dictionary_value(self, key, index, dct_new_fit=None):
        ""
        if dct_new_fit is not None:
            return dct_new_fit[key]
        else:
            return self.decomposition[key][index]

    def get_flags(self, dictResults, index, key='None', flag=None,
                  dct_new_fit=None):
        flag_old, flag_new = (0 for _ in range(2))

        if not flag:
            return flag_old, flag_new

        n_old = self.get_dictionary_value(
            key, index, dct_new_fit=dct_new_fit)
        n_new = dictResults[key]
        #  flag if old fitting results showed flagged feature
        if n_old > 0:
            flag_old = 1
        #  punish new fit if it contains more of the flagged features
        if n_new > n_old:
            flag_new = flag_old + 1
        #  same flags if the new and old fitting results show the same
        #  number of features
        elif n_new == n_old:
            flag_new = flag_old

        return flag_old, flag_new

    def get_flags_rchi2(self, dictResults, index, dct_new_fit=None):
        def rchi2_flag_value(rchi2):
            if rchi2 <= self.rchi2_limit:
                return 0
            numerator = rchi2 - self.rchi2_limit
            denominator = self.rchi2_limit / 4
            return int(numerator / denominator) + 1

        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_rchi2:
            return flag_old, flag_new

        rchi2_old = self.get_dictionary_value(
            'best_fit_rchi2', index, dct_new_fit=dct_new_fit)
        rchi2_new = dictResults['best_fit_rchi2']

        #  do not punish fit if it is closer to rchi2 = 1 and thus likely less "overfit"
        if max(rchi2_old, rchi2_new) < self.rchi2_limit:
            if abs(rchi2_new - 1) < abs(rchi2_old - 1):
                flag_old = 1
        else:
            flag_old = rchi2_flag_value(rchi2_old)
            flag_new = rchi2_flag_value(rchi2_new)

            if flag_old == flag_new:
                if rchi2_old < rchi2_new:
                    flag_new += 1
                else:
                    flag_old += 1

        return flag_old, flag_new

    def get_flags_broad(self, dictResults, index, dct_new_fit=None):
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_broad:
            return flag_old, flag_new

        if self.mask_broad_flagged[index]:
            flag_old = 1
            fwhm_max_old = max(self.get_dictionary_value(
                'fwhms_fit', index, dct_new_fit=dct_new_fit))
            fwhm_max_new = max(np.array(dictResults['fwhms_fit']))
            #  no changes to the fit
            if fwhm_max_new == fwhm_max_old:
                flag_new = 1
            #  punish fit if component got even broader
            elif fwhm_max_new > fwhm_max_old:
                flag_new = 2
        else:
            fwhms = dictResults['fwhms_fit']
            if len(fwhms) > 1:
                #  punish fit if broad component was introduced
                fwhms = sorted(dictResults['fwhms_fit'])
                if (fwhms[-1] > self.fwhm_factor * fwhms[-2]) and\
                        (fwhms[-1] - fwhms[-2]) > self.fwhm_separation:
                    flag_new = 1

        return flag_old, flag_new

    def get_flags_ncomps(self, dictResults, index, dct_new_fit=None):
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_ncomps:
            return flag_old, flag_new

        njumps_old = self.ncomps_jumps[index]

        loc = self.location[index]
        indices = get_neighbors(
            loc, exclude_p=False, shape=self.shape, nNeighbors=1,
            get_indices=True)
        ncomps = self.ncomps[indices]
        ncomps[4] = self.get_dictionary_value(
            'N_components', index, dct_new_fit=dct_new_fit)
        njumps_new = self.number_of_component_jumps(ncomps)

        # ncomps_expected = self.ncomps_expected[index]
        # ncomps_old = self.get_dictionary_value(
        #     'N_components', index, dct_new_fit=dct_new_fit)
        # # ncomps_old = self.decomposition['N_components'][index]
        # ncomps_new = dictResults['N_components']
        # # ncomps_difference_old = max(
        # #     self.max_diff_comps, abs(ncomps_old - ncomps_expected))\
        # #     - self.max_diff_comps
        # # if ncomps_difference_old > 0:
        # #     flag_old = 1
        # # ncomps_difference_new = max(
        # #     self.max_diff_comps, abs(ncomps_new - ncomps_expected))\
        # #     - self.max_diff_comps
        # # if ncomps_difference_new > 0:
        # #     flag_new = 1
        # # if ncomps_difference_new > ncomps_difference_old:
        # #     flag_new += 1
        # ncomps_difference_old = abs(ncomps_old - ncomps_expected)
        # ncomps_difference_new = abs(ncomps_new - ncomps_expected)
        #
        # if ncomps_difference_old > self.max_diff_comps:
        #     flag_old = 1
        # if ncomps_difference_new > self.max_diff_comps:
        #     flag_new = flag_old
        # if ncomps_difference_new > ncomps_difference_old:
        #     flag_new += 1

        if njumps_old > self.n_max_jump_comps:
            flag_old = 1
        if njumps_new > self.n_max_jump_comps:
            flag_new = 1
        if njumps_new > njumps_old:
            flag_new += 1

        return flag_old, flag_new

    def get_flags_centroids(self, dictResults, index, dct_new_fit=None,
                            interval=None, n_centroids=None):
        flag_old, flag_new = (0 for _ in range(2))

        if interval is None:
            return flag_old, flag_new

        means_old = self.get_dictionary_value(
            'means_fit', index, dct_new_fit=dct_new_fit)
        means_new = dictResults['means_fit']

        flag_old, flag_new = (2 for _ in range(2))
        # flag_old, flag_new = (1 for _ in range(2))

        n_centroids_old = self.number_of_values_in_interval(means_old, interval)
        n_centroids_new = self.number_of_values_in_interval(means_new, interval)

        if n_centroids_new == n_centroids:
            flag_new = 0

        # if abs(n_centroids_new - n_centroids) < abs(
        #         n_centroids_old - n_centroids):
        #     flag_new = 1

        return flag_old, flag_new

    def choose_new_fit(self, dictResults, index, dct_new_fit=None,
                       interval=None, n_centroids=None):
        flag_blended_old, flag_blended_new = self.get_flags(
            dictResults, index, key='N_blended', flag=self.flag_blended,
            dct_new_fit=dct_new_fit)

        flag_residual_old, flag_residual_new = self.get_flags(
            dictResults, index, key='N_negative_residuals',
            flag=self.flag_residual, dct_new_fit=dct_new_fit)

        flag_rchi2_old, flag_rchi2_new = self.get_flags_rchi2(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_broad_old, flag_broad_new = self.get_flags_broad(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_ncomps_old, flag_ncomps_new = self.get_flags_ncomps(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_centroids_old, flag_centroids_new = self.get_flags_centroids(
            dictResults, index, dct_new_fit=dct_new_fit,
            interval=interval, n_centroids=n_centroids)

        if (n_centroids is not None) and (flag_centroids_new > 0):
            return False

        n_flags_old = flag_blended_old\
            + flag_residual_old\
            + flag_broad_old\
            + flag_rchi2_old\
            + flag_ncomps_old\
            + flag_centroids_old

        n_flags_new = flag_blended_new\
            + flag_residual_new\
            + flag_broad_new\
            + flag_rchi2_new\
            + flag_ncomps_new\
            + flag_centroids_new

        # if index == 2850:
        #     print('\n', interval, n_centroids)
        #     print('flags:', n_flags_old, n_flags_new)
        #     print('blended:', flag_blended_old, flag_blended_new)
        #     print('residual:', flag_residual_old, flag_residual_new)
        #     print('broad:', flag_broad_old, flag_broad_new)
        #     print('rchi2:', flag_rchi2_old, flag_rchi2_new)
        #     print('ncomps:', flag_ncomps_old, flag_ncomps_new)
        #     print('n centroids:', flag_centroids_old, flag_centroids_new)

        if n_flags_new > n_flags_old:
            return False

        aicc_old = self.get_dictionary_value(
            'best_fit_aicc', index, dct_new_fit=dct_new_fit)
        aicc_new = dictResults['best_fit_aicc']
        residual_signal_mask = dictResults['residual_signal_mask']

        # if (aicc_new > aicc_old) and (n_flags_new == n_flags_old):
        #     return False
        if (aicc_new > aicc_old):
            statistic, pvalue = normaltest(residual_signal_mask)
            if pvalue < self.min_pvalue:
                return False

        # if index == 2850:
        #     print('aicc:', aicc_old, aicc_new)

        return True

    def get_initial_values(self, indices_neighbors):
        amps, means, fwhms = (np.array([]) for i in range(3))

        for i in indices_neighbors:
            amps = np.append(amps, self.decomposition['amplitudes_fit'][i])
            means = np.append(means, self.decomposition['means_fit'][i])
            fwhms = np.append(fwhms, self.decomposition['fwhms_fit'][i])

        sorted_indices = np.argsort(means)
        return amps[sorted_indices], means[sorted_indices], fwhms[sorted_indices]

    def clustering(self, amps_tot, means_tot, fwhms_tot, split_fwhm=True):
        #  cluster with regards to mean positions only
        means_diff = np.append(np.array([0.]), means_tot[1:] - means_tot[:-1])

        split_indices = np.where(means_diff > self.mean_separation)[0]
        split_means_tot = np.split(means_tot, split_indices)
        split_fwhms_tot = np.split(fwhms_tot, split_indices)
        split_amps_tot = np.split(amps_tot, split_indices)

        dictComps = {}

        for amps, fwhms, means in zip(
                split_amps_tot, split_fwhms_tot, split_means_tot):
            if (len(means) == 1) or not split_fwhm:
                key = "{}".format(len(dictComps) + 1)
                dictComps[key] = {
                    "amps": amps, "means": means, "fwhms": fwhms}
                continue

            #  also cluster with regards to fwhm values
            lst_of_grouped_indices = []
            for i in range(len(means)):
                grouped_indices_means = np.where(
                    (np.abs(means - means[i]) < self.mean_separation))[0]
                grouped_indices_fwhms = np.where(
                    (np.abs(fwhms - fwhms[i]) < self.fwhm_separation))[0]
                ind = np.intersect1d(
                    grouped_indices_means, grouped_indices_fwhms)
                lst_of_grouped_indices.append(list(ind))

            #  merge all sublists from lst_of_grouped_indices that share common indices
            G = to_graph(lst_of_grouped_indices)
            lst = list(connected_components(G))
            lst = [list(l) for l in lst]

            for sublst in lst:
                key = "{}".format(len(dictComps) + 1)
                dictComps[key] = {"amps": amps[sublst],
                                  "means": means[sublst],
                                  "fwhms": fwhms[sublst]}

        dictCompsOrdered = collections.OrderedDict()
        for i, k in enumerate(sorted(dictComps,
                                     key=lambda k: len(dictComps[k]['amps']),
                                     reverse=True)):
            dictCompsOrdered[str(i + 1)] = dictComps[k]

        return dictCompsOrdered

    def get_initial_values_from_neighbor(self, i, spectrum, rms):
        dictComps = {}

        for key in range(self.decomposition['N_components'][i]):
            amp = self.decomposition['amplitudes_fit'][i][key]
            mean = self.decomposition['means_fit'][i][key]
            mean_err = self.decomposition['means_fit_err'][i][key]
            fwhm = self.decomposition['fwhms_fit'][i][key]
            stddev = fwhm / 2.354820045

            idx_low = max(0, int(mean - stddev))
            idx_upp = int(mean + stddev) + 2
            amp_max = np.max(spectrum[idx_low:idx_upp])

            mean_min = min(mean - self.mean_separation, mean - mean_err)
            mean_min = max(0, mean_min)  # prevent negative values
            mean_max = max(mean + self.mean_separation, mean + mean_err)

            keyname = str(key + 1)
            dictComps[keyname] = {}
            dictComps[keyname]['amp_ini'] = amp
            dictComps[keyname]['mean_ini'] = mean
            dictComps[keyname]['fwhm_ini'] = fwhm

            dictComps[keyname]['amp_bounds'] = [0., 1.1*amp_max]
            dictComps[keyname]['mean_bounds'] = [mean_min, mean_max]
            dictComps[keyname]['fwhm_bounds'] = [0., None]

        return dictComps

    def determine_average_values(self, spectrum, rms, dictComps):
        for key in dictComps.copy().keys():
            amps = np.array(dictComps[key]['amps'])
            #  TODO: also exclude all clusters with two points?
            if len(amps) == 1:
                dictComps.pop(key)
                continue
            means = np.array(dictComps[key]['means'])
            fwhms = np.array(dictComps[key]['fwhms'])

            # TODO: take the median instead of the mean??
            amp_ini = np.mean(amps)
            mean_ini = np.mean(means)
            fwhm_ini = np.mean(fwhms)
            stddev_ini = fwhm_ini / 2.354820045

            # TODO: change stddev_ini to fwhm_ini?
            idx_low = max(0, int(mean_ini - stddev_ini))
            idx_upp = int(mean_ini + stddev_ini) + 2

            amp_max = np.max(spectrum[idx_low:idx_upp])
            if amp_max < self.snr*rms:
                dictComps.pop(key)
                continue

            #  determine fitting constraints for mean value
            lower_interval = max(
                abs(mean_ini - np.min(means)), self.mean_separation)
            mean_min = max(0, mean_ini - lower_interval)

            upper_interval = max(
                abs(mean_ini - np.max(means)), self.mean_separation)
            mean_max = mean_ini + upper_interval

            # #  determine fitting constraints for fwhm value
            # lower_interval = max(
            #     abs(fwhm_ini - np.min(fwhms)), self.fwhm_separation)
            # fwhm_min = max(self.min_fwhm, fwhm_ini - lower_interval)
            #
            # upper_interval = max(
            #     abs(fwhm_ini - np.max(fwhms)), self.fwhm_separation)
            # fwhm_max = fwhm_ini + upper_interval

            dictComps[key]['amp_ini'] = amp_ini
            dictComps[key]['mean_ini'] = mean_ini
            dictComps[key]['fwhm_ini'] = fwhm_ini

            dictComps[key]['amp_bounds'] = [0., 1.1*amp_max]
            dictComps[key]['mean_bounds'] = [mean_min, mean_max]
            dictComps[key]['fwhm_bounds'] = [0., None]
            # dictComps[key]['fwhm_bounds'] = [fwhm_min, fwhm_max]
        return dictComps

    def gaussian_fitting(self, spectrum, rms, dictComps, signal_ranges,
                         noise_spike_ranges, signal_mask, params_only=False,
                         channels=None):
        if channels is None:
            n_channels = self.n_channels
            channels = self.channels
        else:
            n_channels = len(channels)

        errors = np.ones(n_channels)*rms

        dct = self.decomposition['improve_fit_settings'].copy()
        dct['max_amp'] = dct['max_amp_factor'] * np.max(spectrum)
        # dct['min_fit_snr'] = dct['min_snr']
        # dct['min_data_snr'] = dct['peak_search_snr']

        params, params_min, params_max = ([] for _ in range(3))
        for key in ['amp', 'fwhm', 'mean']:
            for nr in dictComps.keys():
                params.append(dictComps[nr]['{}_ini'.format(key)])
                params_min.append(dictComps[nr]['{}_bounds'.format(key)][0])
                params_max.append(dictComps[nr]['{}_bounds'.format(key)][1])

        best_fit_list = get_best_fit(
            channels, spectrum, errors, params, dct, first=True,
            signal_ranges=signal_ranges, signal_mask=signal_mask,
            params_min=params_min, params_max=params_max)

        # #  get a new best fit that is unconstrained
        # params = best_fit_list[0]
        #
        # best_fit_list = get_best_fit(
        #     self.channels, spectrum, errors, params, dct, first=True,
        #     signal_ranges=signal_ranges, signal_mask=signal_mask)

        #  TODO: set fitted_residual_peaks to input offset positions??
        fitted_residual_peaks = []
        new_fit = True

        while new_fit:
            best_fit_list[7] = False
            best_fit_list, fitted_residual_peaks = check_for_peaks_in_residual(
                channels, spectrum, errors, best_fit_list, dct,
                fitted_residual_peaks, signal_ranges=signal_ranges,
                signal_mask=signal_mask)
            new_fit = best_fit_list[7]

        params = best_fit_list[0]
        params_errs = best_fit_list[1]
        ncomps = best_fit_list[2]
        best_fit = best_fit_list[3]
        residual_signal_mask = best_fit_list[4][signal_mask]
        rchi2 = best_fit_list[5]
        aicc = best_fit_list[6]

        if ncomps == 0:
            return None

        amps, fwhms, means = split_params(params, ncomps)
        amps_errs, fwhms_errs, means_errs = split_params(params_errs, ncomps)

        keys = ['amplitudes_fit', 'fwhms_fit', 'means_fit',
                'amplitudes_fit_err', 'fwhms_fit_err', 'means_fit_err']
        vals = [amps, fwhms, means, amps_errs, fwhms_errs, means_errs]
        dictResults = {key: val for key, val in zip(keys, vals)}

        if params_only:
            return dictResults

        mask = mask_covering_gaussians(
            means, fwhms, n_channels, remove_intervals=noise_spike_ranges)
        rchi2_gauss, aicc_gauss = goodness_of_fit(
            spectrum, best_fit, rms, ncomps, mask=mask, get_aicc=True)

        N_blended = get_fully_blended_gaussians(params, get_count=True)
        N_negative_residuals = check_for_negative_residual(
            channels, spectrum, rms, best_fit_list, dct, get_count=True)

        keys = ["best_fit_rchi2", "best_fit_aicc", "residual_signal_mask",
                "gaussians_rchi2", "gaussians_aicc",
                "N_components", "N_blended", "N_negative_residuals"]
        values = [rchi2, aicc, residual_signal_mask,
                  rchi2_gauss, aicc_gauss,
                  ncomps, N_blended, N_negative_residuals]
        for key, val in zip(keys, values):
            dictResults[key] = val

        return dictResults

    def save_final_results(self):
        pathToFile = os.path.join(
            self.dirname, '{}.pickle'.format(self.finFilename))
        pickle.dump(self.decomposition, open(pathToFile, 'wb'), protocol=2)
        self.say(">> saved '{}' in {}".format(self.finFilename, self.dirname))

    def say(self, message):
        """Diagnostic messages."""
        if self.log_output:
            self.logger.info(message)
        if self.verbose:
            print(message)

    #
    #  --- Phase 2: Refitting towards coherence in centroid positions ---
    #

    def get_centroid_interval(self, dct):
        """Calculate interval spanned by centroids from clustering results."""
        dct['means_interval'] = {}
        for key in dct['clustering']:
            mean_min = min(dct['clustering'][key]['means'])
            mean_min = max(0, mean_min - self.mean_separation / 2)
            mean_max = max(dct['clustering'][key]['means'])\
                + self.mean_separation / 2
            dct['means_interval'][key] = [mean_min, mean_max]
        return dct

    def components_per_interval(self, dct):
        """Calculate how many components neighboring fits had per clustered centroid interval."""
        dct['ncomps_per_interval'] = {}
        for key in dct['clustering']:
            ncomps_per_interval = []
            means_interval = dct['means_interval'][key]

            for idx in dct['indices_neighbors']:
                means = self.decomposition['means_fit'][idx]
                if means is None:
                    ncomps_per_interval.append(0)
                    continue
                if len(means) == 0:
                    ncomps_per_interval.append(0)
                    continue
                condition_1 = means_interval[0] < np.array(means)
                condition_2 = means_interval[1] > np.array(means)
                mask = np.logical_and(condition_1, condition_2)
                ncomps_per_interval.append(np.count_nonzero(mask))
            dct['ncomps_per_interval'][key] = ncomps_per_interval

        return dct

    def get_n_centroid(self, n_centroids, weights):
        """Calculate expected value for number of centroids per clustered centroid interval."""
        choices = list(set(n_centroids))
        #
        #  first, check only immediate neighboring spectra
        #
        mask_weight = weights > 1
        n_neighbors = np.count_nonzero(mask_weight)

        counts_choices = []
        for choice in choices:
            if choice == 0:
                counts_choices.append(0)
                continue
            count_choice = np.count_nonzero(n_centroids[mask_weight] == choice)
            counts_choices.append(count_choice)

        if np.max(counts_choices) > 0.5 * n_neighbors:
            idx = np.argmax(counts_choices)
            return choices[idx]
        #
        #  include additional neighbors that are two pixels away
        #
        weights_choices = []
        for choice in choices:
            if choice == 0:
                weights_choices.append(0)
                continue
            mask = n_centroids == choice
            weights_choices.append(sum(weights[mask]))
        idx = np.argmax(weights_choices)
        return choices[idx]

    def compute_weights(self, dct, weights):
        """Calculate weight of required components per centroid interval."""
        weights = np.array(weights)*(1/6)

        dct['factor_required'] = {}
        dct['n_centroids'] = {}
        for key in dct['clustering']:
            array = np.array(dct['ncomps_per_interval'][key])
            dct['n_centroids'][key] = self.get_n_centroid(array, weights)
            array = array.astype('bool')
            dct['factor_required'][key] = sum(array * weights)
        return dct

    def sort_out_keys(self, dct):
        """Keep only centroid intervals that have a certain minimum weight."""
        dct_new = {}
        keys = ['indices_neighbors', 'weights', 'means_interval',
                'n_centroids', 'factor_required']
        dct_new = {key: {} for key in keys}
        dct_new['indices_neighbors'] = dct['indices_neighbors']
        dct_new['weights'] = dct['weights']

        means_interval = []
        for key in dct['factor_required']:
            if dct['factor_required'][key] > self.min_p:
                means_interval.append(dct['means_interval'][key])

        dct_new['means_interval'] = means_interval
        return dct_new

    def add_key_to_dict(self, dct, key='means_interval', val=None):
        """Add a new key number & value to an existing dictionary key."""
        key_new = str(len(dct[key]) + 1)
        dct[key][key_new] = val
        return dct

    def merge_dictionaries(self, dct_1, dct_2):
        """Merge two dictionaries to a single one and calculate new centroid intervals."""
        dct_merged = {key: {} for key in [
            'factor_required', 'n_centroids', 'means_interval']}

        for key in ['indices_neighbors', 'weights']:
            dct_merged[key] = []
            for dct in [dct_1, dct_2]:
                dct_merged[key] = np.append(dct_merged[key], dct[key])

        key = 'means_interval'
        intervals = dct_1[key] + dct_2[key]
        dct_merged[key] = self.merge_intervals(intervals)

        return dct_merged

    def merge_intervals(self, intervals):
        """Merge overlapping intervals.

        Original code by amon: https://codereview.stackexchange.com/questions/69242/merging-overlapping-intervals
        """
        sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
        merged = []

        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                # test for intersection between lower and higher:
                # we know via sorting that lower[0] <= higher[0]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound)  # replace by merged interval
                else:
                    merged.append(higher)
        return merged

    def combine_directions(self, dct):
        """Combine directions and get master dictionary."""
        dct_hv = self.merge_dictionaries(
            dct['horizontal'].copy(), dct['vertical'].copy())

        dct_dd = self.merge_dictionaries(
            dct['diagonal_ul'].copy(), dct['diagonal_ur'].copy())

        dct_total = self.merge_dictionaries(dct_hv, dct_dd)

        intervals = dct_total['means_interval'].copy()
        dct_total['means_interval'] = {}
        for interval in intervals:
            dct_total = self.add_key_to_dict(
                dct_total, key='means_interval', val=interval)

        #  add buffer of half the mean_separation to left and right of means_interval
        for key in dct_total['means_interval']:
            lower, upper = dct_total['means_interval'][key]
            lower = max(0, lower - self.mean_separation / 2)
            upper = upper + self.mean_separation / 2
            dct_total['means_interval'][key] = [lower, upper]

        for key in ['indices_neighbors', 'weights']:
            dct_total[key] = dct_total[key].astype('int')
        #
        #  Calculate number of centroids per centroid interval of neighbors
        #  and estimate the expected number of centroids for interval
        #
        dct_total['n_comps'] = {}
        dct_total['n_centroids'] = {}
        for key in dct_total['means_interval']:
            dct_total['n_comps'][key] = []
            lower, upper = dct_total['means_interval'][key]
            for idx in dct_total['indices_neighbors']:
                means = self.decomposition['means_fit'][idx]
                ncomps = 0
                for mean in means:
                    if lower < mean < upper:
                        ncomps += 1
                dct_total['n_comps'][key].append(ncomps)
            dct_total['n_centroids'][key] = self.get_n_centroid(
                 np.array(dct_total['n_comps'][key]), dct_total['weights'])

        return dct_total

    def get_weights(self, indices, idx, direction):
        """Define the weights for neighboring spectra (2 if immediate neighbor, 1 if two pixels away)."""
        chosen_weights, indices_neighbors = ([] for _ in range(2))
        possible_indices = np.arange(len(indices))
        index = np.where(indices == idx)[0][0]
        indices = np.delete(indices, index)
        possible_indices -= index
        possible_indices = np.delete(possible_indices, index)

        normalization_factor = 1 / (2 * (self.weight_factor + 1))
        w_2 = normalization_factor
        w_1 = self.weight_factor * normalization_factor

        if direction in ['horizontal', 'vertical']:
            weights = [w_2, w_1, w_1, w_2]
        else:
            weights = np.array([w_2 / np.sqrt(8), w_1 / np.sqrt(2),
                                w_1 / np.sqrt(2), w_2 / np.sqrt(8)])

        counter = 0
        for i, weight in zip([-2, -1, 1, 2], weights):
            if i in possible_indices:
                idx_neighbor = indices[counter]
                counter += 1
                if self.decomposition['N_components'][idx_neighbor] is not None:
                    if self.decomposition['N_components'][idx_neighbor] != 0:
                        indices_neighbors.append(idx_neighbor)
                        chosen_weights.append(weight)
        return np.array(indices_neighbors), np.array(chosen_weights)

    def check_continuity_centroids(self, idx, loc):
        dct, dct_total = [{} for _ in range(2)]

        for direction in [
                'horizontal', 'vertical', 'diagonal_ul', 'diagonal_ur']:
            indices_neighbors_and_central = get_neighbors(
                loc, exclude_p=False, shape=self.shape, nNeighbors=2,
                direction=direction, get_indices=True)

            indices_neighbors, weights = self.get_weights(
                indices_neighbors_and_central, idx, direction)

            if len(indices_neighbors) == 0:
                dct_total[direction] = {key: {} for key in [
                    'indices_neighbors', 'weights', 'means_interval',
                    'n_centroids', 'factor_required']}
                dct_total[direction]['means_interval'] = []
                dct_total[direction]['indices_neighbors'] = np.array([])
                dct_total[direction]['weights'] = np.array([])
                continue

            dct['indices_neighbors'] = indices_neighbors
            dct['weights'] = weights

            amps, means, fwhms = self.get_initial_values(indices_neighbors)
            dct['clustering'] = self.clustering(
                amps, means, fwhms, split_fwhm=False)
            dct = self.get_centroid_interval(dct)
            dct = self.components_per_interval(dct)
            dct = self.compute_weights(dct, weights)
            dct = self.sort_out_keys(dct)

            #  TODO: check why copy() is needed here
            dct_total[direction] = dct.copy()

        dct_total = self.combine_directions(dct_total)
        return dct_total

    def check_for_required_components(self, idx, dct):
        dct_refit = {key: {} for key in ['n_centroids', 'means_interval']}
        for key in ['indices_neighbors', 'weights']:
            dct_refit[key] = dct[key]
        means = self.decomposition['means_fit'][idx]
        for key in dct['means_interval']:
            ncomps_expected = dct['n_centroids'][key]
            interval = dct['means_interval'][key]
            ncomps = self.number_of_values_in_interval(means, interval)
            if ncomps != ncomps_expected:
                dct_refit = self.add_key_to_dict(
                    dct_refit, key='n_centroids', val=ncomps_expected)
                dct_refit = self.add_key_to_dict(
                    dct_refit, key='means_interval', val=interval)
        return dct_refit

    def number_of_values_in_interval(self, lst, interval):
        """Return number of points in list that lie in interval."""
        lower, upper = interval
        array = np.array(lst)
        mask = np.logical_and(array > lower, array < upper)
        return np.count_nonzero(mask)

    def select_neighbors_to_use_for_refit(self, dct):
        from functools import reduce

        mask = dct['weights'] == 2
        indices = dct['indices_neighbors'][mask]
        dct['indices_refit'] = {}

        for key in dct['means_interval']:
            interval = dct['means_interval'][key]
            ncomps_expected = dct['n_centroids'][key]

            indices_refit = []
            for idx in indices:
                means = self.decomposition['means_fit'][idx]
                ncomps = self.number_of_values_in_interval(means, interval)
                if ncomps == ncomps_expected:
                    indices_refit.append(idx)

            dct['indices_refit'][key] = indices_refit

        indices_refit_all_individual = list(dct['indices_refit'].values())
        if len(indices_refit_all_individual) > 1:
            indices_refit_all = reduce(
                np.intersect1d, indices_refit_all_individual)
            dct['indices_refit_all'] = indices_refit_all
        else:
            dct['indices_refit_all'] = indices_refit_all_individual[0]

        return dct

    def determine_all_neighbors(self):
        self.say("\ndetermine neighbors for all spectra...")

        mask_all = np.array(
            [0 if x is None else 1 for x in self.decomposition['N_components']]).astype('bool')
        self.indices_all = np.array(
            self.decomposition['index_fit'])[mask_all]
        self.locations_all = np.take(
            np.array(self.location), self.indices_all, axis=0)

        for i, loc in tqdm(zip(self.indices_all, self.locations_all)):
            indices_neighbors_total = np.array([])
            for direction in [
                    'horizontal', 'vertical', 'diagonal_ul', 'diagonal_ur']:
                indices_neighbors = get_neighbors(
                    loc, exclude_p=True, shape=self.shape, nNeighbors=2,
                    direction=direction, get_indices=True)
                indices_neighbors_total = np.append(
                    indices_neighbors_total, indices_neighbors)
            indices_neighbors_total = indices_neighbors_total.astype('int')
            self.neighbor_indices_all[i] = indices_neighbors_total

    def check_indices_refit(self):
        self.say('\ncheck which spectra require refitting...')
        if self.refitting_iteration == 1:
            self.determine_all_neighbors()

        if np.count_nonzero(self.mask_refitted) == len(self.mask_refitted):
            self.indices_refit = self.indices_all.copy()
            self.locations_refit = self.locations_all.copy()
            return

        indices_remove = np.array([])

        for i in self.indices_all:
            if np.count_nonzero(
                    self.mask_refitted[self.neighbor_indices_all[i]]) == 0:
                indices_remove = np.append(indices_remove, i).astype('int')

        self.indices_refit = np.delete(self.indices_all.copy(), indices_remove)

        self.locations_refit = np.take(
            np.array(self.location), self.indices_refit, axis=0)

    def check_continuity(self):
        self.refitting_iteration += 1
        self.say('\nthreshold for required components: {:.3f}'.format(self.min_p))

        self.determine_spectra_for_flagging()

        self.check_indices_refit()
        self.refitting()

    def refit_spectrum_phase_2(self, index, i):
        refit = False
        loc = self.locations_refit[i]
        spectrum = self.data[index]
        rms = self.errors[index][0]
        signal_ranges = self.signalRanges[index]
        noise_spike_ranges = self.noiseSpikeRanges[index]
        signal_mask = self.signal_mask[index]

        dictResults, dictResults_best = (None for _ in range(2))
        #  TODO: check if this is correct:
        indices_neighbors = []

        dictComps = self.check_continuity_centroids(index, loc)
        dct_refit = self.check_for_required_components(index, dictComps)

        if len(dct_refit['means_interval'].keys()) == 0:
            return [index, None, indices_neighbors, refit]

        dct_refit = self.select_neighbors_to_use_for_refit(dct_refit)

        #  TODO: first try to fit with indices_refit_all if present

        for key in dct_refit['indices_refit']:
            indices_neighbors = np.array(dct_refit['indices_refit'][key]).astype('int')
            interval = dct_refit['means_interval'][key]
            n_centroids = dct_refit['n_centroids'][key]

            dictResults, refit = self.try_refit_with_individual_neighbors(
                index, spectrum, rms, indices_neighbors, signal_ranges,
                noise_spike_ranges, signal_mask, interval=interval,
                n_centroids=n_centroids, dct_new_fit=dictResults)

            if dictResults is not None:
                dictResults_best = dictResults

        return [index, dictResults_best, indices_neighbors, refit]
