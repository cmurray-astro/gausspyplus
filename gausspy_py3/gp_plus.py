# @Author: riener
# @Date:   2018-12-19T17:30:53+01:00
# @Filename: gp_plus.py
# @Last modified by:   riener
# @Last modified time: 2019-03-05T12:23:46+01:00

import sys
import numpy as np

from lmfit import minimize as lmfit_minimize
from lmfit import Parameters


def say(message, verbose=False):
    """Diagnostic messages."""
    if verbose is True:
        print(message)


def split_params(params, ncomps):
    """Split params into amps, fwhms, offsets."""
    amps = params[0:ncomps]
    fwhms = params[ncomps:2*ncomps]
    offsets = params[2*ncomps:3*ncomps]
    return amps, fwhms, offsets


def number_of_components(params):
    """Compute number of Gaussian components."""
    return int(len(params) / 3)


def gaussian_function(peak, FWHM, mean):
    """Return a Gaussian function."""
    sigma = FWHM / 2.354820045  # (2 * sqrt( 2 * ln(2)))
    return lambda x: peak * np.exp(-(x - mean)**2 / 2. / sigma**2)


def func(x, *args):
    """Return multi-component Gaussian model F(x).

    Parameter vector kargs = [amp1, ..., ampN, width1, ..., widthN, mean1, ..., meanN],
    and therefore has len(args) = 3 x N_components.
    """
    ncomps = number_of_components(args)
    yout = x * 0.
    for i in range(ncomps):
        yout = yout + gaussian_function(
            args[i], args[i+ncomps], args[i+2*ncomps])(x)
    return yout


def vals_vec_from_lmfit(lmfit_params):
    """Return Python list of parameter values from LMFIT Parameters object."""
    if (sys.version_info >= (3, 0)):
        vals = [value.value for value in list(lmfit_params.values())]
    else:
        vals = [value.value for value in lmfit_params.values()]
    return vals


def errs_vec_from_lmfit(lmfit_params):
    """Return Python list of parameter uncertainties from LMFIT Parameters object."""
    if (sys.version_info >= (3, 0)):
        errs = [value.stderr for value in list(lmfit_params.values())]
    else:
        errs = [value.stderr for value in lmfit_params.values()]
    return errs


def paramvec_to_lmfit(paramvec, max_amp=None, max_fwhm=None,
                      params_min=None, params_max=None):
    """ Transform a Python iterable of parameters into a LMFIT Parameters object"""
    ncomps = number_of_components(paramvec)
    params = Parameters()

    if params_min is None:
        params_min = len(paramvec)*[0.]

    if params_max is None:
        params_max = len(paramvec)*[None]

        if max_amp is not None:
            params_max[0:ncomps] = ncomps*[max_amp]
        if max_fwhm is not None:
            params_max[ncomps:2*ncomps] = ncomps*[max_fwhm]

    for i in range(len(paramvec)):
        params.add('p{}'.format(str(i + 1)), value=paramvec[i],
                   min=params_min[i], max=params_max[i])
    return params


def gaussian(amp, fwhm, mean, x):
    """Return results of a Gaussian function."""
    gauss = amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)
    return gauss


def combined_gaussian(amps, fwhms, means, x):
    """Return results of multiple combined Gaussian functions."""
    if len(amps) > 0.:
        for i in range(len(amps)):
            gauss = gaussian(amps[i], fwhms[i], means[i], x)
            if i == 0:
                combined_gauss = gauss
            else:
                combined_gauss += gauss
    else:
        combined_gauss = np.zeros(len(x))
    return combined_gauss


def remove_components_from_list(lst, remove_indices):
    for idx, sublst in enumerate(lst):
        lst[idx] = [val for i, val in enumerate(sublst)
                    if i not in remove_indices]
    return lst


def determine_significance(amp, fwhm, rms):
    """Calculate the equivalent SNR of the fitted Gaussian.

    Gaussians with an equivalent SNR < 5 should be discarded.

    The area of the Gaussian is:
    area_gauss = amp * fwhm / ((1. / np.sqrt(2*np.pi)) * 2*np.sqrt(2*np.log(2)))

    This is then compared to the integrated rms, with 2*fwhm being a good
    approximation for the width of the emission line

    significance = area_gauss / (np.sqrt(2*fwhm) * rms)

    combining all constants yields a factor of 0.75269184778925247
    """
    return amp * np.sqrt(fwhm) * 0.75269184778925247 / rms


def goodness_of_fit(data, best_fit_final, errors, ncomps_fit, mask=None,
                    get_aicc=False):
    """Determine the goodness of fit (reduced chi2, AICc).

    Parameters
    ----------
    data : list
        Original data.
    best_fit_final : list
        Fit to the original data.
    errors : list
        Root-mean-square noise for each channel.
    ncomps_fit : int
        Number of Gaussian components used for the fit.
    mask : type
        Mask specifying which regions of the spectrum should be used.
    get_aicc : type
        If set to `True`, the AICc value will be returned in addition to the
        reduced chi2 value.

    Returns
    -------
    rchi2 : float
        Reduced chi2 value.
    aicc : float
        (optional). AICc value is returned if get_aicc is set to `True`.

    """
    #  create array if single rms value was supplied
    if type(errors) is not np.ndarray:
        errors = np.ones(len(data)) * errors

    #  use the whole spectrum if no mask was supplied
    # TODO: check if mask is set to None everywehere there is no mask
    if mask is None:
        mask = np.ones(len(data))
        mask = mask.astype('bool')
    elif len(mask) == 0:
        mask = np.ones(len(data))
        mask = mask.astype('bool')
    elif np.count_nonzero(mask) == 0:
        mask = np.ones(len(data))
        mask = mask.astype('bool')

    chi2 = np.sum((data[mask] - best_fit_final[mask])**2 / errors[mask]**2)
    k = 3*ncomps_fit  # degrees of freedom
    N = len(data[mask])
    rchi2 = chi2 / (N - k)

    #  corrected Akaike information criterion
    if get_aicc:
        aicc = chi2 + 2*k + (2*k**2 + 2*k) / (N - k - 1)
        return rchi2, aicc
    return rchi2


def check_if_intervals_contain_signal(spectrum, rms, ranges, snr=3.,
                                      significance=5., minChannelsSnr=2):
    """Check if selected intervals contain positive signal.

    Parameters
    ----------
    spectrum : list
        Description of parameter `spectrum`.
    rms : float
        Root-mean-square noise of the spectrum.
    ranges : list
        List of intervals [(low, upp), ...] that were identified as containing
        positive signal.
    snr : float
        If maximum intensity value of interval (low, upp) is smaller than
        snr * rms the interval is not retained
    significance : float
        Additional threshold check for the S/N that compares the sum of the
        intensities to the square root of the channels times the noise. This
        helps to remove narrow noise spikes or insignificant positive intensity
        peaks.

    Returns
    -------
    ranges_new : list
        New list of intervals [(low, upp), ...] that contain positive signal.

    """
    # TODO: incorporate minChannelsSnr, rethink snr - 0.5 ?
    ranges_new = []
    for low, upp in ranges:
        if np.max(spectrum[low:upp]) > snr*rms:
            if sum(spectrum[low:upp] > (snr - 0.5)*rms) >= minChannelsSnr:
                if np.sum(spectrum[low:upp]) / (np.sqrt(upp - low)*rms) > significance:
                    ranges_new.append([low, upp])
    return ranges_new


def determine_peaks(spectrum, peak='both', amp_threshold=None):
    import numpy as np

    #  get ranges of positive consecutive channels
    if (peak == 'both') or (peak == 'positive'):
        clipped_spectrum = spectrum.clip(max=0)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0
        iszero = np.concatenate(
            ([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    #  get ranges of negative consecutive channels
    if (peak == 'both') or (peak == 'negative'):
        clipped_spectrum = spectrum.clip(min=0)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0
        iszero = np.concatenate(
            ([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        if peak == 'both':
            # Runs start and end where absdiff is 1
            ranges = np.append(
                ranges, np.where(absdiff == 1)[0].reshape(-1, 2), axis=0)
        else:
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    #  retain only those ranges that contain an intensity value above or
    #  below the threshold
    if amp_threshold is not None:
        if peak == 'positive':
            mask = spectrum > abs(amp_threshold)
        elif peak == 'negative':
            mask = spectrum < -abs(amp_threshold)
        else:
            mask = np.abs(spectrum) > abs(amp_threshold)

        if np.count_nonzero(mask) == 0:
            return np.array([]), np.array([])

        peak_mask = np.split(mask, ranges[:, 1])
        mask_true = np.array([any(array) for array in peak_mask[:-1]])

        ranges = ranges[mask_true]
        if peak == 'positive':
            amp_vals = np.array([max(spectrum[low:upp]) for low, upp in ranges])
        elif peak == 'negative':
            amp_vals = np.array([min(spectrum[low:upp]) for low, upp in ranges])
        else:
            amp_vals = np.array(
                np.sign(spectrum[low])*max(np.abs(spectrum[low:upp]))
                for low, upp in ranges)
        #  TODO: check if sorting really necessary??
        sort_indices = np.argsort(amp_vals)[::-1]
        return amp_vals[sort_indices], ranges[sort_indices]
    else:
        sort_indices = np.argsort(ranges[:, 0])
        ranges = ranges[sort_indices]

        consecutive_channels = ranges[:, 1] - ranges[:, 0]
        return consecutive_channels, ranges


# def check_offset_difference(params_fit, params_errs, min_offset=0.):
#     import itertools
#     ncomps_fit = number_of_components(params_fit)
#     amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)
#     amps_errs, fwhms_errs, offsets_errs = split_params(params_errs, ncomps_fit)
#     remove_indices = []
#     items = range(ncomps_fit)
#
#     for idx1, idx2 in itertools.combinations(items, 2):
#         offset_err = max(min_offset, offsets_errs[idx1])
#         min1 = offsets_fit[idx1] - offset_err
#         max1 = offsets_fit[idx1] + offset_err
#
#         offset_err = max(min_offset, offsets_errs[idx2])
#         min2 = offsets_fit[idx2] - offset_err
#         max2 = offsets_fit[idx2] + offset_err
#
#         #  TODO: rework this? replace decision based on errors?
#         if (min1 < offsets_fit[idx2] < max1) or (
#                 min2 < offsets_fit[idx1] < max2):
#             if offsets_errs[idx2] < offsets_errs[idx1]:
#                 remove_indices.append(idx1)
#             else:
#                 remove_indices.append(idx2)
#
#     remove_indices = list(set(remove_indices))
#
#     amps_fit, fwhms_fit, offsets_fit = remove_components_from_list(
#         [amps_fit, fwhms_fit, offsets_fit], remove_indices)
#     amps_errs, fwhms_errs, offsets_errs = remove_components_from_list(
#         [amps_errs, fwhms_errs, offsets_errs], remove_indices)
#
#     params_fit = amps_fit + fwhms_fit + offsets_fit
#     params_errs = amps_errs + fwhms_errs + offsets_errs
#
#     return params_fit, params_errs, len(amps_fit)


def check_params_fit(data, params_fit, params_errs, vel, error, max_amp,
                     max_fwhm, snr=3., significance=5., snr_fit=3.,
                     min_fwhm=None, signal_ranges=None,
                     params_min=None, params_max=None):
    ncomps_fit = number_of_components(params_fit)

    # # check minimum offset criterion if there is more than one Gaussian
    # if ncomps_fit > 1:
    #     params_fit, params_errs, ncomps_fit = check_offset_difference(
    #         params_fit, params_errs, min_offset=min_offset)

    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)
    amps_errs, fwhms_errs, offsets_errs = split_params(params_errs, ncomps_fit)
    if params_min is not None:
        amps_min, fwhms_min, offsets_min = split_params(params_min, ncomps_fit)
    if params_max is not None:
        amps_max, fwhms_max, offsets_max = split_params(params_max, ncomps_fit)

    remove_indices = []
    for i, (amp, fwhm, offset) in enumerate(
            zip(amps_fit, fwhms_fit, offsets_fit)):
        if (offset < np.min(vel)) or (offset > np.max(vel)):
            remove_indices.append(i)
            continue

        if (amp < snr_fit*error) or (amp > max_amp):
            remove_indices.append(i)
            continue

        if determine_significance(amp, fwhm, error) < significance:
            remove_indices.append(i)
            continue

        #  check significance of fit component if it was fit outside the
        #  determined signal ranges
        if signal_ranges:
            if not any(low <= offset <= upp for low, upp in signal_ranges):
                low = max(0, int(offset - fwhm))
                upp = int(offset + fwhm) + 2
                #  TODO: rework the minChannelsSnr properly (so that it can be changed from the main module)
                if not check_if_intervals_contain_signal(
                        data, error, [(low, upp)], snr=snr,
                        significance=significance, minChannelsSnr=3):
                    remove_indices.append(i)
                    continue

    remove_indices = list(set(remove_indices))

    amps_fit, fwhms_fit, offsets_fit = remove_components_from_list(
        [amps_fit, fwhms_fit, offsets_fit], remove_indices)
    params_fit = amps_fit + fwhms_fit + offsets_fit

    amps_errs, fwhms_errs, offsets_errs = remove_components_from_list(
        [amps_errs, fwhms_errs, offsets_errs], remove_indices)
    params_errs = amps_errs + fwhms_errs + offsets_errs

    if params_min is not None:
        amps_min, fwhms_min, offsets_min = remove_components_from_list(
            [amps_min, fwhms_min, offsets_min], remove_indices)
        params_min = amps_min + fwhms_min + offsets_min

    if params_max is not None:
        amps_max, fwhms_max, offsets_max = remove_components_from_list(
            [amps_max, fwhms_max, offsets_max], remove_indices)
        params_max = amps_max + fwhms_max + offsets_max

    return params_fit, params_errs, len(amps_fit), params_min, params_max


def check_which_gaussian_contains_feature(idx_low, idx_upp, fwhms_fit,
                                          offsets_fit):
    lower = [int(offset - fwhm) for fwhm, offset in zip(fwhms_fit, offsets_fit)]
    lower = np.array([0 if x < 0 else x for x in lower])
    upper = np.array([int(offset + fwhm) + 2 for fwhm, offset in zip(fwhms_fit, offsets_fit)])

    indices = np.arange(len(fwhms_fit))
    conditions = np.logical_and(lower <= idx_low, upper >= idx_upp)

    if np.count_nonzero(conditions) == 0:
        return None
    elif np.count_nonzero(conditions) == 1:
        return int(indices[conditions])
    else:
        remaining_indices = indices[conditions]
        select = np.argmax(np.array(fwhms_fit)[remaining_indices])
        return int(remaining_indices[select])


def replace_gaussian_with_two_new_ones(data, vel, rms, snr, significance,
                                       params_fit, exclude_idx, offset):
    ncomps_fit = number_of_components(params_fit)
    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    # TODO: check if this is still necessary?
    if exclude_idx is None:
        return params_fit

    idx_low_residual = max(0, int(
        offsets_fit[exclude_idx] - fwhms_fit[exclude_idx]))
    idx_upp_residual = int(
        offsets_fit[exclude_idx] + fwhms_fit[exclude_idx]) + 2

    mask = np.arange(len(amps_fit)) == exclude_idx
    amps_fit = np.array(amps_fit)[~mask]
    fwhms_fit = np.array(fwhms_fit)[~mask]
    offsets_fit = np.array(offsets_fit)[~mask]

    residual = data - combined_gaussian(amps_fit, fwhms_fit, offsets_fit, vel)

    for low, upp in zip([idx_low_residual, int(offset)], [int(offset), idx_upp_residual]):
        amp_guess, fwhm_guess, offset_guess = get_initial_guesses(
            residual[low:upp], rms, snr, significance, peak='positive', maximum=True)

        if amp_guess.size == 0:
            continue

        amps_fit, fwhms_fit, offsets_fit = list(amps_fit), list(fwhms_fit), list(offsets_fit)

        amps_fit.append(amp_guess)
        fwhms_fit.append(fwhm_guess)
        offsets_fit.append(offset_guess + low)

    params_fit = amps_fit + fwhms_fit + offsets_fit

    return params_fit


def get_initial_guesses(residual, rms, snr, significance, peak='positive',
                        maximum=False, baseline_shift_snr=0):
    # amp_guesses, ranges = determine_peaks(
    #     residual, peak=peak, amp_threshold=snr*rms)
    amp_guesses, ranges = determine_peaks(
        residual - baseline_shift_snr*rms, peak=peak,
        amp_threshold=(snr - baseline_shift_snr)*rms)

    if amp_guesses.size == 0:
        return np.array([]), np.array([]), np.array([])

    amp_guesses = amp_guesses + baseline_shift_snr*rms

    sort = np.argsort(ranges[:, 0])
    amp_guesses = amp_guesses[sort]
    ranges = ranges[sort]

    keep_indices = np.array([])
    significance_vals = np.array([])
    for i, (lower, upper) in enumerate(ranges):
        significance_val = np.sum(
            np.abs(residual[lower:upper])) / (np.sqrt(upper - lower)*rms)
        significance_vals = np.append(significance_vals, significance_val)
        if significance_val > significance:
            keep_indices = np.append(keep_indices, i)

    keep_indices = keep_indices.astype('int')
    amp_guesses = amp_guesses[keep_indices]
    ranges = ranges[keep_indices]
    significance_vals = significance_vals[keep_indices]

    if amp_guesses.size == 0:
        return np.array([]), np.array([]), np.array([])

    amp_guesses_position_mask = np.in1d(residual, amp_guesses)
    offset_guesses = np.where(amp_guesses_position_mask == True)[0]

    #  see determine_significance
    fwhm_guesses = (significance_vals*rms
                    / (amp_guesses * 0.75269184778925247))**2

    if maximum:
        idx_max = np.argmax(amp_guesses)
        amp_guesses = amp_guesses[idx_max]
        fwhm_guesses = fwhm_guesses[idx_max]
        offset_guesses = offset_guesses[idx_max]

    return amp_guesses, fwhm_guesses, offset_guesses


def get_fully_blended_gaussians(params_fit, get_count=False, criterion=None):
    import itertools
    ncomps_fit = number_of_components(params_fit)
    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)
    stddevs_fit = list(np.array(fwhms_fit) / 2.354820045)
    indices_blended = np.array([])
    blended_pairs = []
    items = list(range(ncomps_fit))

    N_blended = 0
    # for idx1, idx2 in itertools.combinations(items, 2):
    #     min1 = offsets_fit[idx1] - abs(fwhms_fit[idx1]) / 2.
    #     max1 = offsets_fit[idx1] + abs(fwhms_fit[idx1]) / 2.
    #
    #     min2 = offsets_fit[idx2] - abs(fwhms_fit[idx2]) / 2.
    #     max2 = offsets_fit[idx2] + abs(fwhms_fit[idx2]) / 2.
    #
    #     if ((min1 > min2) & (max1 < max2)) or ((min2 > min1) & (max2 < max1)):
    #         indices_blended = np.append(indices_blended, np.array([idx1, idx2]))
    #         if [idx1, idx2] not in blended_pairs:
    #             blended_pairs.append([idx1, idx2])
    #             N_blended += 1

    for idx1, idx2 in itertools.combinations(items, 2):
        min1 = offsets_fit[idx1] - stddevs_fit[idx1]
        max1 = offsets_fit[idx1] + stddevs_fit[idx1]

        min2 = offsets_fit[idx2] - stddevs_fit[idx2]
        max2 = offsets_fit[idx2] + stddevs_fit[idx2]

        if (min1 < offsets_fit[idx2] < max1) or (
                min2 < offsets_fit[idx1] < max2):
            indices_blended = np.append(indices_blended, np.array([idx1, idx2]))
            # if [idx1, idx2] not in blended_pairs:
            blended_pairs.append([idx1, idx2])
            N_blended += 1

    if get_count:
        return N_blended

    indices_blended = np.unique(indices_blended).astype('int')

    return indices_blended


def remove_components(params_fit, remove_indices):
    ncomps_fit = number_of_components(params_fit)
    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    #  TODO: not sure if this is okay
    if isinstance(remove_indices, list):
        remove_indices = list(np.array(remove_indices))
    else:
        remove_indices = [remove_indices]

    amps_fit = list(np.delete(np.array(amps_fit), remove_indices))
    fwhms_fit = list(np.delete(np.array(fwhms_fit), remove_indices))
    offsets_fit = list(np.delete(np.array(offsets_fit), remove_indices))

    params_fit = amps_fit + fwhms_fit + offsets_fit

    return params_fit


def mask_channels(nChannels, ranges, padChannels=None, remove_intervals=None):
    import numpy as np

    mask = np.zeros(nChannels)

    for (lower, upper) in ranges:
        if padChannels is not None:
            lower -= padChannels
            lower = max(0, lower)
            upper += padChannels
            if upper > nChannels:
                upper = nChannels
        mask[lower:upper] = 1

    if remove_intervals is not None:
        for (low, upp) in remove_intervals:
            mask[low:upp] = 0

    return mask.astype('bool')


def get_best_fit(vel, data, errors, params_fit, dct, first=False, plot=False,
                 best_fit_list=None, signal_ranges=None, signal_mask=None,
                 force_accept=False, params_min=None, params_max=None):
    # Objective functions for final fit
    def objective_leastsq(paramslm):
        params = vals_vec_from_lmfit(paramslm)
        resids = (func(vel, *params).ravel() - data.ravel()) / errors
        return resids

    if not first:
        best_fit_list[7] = False

    ncomps_fit = number_of_components(params_fit)

    #  get new best fit
    lmfit_params = paramvec_to_lmfit(
        params_fit, max_amp=dct['max_amp'], max_fwhm=None,
        params_min=params_min, params_max=params_max)
    result = lmfit_minimize(
        objective_leastsq, lmfit_params, method='leastsq')
    params_fit = vals_vec_from_lmfit(result.params)
    params_errs = errs_vec_from_lmfit(result.params)
    ncomps_fit = number_of_components(params_fit)

    #  check if fit components satisfy mandatory criteria
    if ncomps_fit > 0:
        params_fit, params_errs, ncomps_fit, params_min, params_max = check_params_fit(
            data, params_fit, params_errs, vel, errors[0], dct['max_amp'],
            dct['max_fwhm'], min_fwhm=dct['min_fwhm'], snr=dct['snr'],
            significance=dct['significance'], snr_fit=dct['snr_fit'],
            signal_ranges=signal_ranges)

        best_fit = func(vel, *params_fit).ravel()
    else:
        best_fit = data * 0

    rchi2, aicc = goodness_of_fit(
        data, best_fit, errors, ncomps_fit, mask=signal_mask, get_aicc=True)

    residual = data - best_fit

    #  return the list of best fit results if there was no old list of best fit results for comparison
    if first:
        new_fit = True
        return [params_fit, params_errs, ncomps_fit, best_fit, residual, rchi2,
                aicc, new_fit, params_min, params_max]

    #  return new best_fit_list if the AICc value is smaller
    aicc_old = best_fit_list[6]
    if ((aicc < aicc_old) and not np.isclose(aicc, aicc_old, atol=1e-1)) or force_accept:
        new_fit = True
        return [params_fit, params_errs, ncomps_fit, best_fit, residual, rchi2,
                aicc, new_fit, params_min, params_max]

    #  return old best_fit_list if the AICc value is higher
    best_fit_list[7] = False
    return best_fit_list


def check_for_negative_residual(vel, data, errors, best_fit_list, dct,
                                plot=False, signal_ranges=None,
                                signal_mask=None, force_accept=False,
                                get_count=False, get_idx=False):
    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]

    #  in case rms value is given instead of errors array
    if not isinstance(errors, np.ndarray):
        errors = np.ones(len(data)) * errors

    if ncomps_fit == 0:
        if get_count:
            return 0
        return best_fit_list

    residual = best_fit_list[4]

    #  in case rms value is given instead of errors array
    if not isinstance(errors, np.ndarray):
        errors = np.ones(len(data)) * errors

    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    amp_guesses, fwhm_guesses, offset_guesses = get_initial_guesses(
        residual, errors[0], dct['snr_negative'], dct['significance'],
        peak='negative')

    #  check if negative residual feature was already present in the data
    remove_indices = []
    for i, offset in enumerate(offset_guesses):
        if residual[offset] > (data[offset] - dct['snr']*errors[0]):
            remove_indices.append(i)

    if len(remove_indices) > 0:
        amp_guesses, fwhm_guesses, offset_guesses = remove_components_from_list(
            [amp_guesses, fwhm_guesses, offset_guesses], remove_indices)

    if get_count:
        return (len(amp_guesses))

    if len(amp_guesses) == 0:
        return best_fit_list

    for amp, fwhm, offset in zip(amp_guesses, fwhm_guesses, offset_guesses):
        idx_low = max(0, int(offset - fwhm))
        idx_upp = int(offset + fwhm) + 2
        exclude_idx = check_which_gaussian_contains_feature(
            idx_low, idx_upp, fwhms_fit, offsets_fit)
        if get_idx:
            return exclude_idx
        if exclude_idx is None:
            continue

        params_fit = replace_gaussian_with_two_new_ones(
            data, vel, errors[0], dct['snr'], dct['significance'],
            params_fit, exclude_idx, offset)

        best_fit_list = get_best_fit(
            vel, data, errors, params_fit, dct, first=False, plot=False,
            best_fit_list=best_fit_list, signal_ranges=signal_ranges,
            signal_mask=signal_mask, force_accept=force_accept)

        params_fit = best_fit_list[0]
        ncomps_fit = best_fit_list[2]
        amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)
    return best_fit_list


def try_fit_with_new_components(vel, data, errors, best_fit_list, dct,
                                exclude_idx, plot=False, signal_ranges=None,
                                signal_mask=None, force_accept=False,
                                baseline_shift_snr=0):
    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]
    aicc_old = best_fit_list[6]
    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    #  exclude component from parameter list of components
    idx_low_residual = max(
        0, int(offsets_fit[exclude_idx] - fwhms_fit[exclude_idx]/2))
    idx_upp_residual = int(
        offsets_fit[exclude_idx] + fwhms_fit[exclude_idx]/2) + 2

    params_fit_new = remove_components(params_fit, exclude_idx)

    #  produce new best fit with excluded components
    best_fit_list_new = get_best_fit(
        vel, data, errors, params_fit_new, dct, first=True, plot=False,
        best_fit_list=None, signal_ranges=signal_ranges,
        signal_mask=signal_mask, force_accept=force_accept)

    #  return new best fit with excluded component if its AICc value is lower
    aicc = best_fit_list_new[6]
    if ((aicc < aicc_old) and not np.isclose(aicc, aicc_old, atol=1e-1)):
        return best_fit_list_new

    #  search for new positive residual peaks
    params_fit = best_fit_list_new[0]
    ncomps_fit = best_fit_list_new[2]

    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    residual = data - combined_gaussian(amps_fit, fwhms_fit, offsets_fit, vel)

    # amp_guesses, fwhm_guesses, offset_guesses = get_initial_guesses(
    #     residual[idx_low_residual:idx_upp_residual], errors[0],
    #     dct['snr'], dct['significance'], peak='positive',
    #     baseline_shift_snr=baseline_shift_snr)
    # offset_guesses = offset_guesses + idx_low_residual
    amp_guesses, fwhm_guesses, offset_guesses = get_initial_guesses(
        residual, errors[0], dct['snr'], dct['significance'], peak='positive',
        baseline_shift_snr=baseline_shift_snr)

    #  return original best fit list if there are no guesses for new components to fit in the residual
    if amp_guesses.size == 0:
        return best_fit_list

    #  get new best fit with additional components guessed from the residual
    amps_fit = list(amps_fit) + list(amp_guesses)
    fwhms_fit = list(fwhms_fit) + list(fwhm_guesses)
    offsets_fit = list(offsets_fit) + list(offset_guesses)

    params_fit_new = amps_fit + fwhms_fit + offsets_fit

    best_fit_list_new = get_best_fit(
        vel, data, errors, params_fit_new, dct, first=False, plot=False,
        best_fit_list=best_fit_list_new, signal_ranges=signal_ranges,
        signal_mask=signal_mask, force_accept=force_accept)

    #  return new best fit if its AICc value is lower
    aicc = best_fit_list_new[6]
    if ((aicc < aicc_old) and not np.isclose(aicc, aicc_old, atol=1e-1)):
        return best_fit_list_new

    return best_fit_list


def check_for_broad_feature(vel, data, errors, best_fit_list, dct,
                            plot=False, signal_ranges=None,
                            signal_mask=None, force_accept=False,
                            get_count=False):
    best_fit_list[7] = False

    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]
    if ncomps_fit < 2 and dct['fwhm_factor'] > 0:
        return best_fit_list

    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    if len(fwhms_fit) > 1:
        fwhms_sorted = sorted(fwhms_fit)
        if (fwhms_sorted[-1] < dct['fwhm_factor'] * fwhms_sorted[-2]):
            return best_fit_list
    else:
        exclude_idx = 0

    exclude_idx = np.argmax(np.array(fwhms_fit))

    params_fit = replace_gaussian_with_two_new_ones(
        data, vel, errors[0], dct['snr'], dct['significance'],
        params_fit, exclude_idx, offsets_fit[exclude_idx])

    #  TODO: check if this makes sense
    if len(params_fit) == 0:
        return best_fit_list

    best_fit_list = get_best_fit(
        vel, data, errors, params_fit, dct, first=False, plot=False,
        best_fit_list=best_fit_list, signal_ranges=signal_ranges,
        signal_mask=signal_mask, force_accept=force_accept)

    if best_fit_list[7]:
        return best_fit_list

    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]
    if ncomps_fit == 0:
        return best_fit_list

    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    exclude_idx = np.argmax(np.array(fwhms_fit))

    # for baseline_shift_snr in range(int(dct['snr'])):
    #     best_fit_list = try_fit_with_new_components(
    #         vel, data, errors, best_fit_list, dct, exclude_idx, plot=plot,
    #         signal_ranges=signal_ranges, signal_mask=signal_mask,
    #         force_accept=force_accept, baseline_shift_snr=baseline_shift_snr)
    #     if best_fit_list[7]:
    #         break
    best_fit_list = try_fit_with_new_components(
        vel, data, errors, best_fit_list, dct, exclude_idx, plot=plot,
        signal_ranges=signal_ranges, signal_mask=signal_mask,
        force_accept=force_accept)

    return best_fit_list


def check_for_blended_feature(vel, data, errors, best_fit_list, dct,
                              plot=False, signal_ranges=None,
                              signal_mask=None, force_accept=False):
    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]
    if ncomps_fit == 0:
        return best_fit_list

    exclude_indices = get_fully_blended_gaussians(params_fit)

    #  skip if there are no blended features
    if exclude_indices.size == 0:
        return best_fit_list

    for exclude_idx in exclude_indices:
        best_fit_list = try_fit_with_new_components(
            vel, data, errors, best_fit_list, dct, exclude_idx, plot=plot,
            signal_ranges=signal_ranges, signal_mask=signal_mask,
            force_accept=force_accept)
        if best_fit_list[7]:
            break
        # for baseline_shift_snr in range(int(dct['snr'])):
        #     best_fit_list = try_fit_with_new_components(
        #         vel, data, errors, best_fit_list, dct, exclude_idx, plot=plot,
        #         signal_ranges=signal_ranges, signal_mask=signal_mask,
        #         force_accept=force_accept, baseline_shift_snr=baseline_shift_snr)
        #     if best_fit_list[7]:
        #         return best_fit_list

    return best_fit_list


def quality_check(vel, data, errors, params_fit, ncomps_fit, dct,
                  plot=False, signal_ranges=None, signal_mask=None,
                  params_min=None, params_max=None):
    #  TODO: check if this should be ncomps_gf
    if ncomps_fit == 0:
        new_fit = False
        best_fit_final = data*0
        residual = data
        params_fit, params_errs = [], []

        rchi2, aicc = goodness_of_fit(
            data, best_fit_final, errors, ncomps_fit, mask=signal_mask, get_aicc=True)

        best_fit_list = [params_fit, params_errs, ncomps_fit, best_fit_final, residual, rchi2, aicc, new_fit, params_min, params_max]
        # N_flags = 0
        #
        # return best_fit_list, N_flags
        return best_fit_list

    best_fit_list = get_best_fit(
        vel, data, errors, params_fit, dct, first=True, plot=plot,
        best_fit_list=None, signal_ranges=signal_ranges,
        signal_mask=signal_mask)
    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]

    return best_fit_list

    # if ncomps_fit > 0:
    #     N_negative_residuals = check_for_negative_residual(
    #         vel, data, errors, best_fit_list, dct, plot=plot,
    #         signal_ranges=signal_ranges, signal_mask=signal_mask,
    #         get_count=True)
    #     N_broad = check_for_broad_feature(
    #         vel, data, errors, best_fit_list, dct, plot=plot,
    #         signal_ranges=signal_ranges, signal_mask=signal_mask,
    #         get_count=True)
    #     N_blended = get_fully_blended_gaussians(params_fit, get_count=True)
    #
    #     N_flags = N_negative_residuals + N_broad + N_blended
    # else:
    #     N_flags = 0
    #
    # return best_fit_list, N_flags


def check_for_peaks_in_residual(vel, data, errors, best_fit_list, dct,
                                fitted_residual_peaks, plot=False, signal_ranges=None, signal_mask=None,
                                force_accept=False,
                                params_min=None, params_max=None):
    #  TODO: remove params_min and params_max keywords
    params_fit = best_fit_list[0]
    ncomps_fit = best_fit_list[2]
    residual = best_fit_list[4]
    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)

    amp_guesses, fwhm_guesses, offset_guesses = get_initial_guesses(
        residual, errors[0], dct['snr'], dct['significance'],
        peak='positive')

    if amp_guesses.size == 0:
        best_fit_list[7] = False
        return best_fit_list, fitted_residual_peaks
    if list(offset_guesses) in fitted_residual_peaks:
        best_fit_list[7] = False
        return best_fit_list, fitted_residual_peaks

    fitted_residual_peaks.append(list(offset_guesses))

    amps_fit = list(amps_fit) + list(amp_guesses)
    fwhms_fit = list(fwhms_fit) + list(fwhm_guesses)
    offsets_fit = list(offsets_fit) + list(offset_guesses)

    params_fit = amps_fit + fwhms_fit + offsets_fit

    best_fit_list = get_best_fit(
        vel, data, errors, params_fit, dct, first=False, plot=False,
        best_fit_list=best_fit_list, signal_ranges=signal_ranges,
        signal_mask=signal_mask, force_accept=force_accept,
        params_min=params_min, params_max=params_max)

    return best_fit_list, fitted_residual_peaks


def log_new_fit(new_fit, log_gplus, mode='residual'):
    if not new_fit:
        return log_gplus

    modes = {'residual': 1, 'negative_residual': 2, 'broad': 3, 'blended': 4}
    log_gplus.append(modes[mode])
    return log_gplus


def try_to_improve_fitting(vel, data, errors, params_fit, ncomps_fit, dct,
                           plot=False, signal_ranges=None,
                           noise_spike_ranges=None):
    if signal_ranges:
        signal_mask = mask_channels(len(data), signal_ranges,
                                    remove_intervals=noise_spike_ranges)
    else:
        signal_mask = None
    # Check how good the final fit from the previous stage was
    # --------------------------------------------------------
    best_fit_list = quality_check(
        vel, data, errors, params_fit, ncomps_fit, dct, plot=plot,
        signal_ranges=signal_ranges, signal_mask=signal_mask)
    # best_fit_list, N_flags = quality_check(
    #     vel, data, errors, params_fit, ncomps_fit, dct, plot=plot,
    #     signal_ranges=signal_ranges, signal_mask=signal_mask)

    params_fit, params_errs, ncomps_fit, best_fit_final, residual,\
        rchi2, aicc, new_fit, params_min, params_max = best_fit_list

    # Try to improve fit by searching for peaks in the residual
    # ---------------------------------------------------------
    first_run = True
    fitted_residual_peaks = []
    log_gplus = []

    # while rchi2 > dct['rchi2_limit']:
    # while (rchi2 > dct['rchi2_limit']) or (N_flags > 0) or (ncomps_fit == 0):
    while (rchi2 > dct['rchi2_limit']) or first_run:
        new_fit = True
        new_peaks = False

        count_old = len(fitted_residual_peaks)
        while new_fit:
            best_fit_list[7] = False
            best_fit_list, fitted_residual_peaks = check_for_peaks_in_residual(
                vel, data, errors, best_fit_list, dct, fitted_residual_peaks, plot=plot, signal_ranges=signal_ranges,
                signal_mask=signal_mask)
            new_fit = best_fit_list[7]
            log_gplus = log_new_fit(new_fit, log_gplus, mode='residual')
        count_new = len(fitted_residual_peaks)

        if count_old != count_new:
            new_peaks = True

        #  TODO: remove the if ncomps == 0: break conditions in check_for...
        #  stop refitting loop if no new peaks were fit from the residual
        if (not first_run and not new_peaks) or (best_fit_list[2] == 0):
            break

        #  try to refit negative residual feature
        if dct['negative_residual']:
            best_fit_list = check_for_negative_residual(
                vel, data, errors, best_fit_list, dct, plot=plot,
                signal_ranges=signal_ranges, signal_mask=signal_mask)
            new_fit = best_fit_list[7]
            log_gplus = log_new_fit(new_fit, log_gplus, mode='negative_residual')

        #  try to refit broad Gaussian components
        if dct['broad']:
            new_fit = True
            while new_fit:
                best_fit_list[7] = False
                best_fit_list = check_for_broad_feature(
                    vel, data, errors, best_fit_list, dct, plot=plot, signal_ranges=signal_ranges, signal_mask=signal_mask)
                new_fit = best_fit_list[7]
                log_gplus = log_new_fit(new_fit, log_gplus, mode='broad')

        #  try to refit blended Gaussian components
        if dct['blended']:
            new_fit = True
            while new_fit:
                best_fit_list[7] = False
                best_fit_list = check_for_blended_feature(
                    vel, data, errors, best_fit_list, dct, plot=plot,
                    signal_ranges=signal_ranges, signal_mask=signal_mask)
                new_fit = best_fit_list[7]
                log_gplus = log_new_fit(new_fit, log_gplus, mode='blended')

        first_run = False

    N_negative_residuals = check_for_negative_residual(
        vel, data, errors, best_fit_list, dct, plot=plot,
        signal_ranges=signal_ranges, signal_mask=signal_mask,
        get_count=True)

    params_fit = best_fit_list[0]
    N_blended = get_fully_blended_gaussians(params_fit, get_count=True)

    return best_fit_list, N_negative_residuals, N_blended, log_gplus
