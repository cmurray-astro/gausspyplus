# @Author: riener
# @Date:   2019-02-18T16:27:12+01:00
# @Filename: spectral_cube_functions.py
# @Last modified by:   riener
# @Last modified time: 2019-03-13T13:08:54+01:00

import getpass
import itertools
import os
import socket
import warnings

import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.stats import median_absolute_deviation
from astropy.wcs import WCS
from datetime import datetime
from tqdm import tqdm

from gausspyplus.shared_functions import get_rms_noise, max_consecutive_channels


def check_if_value_is_none(condition, value):
    """Raise error message if no value is supplied for a selected condition.

    The error message is raised if the condition is 'True' and the value is 'None'.

    Parameters
    ----------
    condition : bool
        Selected condition.
    value : type
        Value for the condition.

    """
    if condition and (value is None):
        errorMessage = str("need to specify {} for {}=True".format(value, condition))
        raise Exception(errorMessage)


def check_if_all_values_are_none(value1, value2):
    """Raise error message if both values are 'None'.

    Parameters
    ----------
    value1 : type
        Description of parameter `value1`.
    value2 : type
        Description of parameter `value2`.

    """
    if (value1 is None) and (value2 is None):
        errorMessage = str("need to specify either {} or {}".format(value1, value2))
        raise Exception(errorMessage)


def correct_header(header, check_keywords={'BUNIT': 'K', 'CUNIT3': 'm/s'},
                   keep_only_wcs_keywords=False):
    for keyword, value in check_keywords.items():
        if keyword not in header.keys():
            warnings.warn("{a} keyword not found in header. Assuming {a}={b}".format(a=keyword, b=value))
            header[keyword] = value
    if header['CTYPE3'] == 'VELOCITY':
        warnings.warn("Changed header keyword CTYPE3 from VELOCITY to VELO-LSR")
        header['CTYPE3'] = 'VELO-LSR'
    if keep_only_wcs_keywords:
        wcs = WCS(header)
        dct_naxis = {}
        for keyword in header.keys():
            if keyword.startswith('NAXIS'):
                dct_naxis[keyword] = header[keyword]
        header = wcs.to_header()
        for keyword, value in dct_naxis.items():
            header[keyword] = value
    return header


def update_header(header, comments=[], remove_keywords=[], update_keywords={},
                  remove_old_comments=False, write_meta=True):
    if remove_old_comments:
        while ['COMMENT'] in header.keys():
            header.remove('COMMENT')

    for keyword in remove_keywords:
        if keyword in header.keys():
            header.remove(keyword)

    for keyword, value in update_keywords.items():
        header[keyword] = value[0][1]

    if write_meta:
        header['AUTHOR'] = getpass.getuser()
        header['ORIGIN'] = socket.gethostname()
        header['DATE'] = (datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'), '(GMT)')

    for comment in comments:
        header['COMMENT'] = comment

    return header


def remove_additional_axes(data, header, verbose=True, max_dim=3,
                           keep_only_wcs_keywords=False):
    """Remove additional axes (Stokes, etc.) from spectral cube."""
    #  old name was remove_stokes

    wcs = WCS(header)

    if header['NAXIS'] <= max_dim and wcs.wcs.naxis <= max_dim:
        return data, header

    warnings.warn('remove additional axes (Stokes, etc.) from cube and/or header')

    while data.ndim > max_dim:
        data = np.squeeze(data, axis=(0,))

    wcs_header_old = wcs.to_header()
    while wcs.wcs.naxis > max_dim:
        axes = range(wcs.wcs.naxis)
        wcs = wcs.dropaxis(axes[-1])
    wcs_header_new = wcs.to_header()

    if keep_only_wcs_keywords:
        hdu = fits.PrimaryHDU(data=data, header=wcs_header_new)
        return hdu.data, hdu.header

    wcs_header_diff = fits.HeaderDiff(wcs_header_old, wcs_header_new)
    header_diff = fits.HeaderDiff(header, wcs_header_new)
    update_header(header, remove_keywords=wcs_header_diff.diff_keywords[0],
                  update_keywords=header_diff.diff_keyword_values,
                  write_meta=False)

    return data, header


def swap_axes(data, header, new_order):
    dims = data.ndim
    data = np.transpose(data, new_order)
    hdu = fits.PrimaryHDU(data=data)
    header_new = hdu.header

    if 'CD1_1' in header.keys():
        raise Exception('Cannot swap_axes for CDX_X keywords. Convert them to CDELTX.')

    for keyword in header.keys():
        for axis in range(dims):
            if keyword.endswith(str(axis + 1)):
                keyword_new = keyword.replace(str(axis + 1), str(new_order[axis] + 1))
                header_new[keyword_new] = header[keyword]

    header_diff = fits.HeaderDiff(header, header_new)
    for keyword, value in header_diff.diff_keyword_values.items():
        header[keyword] = value[0][1]
    return data, header


def get_slices(size, n):
    """Calculate slices in individual direction."""
    limits, slices = ([] for _ in range(2))

    for i in range(n):
        limits.append(i * size)
    limits.append(None)

    for a, b in zip(limits[:-1], limits[1:]):
        slices.append(slice(a, b))

    return slices


def get_list_slice_params(pathToFile=None, hdu=None, ncols=1, nrows=1,
                          velocity_slice=slice(None, None)):
    """Calculate slices needed to slice PPV cube into individual subcubes."""
    import itertools

    check_if_all_values_are_none(hdu, pathToFile)

    if pathToFile is not None:
        hdu = fits.open(pathToFile)[0]

    x = hdu.header['NAXIS1']
    y = hdu.header['NAXIS2']

    x_size = int(x / ncols)
    y_size = int(y / nrows)

    x_slices = get_slices(x_size, ncols)
    y_slices = get_slices(y_size, nrows)

    slices = []

    for y_slice, x_slice in itertools.product(y_slices, x_slices):
        slices.append([velocity_slice, y_slice, x_slice])
    return slices


def get_locations(data=None, header=None):
    if data is not None:
        yValues = np.arange(data.shape[1])
        xValues = np.arange(data.shape[2])
    else:
        yValues = np.arange(header['NAXIS2'])
        xValues = np.arange(header['NAXIS1'])
    return list(itertools.product(yValues, xValues))


def save_fits(data, header, pathToFile, verbose=True):
    if not os.path.exists(os.path.dirname(pathToFile)):
        os.makedirs(os.path.dirname(pathToFile))
    fits.writeto(pathToFile, data, header=header, overwrite=True)
    if verbose:
        print("\n'{}' saved in {}".format(
            os.path.basename(pathToFile), os.path.dirname(pathToFile)))


def open_fits_file(pathToFile, get_hdu=False, get_data=True, get_header=True,
                   remove_Stokes=True, check_wcs=True):
    """"""
    #  TODO: rework the check_wcs condition
    from astropy.io import fits

    data = fits.getdata(pathToFile)
    header = fits.getheader(pathToFile)

    if remove_Stokes:
        data, header = remove_additional_axes(data, header)

    if check_wcs:
        header = correct_header(header)

    if get_hdu:
        return fits.PrimaryHDU(data, header)
    elif get_data and (not get_header):
        return data
    elif (not get_data) and get_header:
        return header
    else:
        return data, header


def spatial_smoothing(data, header, save=False, pathOutputFile=None,
                      suffix=None, current_resolution=None,
                      target_resolution=None, verbose=True):
    """Smooth the cube spatially.

    NB: Note sure if this entirely correct but the smoothing seems to give
    reasonable results.

    Parameters
    ----------
    saveCube : boolean
        If True it saves the smoothed cube as a FITS file to ~/gpy_prepared/FITS.
    suffix : type
        Suffix that will be added to the end of the filename if 'saveCube=True'.
    smoothed_resolution : type
        Resolution the slices of the cube should be smoothed to. Has to be
        an astropy unit object (e.g., 5*u.arcmin, 0.6*u.deg, etc.)

    Returns
    -------
    type
        Description of returned object.

    """
    # from radio_beam import Beam
    # from spectral_cube import SpectralCube
    from astropy import units as u
    from astropy.convolution import Gaussian2DKernel, convolve

    check_if_value_is_none(save, pathOutputFile)
    check_if_all_values_are_none(current_resolution, target_resolution)

    # # in case no beam info is present in the header use the pixel size as beam proxy
    # if 'BMAJ' not in header:
    #     header['BMAJ'] = abs(header['CDELT1'])
    #     header['BMIN'] = abs(header['CDELT1'])
    #     header['BPA'] = 0.

    wcs = WCS(header)
    # cube = SpectralCube(data=data, wcs=wcs, header=header)

    fwhm_factor = np.sqrt(8*np.log(2))
    pixel_scale = abs(wcs.wcs.cdelt[0]) * wcs.wcs.cunit[0]

    if target_resolution is None:
        target_resolution = 2*current_resolution
        warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(target_resolution))

    # beam = Beam(target_resolution)
    # new_cube = cube.convolve_to(beam)
    #
    # data = new_cube.hdu.data
    # header = new_cube.hdu.header

    current_resolution = current_resolution.to(u.deg)
    target_resolution = target_resolution.to(u.deg)
    pixel_scale = pixel_scale.to(u.deg)

    kernel_fwhm = np.sqrt(target_resolution.value**2 -
                          current_resolution.value**2)
    kernel_std = (kernel_fwhm / fwhm_factor) / pixel_scale.value
    # TODO: leave the kernel size optional
    kernel = Gaussian2DKernel(kernel_std, x_size=9, y_size=9)

    nSpectra = data.shape[0]
    for i in tqdm(range(nSpectra)):
        channel = data[i, :, :]
        channel_smoothed = convolve(channel, kernel)
        data[i, :, :] = channel_smoothed

    comments = ['spatially smoothed to a resolution of {}'.format(
        target_resolution)]
    header = update_header(header, comments=comments)

    if save:
        save_fits(data, header, pathOutputFile, verbose=verbose)

    return data, header


def spectral_smoothing(data, header, save=False, pathOutputFile=None,
                       suffix=None, current_resolution=None,
                       target_resolution=None, verbose=True):
    """"""
    from astropy import units as u
    # from spectral_cube import SpectralCube
    from astropy.convolution import convolve, Gaussian1DKernel

    check_if_value_is_none(save, pathOutputFile)

    wcs = WCS(header)
    # cube = SpectralCube(data=data, wcs=wcs, header=header)

    fwhm_factor = np.sqrt(8*np.log(2))
    pixel_scale = wcs.wcs.cdelt[2] * wcs.wcs.cunit[2]

    if target_resolution is None:
        target_resolution = 2*current_resolution
        warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(target_resolution))

    current_resolution = current_resolution.to(u.m/u.s)
    target_resolution = target_resolution.to(u.m/u.s)
    pixel_scale = pixel_scale.to(u.m/u.s)

    gaussian_width = (
        (target_resolution.value**2 - current_resolution.value**2)**0.5 /
        pixel_scale.value / fwhm_factor)
    kernel = Gaussian1DKernel(gaussian_width)

    #  the next line doesn't work because of a bug in spectral_cube
    #  the new_cube.mask attribute is set to None if cube is defined with data=X and header=X instead of reading it in from the cube; this leads to an error with spectral_smooth
    # new_cube = cube.spectral_smooth(kernel)
    # data = new_cube.hdu.data
    # header = new_cube.hdu.header

    yMax = data.shape[1]
    xMax = data.shape[2]
    locations = list(
            itertools.product(range(yMax), range(xMax)))
    for ypos, xpos in tqdm(locations):
        spectrum = data[:, ypos, xpos]
        spectrum_smoothed = convolve(spectrum, kernel)
        data[:, ypos, xpos] = spectrum_smoothed

    comments = ['spectrally smoothed cube to a resolution of {}'.format(target_resolution)]
    header = update_header(header, comments=comments)

    if save:
        save_fits(data, header, pathOutputFile, verbose=verbose)

    return data, header


def determine_noise(spectrum, maxConsecutiveChannels=14, pad_channels=5,
                    idx=None, averageRms=None, random_seed=111):
    np.random.seed(random_seed)
    if not np.isnan(spectrum).all():
        if np.isnan(spectrum).any():
            # TODO: Case where spectrum contains nans and only positive values
            nans = np.isnan(spectrum)
            error = get_rms_noise(spectrum[~nans], maxConsecutiveChannels=maxConsecutiveChannels, pad_channels=pad_channels, idx=idx, averageRms=averageRms)
            spectrum[nans] = np.random.randn(len(spectrum[nans])) * error

        elif (spectrum >= 0).all():
            warnings.warn('Masking spectra that contain only values >= 0')
            error = np.NAN
        else:
            error = get_rms_noise(spectrum, maxConsecutiveChannels=maxConsecutiveChannels, pad_channels=pad_channels, idx=idx, averageRms=averageRms)
    else:
        error = np.NAN
    return error


def calculate_average_rms_noise(data, numberRmsSpectra, random_seed=111,
                                maxConsecutiveChannels=14, pad_channels=5):
    import random

    random.seed(random_seed)
    yValues = np.arange(data.shape[1])
    xValues = np.arange(data.shape[2])
    locations = list(itertools.product(yValues, xValues))
    if len(locations) > numberRmsSpectra:
        locations = random.sample(locations, len(locations))
    rmsList = []
    counter = 0
    pbar = tqdm(total=numberRmsSpectra)
    for y, x in locations:
        spectrum = data[:, y, x]
        error = determine_noise(
            spectrum, maxConsecutiveChannels=maxConsecutiveChannels,
            pad_channels=pad_channels)

        if not np.isnan(error):
            rmsList.append(error)
            counter += 1
            pbar.update(1)

        if counter >= numberRmsSpectra:
            break

    pbar.close()
    return np.nanmean(rmsList), np.nanstd(rmsList)
    # return np.nanmedian(rmsList), median_absolute_deviation(rmsList, ignore_nan=True)


def get_path_to_output_file(pathToInputFile, suffix='_',
                            filename='foo.fits'):
    if pathToInputFile is None:
        pathToOutputFile = os.path.join(os.getcwd(), filename)
    else:
        dirname = os.path.dirname(pathToInputFile)
        filename = os.path.basename(pathToInputFile)
        fileBase, fileExtension = os.path.splitext(pathToInputFile)
        filename = '{}{}{}'.format(fileBase, suffix, fileExtension)
        pathToOutputFile = os.path.join(dirname, filename)
    return pathToOutputFile


def add_noise(average_rms, pathToInputFile=None, hdu=None, save=False,
              overwrite=True, pathToOutputFile=None, get_hdu=False,
              get_data=True, get_header=True, random_seed=111):
    print('\nadding noise (rms = {}) to data...'.format(average_rms))

    check_if_all_values_are_none(hdu, pathToInputFile)

    np.random.seed(random_seed)

    if pathToInputFile is not None:
        hdu = fits.open(pathToInputFile)[0]

    data = hdu.data
    header = hdu.header

    channels = data.shape[0]
    yValues = np.arange(data.shape[1])
    xValues = np.arange(data.shape[2])
    locations = list(itertools.product(yValues, xValues))
    for y, x in locations:
        data[:, y, x] += np.random.randn(channels) * average_rms

    header['COMMENT'] = "Added rms noise of {} ({})".format(
            average_rms, datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))

    if save:
        if pathToOutputFile is None:
            pathToOutputFile = get_path_to_output_file(pathToInputFile, suffix='_w_noise', filename='cube_w_noise.fits')

        save_fits(data, header, pathToOutputFile, verbose=True)

    if get_hdu:
        return fits.PrimaryHDU(data, header)
    elif get_data and (not get_header):
        return data
    elif (not get_data) and get_header:
        return header
    else:
        return data, header


def transform_coordinates_to_pixel(coordinates, header):
    if not isinstance(coordinates, list):
        coordinates = list(coordinates)
    wcs = WCS(header)
    units = wcs.wcs.cunit

    for i, (coordinate, unit) in enumerate(zip(coordinates, units)):
        if isinstance(coordinate, u.Quantity):
            coordinates[i] = coordinate.to(unit)
        else:
            raise Exception('coordinates must be specified with astropy.units')

    lon, lat, vel = coordinates
    x, y, z = wcs.wcs_world2pix(lon, lat, vel, 0)
    return [max(int(x), 0), max(int(y), 0), max(0, int(z))]


def transform_header_from_crota_to_pc(header):
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    crota = np.radians(header['CROTA1'])
    header['PC1_1'] = np.cos(crota)
    header['PC1_2'] = -(cdelt2 / cdelt1) * np.sin(crota)
    header['PC2_1'] = (cdelt1 / cdelt1) * np.sin(crota)
    header['PC2_2'] = np.cos(crota)

    return header


def make_subcube(sliceParams, pathToInputFile=None, hdu=None, dtype='float32',
                 save=False, overwrite=True, pathToOutputFile=None,
                 get_hdu=False, get_data=True, get_header=True):
    print('\nmaking subcube with the slice parameters {}...'.format(
        sliceParams))

    check_if_all_values_are_none(hdu, pathToInputFile)

    if pathToInputFile is not None:
        hdu = fits.open(pathToInputFile)[0]

    # TODO: remove next line and use sliceParams directly?
    sliceZ, sliceY, sliceX = sliceParams

    data = hdu.data
    header = hdu.header
    # header = transform_header_from_crota_to_pc(header)
    # for key in list(header.keys()):
    #     if key.startswith('PC'):
    #         print(key, header[key])

    data = data[sliceZ, sliceY, sliceX]
    data = data.astype(dtype)
    wcs = WCS(header)
    wcs_cropped = wcs[sliceZ, sliceY, sliceX]
    header.update(wcs_cropped.to_header())
    for key in ['CUNIT1', 'CUNIT2', 'CUNIT3']:
        if key in list(header.keys()):
            header.remove(key)
    header['NAXIS1'] = data.shape[2]
    header['NAXIS2'] = data.shape[1]
    header['NAXIS3'] = data.shape[0]
    header['COMMENT'] = "Cropped FITS file ({})".format(
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))

    if save:
        if pathToOutputFile is None:
            pathToOutputFile = get_path_to_output_file(pathToInputFile, suffix='_sub', filename='subcube.fits')

        save_fits(data, header, pathToOutputFile, verbose=True)

    if get_hdu:
        return fits.PrimaryHDU(data, header)
    elif get_data and (not get_header):
        return data
    elif (not get_data) and get_header:
        return header
    else:
        return data, header


def apply_noise_threshold(data, snr=3, pathToNoiseMap=None,
                          sliceParams=(slice(None), slice(None)),
                          p_limit=0.025, pad_channels=5, use_ncpus=None):
    """"""
    yMax = data.shape[1]
    xMax = data.shape[2]
    n_channels = data.shape[0]
    locations = list(
            itertools.product(range(yMax), range(xMax)))

    if pathToNoiseMap is not None:
        print('\nusing supplied noise map to apply noise threshold '
              'with snr={}...'.format(snr))
        noiseMap = open_fits_file(
            pathToNoiseMap, get_header=False, remove_Stokes=False, check_wcs=False)
        noiseMap = noiseMap[sliceParams]
    else:
        print('\napplying noise threshold to data with snr={}...'.format(snr))
        noiseMap = np.zeros((yMax, xMax))
        maxConsecutiveChannels = max_consecutive_channels(n_channels, p_limit)

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([locations, determine_noise, [data, maxConsecutiveChannels, pad_channels]])

        results_list = gausspyplus.parallel_processing.func(use_ncpus=use_ncpus, function='noise')

        for i, rms in tqdm(enumerate(results_list)):
            if not isinstance(rms, np.float):
                warnings.warn('Problems with entry {} from resulting parallel_processing list, skipping entry'.format(i))
                continue
            else:
                ypos, xpos = locations[i]
                noiseMap[ypos, xpos] = rms

    for idx, (y, x) in enumerate(locations):
        spectrum = data[:, y, x]
        noise = noiseMap[y, x]
        spectrum = spectrum - snr*noise

        if not np.isnan(spectrum).any():
            if len(spectrum[np.nonzero(spectrum)]) == 0:
                spectrum = np.array([0.0])*data.shape[0]
            elif not (spectrum > 0).all():
                mask = np.nan_to_num(spectrum) < 0.  # snr*noise
                spectrum[mask] = 0
                spectrum[~mask] += snr*noise
            else:
                """
                To be implemented -> What to do when spectrum only has
                positive values?
                """
        elif not np.isnan(spectrum).all():
            mask = np.nan_to_num(spectrum) < 0.  # snr*noise
            spectrum[mask] = 0
            spectrum[~mask] += snr*noise

        data[:, y, x] = spectrum

    return data


def change_header(header, format='pp', keep_axis='1', comments=[], dct_keys={}):
    import getpass
    import socket

    prihdr = fits.Header()
    for key in ['SIMPLE', 'BITPIX']:
        prihdr[key] = header[key]

    prihdr['NAXIS'] = 2
    prihdr['WCSAXES'] = 2

    keys = ['NAXIS', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA']

    if format == 'pv':
        keep_axes = [keep_axis, '3']
        prihdr['CTYPE1'] = '        '
        prihdr['CTYPE2'] = '        '
    else:
        keep_axes = ['1', '2']
        keys += ['CTYPE']

    for key in keys:
        if key + keep_axes[0] in header.keys():
            prihdr[key + '1'] = header[key + keep_axes[0]]
        if key + keep_axes[1] in header.keys():
            prihdr[key + '2'] = header[key + keep_axes[1]]

    for key_new, axis in zip(['CDELT1', 'CDELT2'], keep_axes):
        key = 'CD{a}_{a}'.format(a=axis)
        if key in header.keys():
            prihdr[key_new] = header[key]

    prihdr['AUTHOR'] = getpass.getuser()
    prihdr['ORIGIN'] = socket.gethostname()
    prihdr['DATE'] = (datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'), '(GMT)')

    for comment in comments:
        prihdr['COMMENT'] = comment

    for key, val in dct_keys.items():
        prihdr[key] = val

    return prihdr


def get_moment_map(data, header, order=0, linewidth='sigma', vel_unit=u.km/u.s):
    """"""
    wcs = WCS(header)

    #  convert from the velocity unit of the cube to the desired unit
    factor = wcs.wcs.cunit[2].to(vel_unit)
    wcs.wcs.cunit[2] = vel_unit
    wcs.wcs.cdelt[2] *= factor
    wcs.wcs.crval[2] *= factor

    header.update(wcs.to_header())

    bunit = u.Unit('')
    velocity_bin = wcs.wcs.cdelt[2]
    offset = wcs.wcs.crval[2] - wcs.wcs.cdelt[2]*(wcs.wcs.crpix[2] - 1)
    spectral_channels = offset + np.arange(data.shape[0])*wcs.wcs.cdelt[2]
    moment_data = np.zeros(data.shape[1:])
    locations = list(
        itertools.product(range(data.shape[1]), range(data.shape[2])))

    for y, x in locations:
        spectrum = data[:, y, x]
        nanmask = np.logical_not(np.isnan(spectrum))

        if order == 0:
            moment_0 = velocity_bin * np.nansum(spectrum)
            moment_data[y, x] = moment_0
            bunit = u.Unit(header['BUNIT'])
        if order == 1 or order == 2:
            moment_1 = np.nansum(spectral_channels[nanmask]) * spectrum[nanmask]
            moment_data[y, x] = moment_1
        if order == 2:
            numerator = np.nansum(
                (spectral_channels[nanmask] - moment_1)**2 * spectrum[nanmask])
            denominator = np.nansum(spectrum[nanmask])
            moment_2 = np.sqrt(numerator / denominator)
            moment_data[y, x] = moment_2

    header = change_header(
        header, comments=['moment {} map'.format(order)],
        dct_keys={'BUNIT': (bunit * vel_unit).to_string()})

    return fits.PrimaryHDU(moment_data, header)


def moment_map(hdu=None, pathToInputFile=None, sliceParams=None,
               pathToOutputFile=None,
               applyNoiseThreshold=False, snr=3, order=0, linewidth='sigma',
               p_limit=0.025, pad_channels=5,
               vel_unit=u.km/u.s, pathToNoiseMap=None,
               save=False, get_hdu=True, use_ncpus=None,
               restoreNans=False, nanMask=None):
    """
    Previously called 'make_moment_fits'
    """
    # from astropy.wcs import WCS

    print('\ncreate a moment{} fits file from the cube'.format(order))

    check_if_value_is_none(restoreNans, nanMask)
    check_if_all_values_are_none(hdu, pathToInputFile)

    if hdu is None:
        hdu = open_fits_file(pathToInputFile, get_hdu=True)

    if sliceParams is not None:
        hdu = make_subcube(sliceParams, hdu=hdu, get_hdu=True)
        sliceParams = (sliceParams[1], sliceParams[2])
    else:
        sliceParams = (slice(None), slice(None))

    data = hdu.data
    header = hdu.header
    # wcs = WCS(header)

    if applyNoiseThreshold:
        data = apply_noise_threshold(data, snr=snr, sliceParams=sliceParams,
                                     pathToNoiseMap=pathToNoiseMap,
                                     p_limit=p_limit, pad_channels=pad_channels,
                                     use_ncpus=use_ncpus)

    hdu = get_moment_map(data, header, order=order, linewidth=linewidth,
                         vel_unit=vel_unit)

    if restoreNans:
        locations = list(
            itertools.product(
                range(hdu.data.shape[0]), range(hdu.data.shape[1])))
        for ypos, xpos in locations:
            if nanMask[ypos, xpos]:
                hdu.data[ypos, xpos] = np.nan

    if save:
        if order == 2:
            suffix = 'mom2_map_{}'.format(linewidth)
        else:
            suffix = 'mom{}_map'.format(order)
        if pathToOutputFile is None:
            pathToOutputFile = get_path_to_output_file(
                pathToInputFile, suffix=suffix,
                filename='moment{}_map.fits'.format(order))

        save_fits(hdu.data, hdu.header, pathToOutputFile, verbose=True)

    if get_hdu:
        return hdu


def get_pv_map(data, header, sum_over_axis=1, vel_unit=u.km/u.s):
    """"""
    wcs = WCS(header)
    if wcs.wcs.cunit[2] == '':
        warnings.warn('Assuming m/s as spectral unit')
        wcs.wcs.cunit[2] = u.m/u.s
    factor = wcs.wcs.cunit[2].to(vel_unit)
    wcs.wcs.cunit[2] = vel_unit
    wcs.wcs.cdelt[2] *= factor
    wcs.wcs.crval[2] *= factor
    header.update(wcs.to_header())

    data = np.nansum(data, sum_over_axis)

    if sum_over_axis == 1:
        keep_axis = '1'
    else:
        keep_axis = '2'

    header = change_header(header, format='pv', keep_axis=keep_axis)

    hdu = fits.PrimaryHDU(data=data, header=header)

    return hdu


def pv_map(pathToInputFile=None, hdu=None, sliceParams=None,
           pathToOutputFile=None, pathToNoiseMap=None,
           applyNoiseThreshold=False, snr=3, p_limit=0.025, pad_channels=5,
           sumOverLatitude=True, vel_unit=u.km/u.s,
           save=False, get_hdu=True, use_ncpus=None):
    """
    Previously called 'make_pv_fits'
    """
    print('\ncreate a PV fits file from the cube')

    check_if_all_values_are_none(hdu, pathToInputFile)

    if hdu is None:
        hdu = open_fits_file(pathToInputFile, get_hdu=True)

    if sliceParams is not None:
        data, header = make_subcube(sliceParams, hdu=hdu)
        sliceParams = (sliceParams[1], sliceParams[2])
    else:
        sliceParams = (slice(None), slice(None))

    data = hdu.data
    header = hdu.header

    if applyNoiseThreshold:
        data = apply_noise_threshold(data, snr=snr, sliceParams=sliceParams,
                                     pathToNoiseMap=pathToNoiseMap,
                                     p_limit=p_limit, pad_channels=pad_channels,
                                     use_ncpus=use_ncpus)

    wcs = WCS(header)
    #  have to reverse the axis since we change between FITS and np standards
    if sumOverLatitude:
        sum_over_axis = wcs.wcs.naxis - wcs.wcs.lat - 1
    else:
        sum_over_axis = wcs.wcs.naxis - wcs.wcs.lng - 1

    hdu = get_pv_map(data, header, sum_over_axis=sum_over_axis, vel_unit=vel_unit)
    data = hdu.data
    header = hdu.header

    if save:
        if pathToOutputFile is None:
            pathToOutputFile = get_path_to_output_file(pathToInputFile, suffix='_pv', filename='pv_map.fits')

        save_fits(hdu.data, hdu.header, pathToOutputFile, verbose=True)

    if get_hdu:
        return hdu


def combine_fields(listPathToFields=[], ncols=3, nrows=2, save=False,
                   header=None, pathOutputFile=None, comments=[], verbose=True):
    """Combine FITS files to a mosaic by stacking them in the spatial coordinates.

    This will only yield a correct combined mosaic if the original mosaic was split in a similar way as obtained by the get_list_slice_params method

    Parameters
    ----------
    listPathToFields : list
        List of filepaths to the fields that should be mosaicked together.
    ncols : int
        Number of fields in the X direction.
    nrows : int
        Number of fields in the Y direction.
    save : bool
        Set to 'True' if the resulting mosaicked file should be saved.
    header : astropy.io.fits.header.Header
        FITS header that will be used for the combined mosaic.
    pathOutputFile : str
        Filepath to which the combined mosaic gets saved if 'save' is set to 'True'.
    comment : str
        Comment that will be written in the FITS header of the combined mosaic.
    verbose : bool
        Set to 'False' if diagnostic messages should not be printed to the terminal.

    Returns
    -------
    data : numpy.ndarray
        Array of the combined mosaic.
    header : astropy.io.fits.header.Header
        FITS header of the combined mosaic.

    """
    check_if_value_is_none(save, pathOutputFile)

    combined_rows = []

    first = True
    for i, pathToFile in enumerate(listPathToFields):
        if first:
            combined_row = open_fits_file(
                pathToFile=pathToFile, get_header=False, check_wcs=False)
            axes = range(combined_row.ndim)
            axis_1 = axes[-1]
            axis_2 = axes[-2]
            first = False
        else:
            data = open_fits_file(
                pathToFile=pathToFile, get_header=False, check_wcs=False)
            combined_row = np.concatenate((combined_row, data), axis=axis_1)

        if i == 0 and header is None:
            header = open_fits_file(
                pathToFile=pathToFile, get_data=False, check_wcs=False)
        elif (i + 1) % ncols == 0:
            combined_rows.append(combined_row)
            first = True

    for combined_row in combined_rows:
        if first:
            data_combined = combined_row
            first = False
        else:
            data_combined = np.concatenate(
                (data_combined, combined_row), axis=axis_2)

    if comments:
        header = update_header(header, comments=comments)

    if save:
        save_fits(data_combined, header, pathOutputFile, verbose=verbose)

    return data, header
