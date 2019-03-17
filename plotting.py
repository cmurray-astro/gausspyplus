# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: plotting.py
# @Last modified by:   riener
# @Last modified time: 2019-03-17T15:03:08+01:00

import itertools
import os
import pickle
import random
import sys

import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from tqdm import tqdm

from gausspyplus.shared_functions import gaussian, combined_gaussian, goodness_of_fit


def get_points_for_colormap(vmin, vmax, central_val=0.):
    lower_interval = abs(central_val - vmin)
    upper_interval = abs(vmax - central_val)

    if lower_interval > upper_interval:
        start = 0.
        stop = 0.5 + (upper_interval / lower_interval)*0.5
    else:
        start = 0.5 - (lower_interval / upper_interval)*0.5
        stop = 1.
    return start, stop


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def pickle_load_file(pathToFile):
    with open(os.path.join(pathToFile), "rb") as pickle_file:
        if (sys.version_info > (3, 0)):
            data = pickle.load(pickle_file, encoding='latin1')
        else:
            data = pickle.load(pickle_file)
    return data


def get_list_indices(data, subcube=False, pixel_range=None,
                     list_indices=None, n_spectra=None, random_seed=111):
    random.seed(random_seed)
    # TODO: incorporate the nan_mask in this scheme
    grid_layout = None
    if subcube or (pixel_range is not None):
        if pixel_range is not None:
            xmin, xmax = pixel_range['x']
            ymin, ymax = pixel_range['y']
        else:
            ymin, xmin = min(data['location'])
            ymax, xmax = max(data['location'])
        yValues = np.arange(ymin, ymax + 1)[::-1]
        xValues = np.arange(xmin, xmax + 1)
        locations = list(itertools.product(yValues, xValues))
        cols = len(xValues)
        rows = len(yValues)
        grid_layout = [cols, rows]
        n_spectra = cols*rows

        list_indices = []
        for location in locations:
            list_indices.append(data['location'].index(location))
    elif list_indices is None:
        if n_spectra is None:
            n_spectra = len(data['data_list'])
            list_indices = np.arange(n_spectra)
        else:
            list_indices = []
            nIndices = len(data['data_list'])
            randomIndices = random.sample(range(nIndices), nIndices)
            for idx in randomIndices:
                if 'nan_mask' in data.keys():
                    yi, xi = data['location'][idx]
                    if data['nan_mask'][:, yi, xi].all() != True:
                        list_indices.append(idx)
                        if len(list_indices) == n_spectra:
                            break
                else:
                    list_indices.append(idx)
                    if len(list_indices) == n_spectra:
                        break
    else:
        n_spectra = len(list_indices)

    list_indices = [i for i in list_indices if data['data_list'][i] is not None]
    n_spectra = len(list_indices)

    return list_indices, n_spectra, grid_layout


def get_figure_params(n_channels, n_spectra, cols, rowsize, rowbreak,
                      grid_layout, subcube=False):
    if n_channels > 700:
        colsize = round(rowsize*n_channels/659, 2)
    else:
        colsize = rowsize

    # if subcube is False:
    if grid_layout is None:
        rows = int(n_spectra / (cols))
        if n_spectra % cols != 0:
            rows += 1

        multiple_pdfs = True
        if rows < rowbreak:
            rowbreak = rows
            multiple_pdfs = False
    else:
        cols, rows = grid_layout
        rowbreak = rows
        multiple_pdfs = False

    if (rowbreak*rowsize*100 > 2**16) or (cols*colsize*100 > 2**16):
        errorMessage = \
            "Image size is too large. It must be less than 2^16 pixels in each direction. Restrict the number of columns or rows."
        raise Exception(errorMessage)

    return cols, rows, rowbreak, colsize, multiple_pdfs


def add_figure_properties(ax, rms, figMinChannel, figMaxChannel,
                          header=None, residual=False, fontsize=10):
    ax.set_xlim(figMinChannel, figMaxChannel)
    if header is not None:
        ax.set_xlabel('Velocity [km/s]', fontsize=fontsize)
    else:
        ax.set_xlabel('Channels', fontsize=fontsize)
    ax.set_ylabel('T_B (K)', fontsize=fontsize)

    ax.tick_params(labelsize=fontsize - 2)

    ax.axhline(color='black', ls='solid', lw=0.5)
    ax.axhline(y=rms, color='red', ls='dotted', lw=0.5)
    ax.axhline(y=-rms, color='red', ls='dotted', lw=0.5)

    if not residual:
        ax.axhline(y=3*rms, color='red', ls='dashed', lw=1)
    else:
        ax.set_title('Residual', fontsize=fontsize)


def plot_signal_ranges(ax, data, idx, figChannels):
    if 'signal_ranges' in data.keys():
        for low, upp in data['signal_ranges'][idx]:
            ax.axvspan(figChannels[low], figChannels[upp - 1], alpha=0.1, color='indianred')


def get_title_string(idx, index_data, xi, yi, nComponents, rchi2, rchi2gauss):
    idx_string = ''
    if index_data is not None:
        if index_data != idx:
            idx_string = ' (Idx$_{{data}}$={})'.format(index_data)

    if xi is None:
        loc_string = ''
    else:
        loc_string = ', X={}, Y={}'.format(xi, yi)

    if rchi2 is None:
        rchi2_string = ''
    else:
        rchi2_string = ', $\\chi_{{red}}^{{2}}$={:.3f}'.format(rchi2)

    if rchi2gauss is None:
        rchi2gauss_string = ''
    else:
        rchi2gauss_string = '\\chi_{{red, gauss}}^{{2}}$={:.3f}'.format(rchi2gauss)

    title = 'Idx={}{}{}, N={}{}{}'.format(
        idx, idx_string, loc_string, nComponents, rchi2_string, rchi2gauss_string)
    return title


def scale_fontsize(rowsize):
    rowsize_scale = 4
    if rowsize >= rowsize_scale:
        fontsize = 10 + int(rowsize - rowsize_scale)
    else:
        fontsize = 10 - int(rowsize - rowsize_scale)
    return fontsize


def plot_spectra(pathToDataPickle, pathToPlots, pathToDecompPickle=None,
                 training_set=False, cols=5, rowsize=7.75, rowbreak=50, dpi=50,
                 n_spectra=None, suffix='', subcube=False, pixel_range=None,
                 list_indices=None, gaussians=True, residual=True, signal_ranges=True, random_seed=111):

    print("\nMake plots...")

    #  check if all necessary files are supplied
    if (pathToDecompPickle is None) and (training_set is False):
        errorMessage = """'pathToDecompPickle' needs to be specified for 'training_set=False'"""
        raise Exception(errorMessage)

    fileName, fileExtension = os.path.splitext(os.path.basename(pathToDataPickle))

    data = pickle_load_file(pathToDataPickle)
    if not training_set:
        decomp = pickle_load_file(pathToDecompPickle)

    channels, figChannels = (data['x_values'] for _ in range(2))
    n_channels = len(channels)
    # TODO. check if figChannels[-1] is correct
    figMinChannel, figMaxChannel = figChannels[0], figChannels[-1]

    if 'header' in data.keys():
        header = data['header']
        offset = header['CRVAL3'] - header['CDELT3']*(header['CRPIX3'] - 1)
        figChannels = (offset + figChannels*header['CDELT3']) / 1000
        figMinChannel = round(offset / 1000)
        figMaxChannel = round((offset + header['NAXIS3']*header['CDELT3']) / 1000)
    else:
        header = None

    list_indices, n_spectra, grid_layout = get_list_indices(
        data, subcube=subcube, pixel_range=pixel_range, list_indices=list_indices, n_spectra=n_spectra, random_seed=random_seed)

    cols, rows, rowbreak, colsize, multiple_pdfs = get_figure_params(
        n_channels, n_spectra, cols, rowsize, rowbreak, grid_layout, subcube=subcube)

    fontsize = scale_fontsize(rowsize)

    figsize = (cols*colsize, rowbreak*rowsize)
    fig = plt.figure(figsize=figsize)

    # set up subplot grid
    # gridspec.GridSpec(cols, rowbreak, wspace=0., hspace=0.)

    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    # fig.subplots_adjust(hspace=0.5)

    pbar = tqdm(total=n_spectra)

    for i, idx in enumerate(list_indices):
        pbar.update(1)
        if 'location' in data.keys():
            yi, xi = data['location'][idx]
        else:
            yi, xi = (None for _ in range(2))

        if 'index' in data.keys():
            index_data = data['index'][idx]
        else:
            index_data = None

        if rowbreak is not None:
            k = int(i / (rowbreak*cols))
            if (k + 1)*rowbreak > rows:
                rows_in_figure = rows - k*rowbreak
            else:
                rows_in_figure = rowbreak

        y = data['data_list'][idx]
        rms = data['error'][idx][0]

        if training_set:
            fit_fwhms = data['fwhms'][idx]
            fit_means = data['means'][idx]
            fit_amps = data['amplitudes'][idx]
        else:
            fit_fwhms = decomp['fwhms_fit'][idx]
            fit_means = decomp['means_fit'][idx]
            fit_amps = decomp['amplitudes_fit'][idx]

        row_i = int((i - (k*rowbreak*cols)) / cols)*3
        col_i = i % cols
        ax = plt.subplot2grid((3*rows_in_figure, cols),
                              (row_i, col_i), rowspan=2)

        ax.step(figChannels, y, color='black', lw=0.5)

        combined_gauss = combined_gaussian(
            fit_amps, fit_fwhms, fit_means, channels)
        ax.plot(figChannels, combined_gauss, lw=2, color='orangered')

        nComponents = len(fit_amps)

        # Plot individual components
        if gaussians:
            for j in range(nComponents):
                gauss = gaussian(
                    fit_amps[j], fit_fwhms[j], fit_means[j], channels)
                ax.plot(figChannels, gauss, ls='solid', lw=1, color='orangered')

        if training_set:
            # TODO: incorporate signal_interval here??
            # rchi2 = goodness_of_fit(y, combined_gauss, rms, nComponents)
            rchi2 = data['best_fit_rchi2'][idx]
        else:
            rchi2 = decomp['best_fit_rchi2'][idx]

        if signal_ranges:
            plot_signal_ranges(ax, data, idx, figChannels)

        rchi2gauss = None
        # TODO: incorporate rchi2_gauss

        title = get_title_string(idx, index_data, xi, yi, nComponents, rchi2, rchi2gauss)
        ax.set_title(title, fontsize=fontsize)

        add_figure_properties(ax, rms, figMinChannel, figMaxChannel, header=header,
                              fontsize=fontsize)

        if residual:
            row_i = int((i - k*(rowbreak*cols)) / cols)*3 + 2
            col_i = i % cols
            ax = plt.subplot2grid((3*rows_in_figure, cols),
                                  (row_i, col_i))

            ax.step(figChannels, y - combined_gauss, color='black', lw=0.5)
            if signal_ranges:
                plot_signal_ranges(ax, data, idx, figChannels)

            add_figure_properties(ax, rms, figMinChannel, figMaxChannel, header=header,
                                  residual=True, fontsize=fontsize)

        if ((i + 1) % (rowbreak*cols) == 0) or ((i + 1) == n_spectra):
            if multiple_pdfs:
                filename = '{}{}_plots_part_{}.pdf'.format(fileName, suffix, k + 1)
            else:
                filename = '{}{}_plots.pdf'.format(fileName, suffix)

            fig.tight_layout()

            if not os.path.exists(pathToPlots):
                os.makedirs(pathToPlots)
            pathname = os.path.join(pathToPlots, filename)
            fig.savefig(pathname, dpi=dpi, overwrite=True)
            plt.close()
            print("\n>> saved '{}' in {}".format(filename, pathToPlots))

            remaining_rows = rowbreak
            #  for last iteration
            if (k + 2)*rowbreak > rows:
                remaining_rows = rows - (k + 1)*rowbreak
                figsize = (cols*colsize, remaining_rows*rowsize)

            fig = plt.figure(figsize=figsize)

            # set up subplot grid
            gridspec.GridSpec(cols, remaining_rows, wspace=0.2, hspace=0.2)

            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)
            fig.subplots_adjust(hspace=0.5)
    pbar.close()
