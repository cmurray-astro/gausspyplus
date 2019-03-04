# @Author: riener
# @Date:   2019-03-03T20:27:37+01:00
# @Filename: config_file.py
# @Last modified by:   riener
# @Last modified time: 2019-03-04T12:03:30+01:00

import configparser
import textwrap
from astropy import units as u


def make(mode='simple', outputDir=''):
    config_file = str('#  Configuration file for GaussPy+\n\n')

    config_file += textwrap.dedent(
        """
        [DEFAULT]
        log_output = True
        verbose = True
        overwrite = True
        suffix = ''
        use_nCpus = None

        snr = 3.
        significance = 5.
        snr_noise_spike = 5.
        min_fwhm: 1.
        max_fwhm: None
        fwhm_factor: 2.
        rchi2_limit = 1.5

        twoPhaseDecomposition = True

        refit_blended: True
        refit_broad: True
        refit_residual: True
        refit_rchi2 = False
        refit_ncomps = False

        pLimit = 0.025
        signalMask = True
        padChannels = 5
        minChannels = 100
        mask_out_ranges = []

        random_seed = 111

        main_beam_efficiency = None
        vel_unit = u.km / u.s
        testing = False


        [training]

        pathToTrainingSet = ''
        trainingSet = True
        numberSpectra = 5
        order = 6
        use_all = False

        paramsFromData = True
        alpha1_guess = None
        alpha2_guess = None
        snrThresh = None
        snr2Thresh = None

        createTrainingSet = False
        nChannels = None
        nSpectra = None
        nCompsLims = None
        ampLims = None
        fwhmLims = None
        meanLims = None
        rms = None
        numberRmsSpectra = 5000
        meanEdgeChans = 10


        [preparation]

        gpyDirname = ''
        numberRmsSpectra = 1000

        gausspyPickle = True
        dataLocation = None
        simulation = False

        rmsFromData = True
        average_rms = None


        [decomposition]

        gaussPyDecomposition = True
        trainingSet = False
        saveInitialGuesses = False
        alpha1 = None
        alpha2 = None
        snrThresh = None
        snr2Thresh = None
        decompDirname = ''

        improve_fitting: True
        min_offset: 2.
        snr_fit: None
        snr_negative: None
        max_amp_factor: 1.1


        [spatial fitting]

        exclude_flagged = False
        rchi2_limit_refit = None
        maxDiffComps = 2
        maxJumpComps = 2
        nMaxJumpComps = 2
        max_refitting_iteration = 20
        flag_blended = False
        flag_residual = False
        flag_rchi2 = False
        flag_broad = False
        flag_ncomps = False

        mean_separation = 2.
        fwhm_separation = 4.
        fwhm_factor_refit = None
        broad_neighbor_fraction = 0.5
        min_weight = 0.5
        only_print_flags = False
        """)

    with open(outputDir + 'gausspy+.ini', 'w') as file:
        file.write(config_file)

    def test():
        config_file += textwrap.dedent(
            """
            [DEFAULT]

            #  parent directory
            dirpath = ''

            #
            filename = ''

            #  number of CPUs to use in multiprocessing (default: 75% of all CPUs)
            use_nCpus = None

            #  minimum signal-to-noise (S/N) ratio
            snr = 3.

            #  minimum significance value
            significance = 5.

            #  minimum S/N ratio for noise spikes
            snr_noise_spike = 5.

            #  output unit for the spectral axis
            vel_unit = u.km / u.s

            #
            randomSeed = None
            """)

        extended_params = textwrap.dedent(
            """
            min_fwhm = 1.
            max_fwhm = None
            #  log messages printed to the terminal in 'gpy_log' directory [True/False]
            log_output = True

            #  print messages to the terminal [True/False]
            verbose = True

            #  overwrite files [True/False]
            overwrite = True

            #  suffix added to filename
            suffix = ''

            #  Main beam efficiency
            main_beam_efficiency = None

            #
            testing = False

            #
            removeHeaderKeywords = []

            #
            headerComments = None

            #
            restoreNans = True
            """)

        if mode == 'extended':
            config_file += extended_params
