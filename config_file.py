# @Author: riener
# @Date:   2019-03-03T20:27:37+01:00
# @Filename: config_file.py
# @Last modified by:   riener
# @Last modified time: 2019-03-10T13:21:37+01:00

import textwrap


def make(mode='simple', outputDir=''):
    config_file = str('#  Configuration file for GaussPy+\n\n')

    config_file += textwrap.dedent(
        """
        [DEFAULT]
        log_output = True
        verbose = True
        overwrite = True
        suffix = ''
        use_ncpus = None

        snr = 3.
        significance = 5.
        snr_noise_spike = 5.
        min_fwhm = 1.
        max_fwhm = None
        fwhm_factor = 2.
        rchi2_limit = 1.5

        two_phase_decomposition = True

        refit_blended = True
        refit_broad = True
        refit_residual = True
        refit_rchi2 = True
        refit_ncomps = True

        p_limit = 0.025
        signal_mask = True
        pad_channels = 5
        min_channels = 100
        mask_out_ranges = []

        random_seed = 111

        main_beam_efficiency = None
        vel_unit = u.km / u.s
        testing = False


        [training]

        training_set = True
        n_spectra = 5
        order = 6
        use_all = False

        params_from_data = True
        alpha1_initial = None
        alpha2_initial = None
        snr_thresh = None
        snr2_thresh = None

        create_training_set = False
        n_channels = None
        ncomps_limits = None
        amp_limits = None
        fwhm_limits = None
        mean_limits = None
        rms = None
        n_spectra_rms = 5000
        n_edge_channels = 10


        [preparation]

        n_spectra_rms = 1000

        gausspy_pickle = True
        data_location = None
        simulation = False

        rms_from_data = True
        average_rms = None


        [decomposition]

        gausspy_decomposition = True
        training_set = False
        save_initial_guesses = False
        alpha1 = None
        alpha2 = None
        snr_thresh = None
        snr2_thresh = None

        improve_fitting = True
        min_offset = 2.
        snr_fit = None
        snr_negative = None
        max_amp_factor = 1.1


        [spatial fitting]

        exclude_flagged = False
        rchi2_limit_refit = None
        max_diff_comps = 2
        max_jump_comps = 2
        n_max_jump_comps = 2
        max_refitting_iteration = 30
        flag_blended = True
        flag_residual = True
        flag_rchi2 = True
        flag_broad = True
        flag_ncomps = True

        mean_separation = 2.
        fwhm_separation = 4.
        fwhm_factor_refit = None
        broad_neighbor_fraction = 0.5
        min_weight = 0.5
        min_pvalue = 0.01
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
            use_ncpus = None

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
