# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: parallel_processing.py
# @Last modified by:   riener
# @Last modified time: 2019-03-04T10:21:25+01:00

"""Parallelization routine.

Used by gpy_compare.py, moment_masking.py
"""
import multiprocessing
import numpy as np
# import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from gausspyplus.spectral_cube_functions import determine_noise
from gausspyplus.prepare import GaussPyPrepare
from gausspyplus.spatial_fitting import SpatialFitting

# ------------MULTIPROCESSING------------

# # TODO: check if this is really necessary
# def init_worker():
#     """Worker initializer to ignore Keyboard interrupt."""
#     signal.signal(signal.SIGINT, signal.SIG_IGN)


def init(mp_info):
    global mp_ilist, mp_data, mp_params
    mp_data, mp_params = mp_info
    mp_ilist = np.arange(len(mp_data))


def calculate_noise(i):
    xpos = mp_data[i][1]
    ypos = mp_data[i][0]
    spectrum = mp_params[0][:, ypos, xpos]
    result = determine_noise(spectrum, maxConsecutiveChannels=mp_params[1], padChannels=mp_params[2], idx=i, averageRms=mp_params[3])
    return result


def refit_spectrum_1(i):
    result = SpatialFitting.refit_spectrum_phase_1(mp_params[0], mp_data[i], i)
    return result


def refit_spectrum_2(i):
    result = SpatialFitting.refit_spectrum_phase_2(mp_params[0], mp_data[i], i)
    return result


def calculate_noise_gpy(i):
    result = GaussPyPrepare.calculate_rms_noise(mp_params[0], mp_data[i], i)
    return result


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """A parallel version of the map function with a progress bar.

    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of array
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
            keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
            Useful for catching bugs
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def func(use_nCpus=None, function='noise'):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_nCpus is None:
        use_nCpus = int(ncpus*0.75)
    print('Using {} of {} cpus'.format(use_nCpus, ncpus))
    try:
        if function == 'noise':
            results_list = parallel_process(mp_ilist, calculate_noise, n_jobs=use_nCpus)
        elif function == 'gpy_noise':
            results_list = parallel_process(mp_ilist, calculate_noise_gpy, n_jobs=use_nCpus)
        elif function == 'refit_phase_1':
            results_list = parallel_process(mp_ilist, refit_spectrum_1, n_jobs=use_nCpus)
        elif function == 'refit_phase_2':
            results_list = parallel_process(mp_ilist, refit_spectrum_2, n_jobs=use_nCpus)
        # results_list = p.map(determine_distance, tqdm(ilist))
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list
