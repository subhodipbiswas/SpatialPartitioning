#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import copy
import random
import argparse
import multiprocessing as mp
from os.path import isfile, join

from utils import Utils
from get_inputs import GetInputs
from functions import set_params,  update_property
from make_sol import make_solutions

from REGAL import SA, SHC, TS


def run_search(r, algo, args, solution, seed, write_path, *argv):
    """
    Performs search by calling required local search procedures
    """
    try:
        # Creat an object method that calls the respective search procedure
        if algo == "SHC":
            method = SHC(args, solution, seed)
        elif algo == "SA":
            method = SA(args, solution, seed)
        elif algo == "TS":
            method = TS(args, solution, seed)

        # Compiling the results of the run
        level = args[6]['Level']
        result = method.search(r, argv)
        with open(join(write_path, "run{}_{}_{}.json".format(r, level, algo)), 'w')\
                as outfile:
            json.dump(result, outfile)
            outfile.close()

    except Exception as e:
        print("Exception in run_search: {}".format(e))


def make_runs(options):
    """
    Simulate runs for the local search methods.
    """
    init = options.initialization
    district = options.district
    level = options.level
    runs = options.runs
    algo = options.algo
    # Read data files
    inputs = GetInputs(district, level)
    args = inputs.get_inputs()  # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attr)
    attributes = args[6]

    # Set weight (w) in the range [0, 1] for calculating F = w * F1 + (1 - w) * F2
    set_params(attributes['weight'])
    # Seeding ensures starting configurations are consistent
    random.seed(options.seed)
    seeds = [s + random.randrange(1000000) for s in range(runs)]
    # Generating starting solutions for each run
    init_type = {1: 'seeded',
                 2: 'distance',
                 3: 'existing'}

    util = Utils(args, seeds)
    print('\n Reading existing solutions (if any) \n')
    solutions = dict()
    read_sol_path = util.create_dir(district, 'solutions', init_type[init])
    try:
        for i in range(runs):
            file_name = join(read_sol_path, "sol{}_{}_{}.json".format(i+1, attributes['Level'], init_type[init]))
            solutions[i] = inputs.read_json_file(file_name)
    except Exception as e:
        print("Didn't find existing solutions!! Need to generate them \n")
        make_solutions(options)

        for i in range(runs):
            file_name = join(read_sol_path, "sol{}_{}_{}.json".format(i+1, attributes['Level'], init_type[init]))
            solutions[i] = inputs.read_json_file(file_name)

    if solutions:
        print('\nRunning {} algorithm\n'.format(algo))
        try:
            # Parallel (asynchronous) version
            num_processes = min(runs, mp.cpu_count() - 1)
            write_path = util.create_dir(district, 'results', init_type[init], algo)

            pool = mp.Pool(processes=num_processes)
            output = [
                pool.apply_async(run_search,
                                 args=(r + 1,
                                       algo, args,
                                       solutions[r]['solution'], solutions[r]['seed'],
                                       write_path, init_type[init], district
                                       )
                                 )
                for r in range(runs)
            ]
            pool.close()
            results = [p.get() for p in output]

        except Exception as e:
            print('Exception in make_runs(): {}'.format(e))
    else:
        print('Couldn\'t run local search. Exiting the program!!')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    parser.add_argument("-a", "--algo", type=str, default="TS")  # district: lcps, fcps
    parser.add_argument("-d", "--district", type=str, default="lcps")  # district: lcps, fcps
    parser.add_argument("-l", "--level", type=str, default="ES")  # school levels: ES, MS, HS
    parser.add_argument("-r", "--runs", type=int, default=25)  # number of runs to be simulated
    parser.add_argument("-e", "--seed", type=int, default=17)  # integer seed for random number generator
    parser.add_argument("-i", "--initialization", default=1, type=int)  # 1: seeded, 2: distance 3: existing
    options = parser.parse_args()
    make_runs(options)

    return 0


if __name__ == "__main__":
    print(sys.platform)
    sys.exit(main())
