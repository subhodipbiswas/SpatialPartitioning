#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import copy
import random
import argparse
import multiprocessing as mp
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import Utils
from get_inputs import GetInputs
from functions import set_params,  update_property


def make_solutions(options):
    """
    Simulate runs for the local search methods.
    """
    init = options.initialization
    district = options.district
    level = options.level
    runs = options.runs

    # Read data files
    # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attr)
    inputs = GetInputs(district, level)
    args = inputs.get_inputs()
    schools, attributes = args[5], args[6]

    # Map the existing districts to our district numbering
    mapping = {}
    for index, school in schools.iterrows():
        location, district_id = schools[attributes['Location']][index], 'District{}'.format(index)
        school_id = school['SCH_CODE']
        mapping[district_id] = school_id

    # Set weight (w) in the range [0, 1] for calculating F = w * F1 + (1 - w) * F2
    weight = args[6]['weight']
    set_params(weight)

    # Seeding ensures starting configurations are consistent
    random.seed(options.seed)
    seeds = [s + random.randrange(1000000) for s in range(runs)]
    # Generating starting solutions for each run
    init_type = {1: 'seeded',
                 2: 'distance',
                 3: 'existing'}

    print('\n Generating {} {} solutions!! \n'.format(len(seeds), init_type[init]))
    util = Utils(args, seeds)
    solutions = copy.deepcopy(util.gen_solutions(init, runs, args, seeds))

    print('\n Checking the correctness of the solutions \n')
    valid, disconnections = False, []
    write_path = util.create_dir(district, 'solutions', init_type[init])

    for i in range(runs):
        valid, disconnections = util.check_solution(solutions[i]['districts'])

        if valid and len(disconnections) == 0:
            print('Solution {} is a valid solution!! '.format(i + 1), end='\t')
        else:
            print('Solution {} is not valid... '.format(i+1), end='\t')
            print('\t Initial functional value: {:.3f}'.format(solutions[i]['FuncVal']))

            while not valid or len(disconnections) > 0:
                # print('Repairing solution {}...'.format(i+1))
                valid, disconnections = util.repair(disconnections,
                                                    solutions[i]['districts'],
                                                    solutions[i]['district_ids'])
            if valid and len(disconnections) == 0:
                print("Repaired solution {:2}. Updating fitness...".format(i+1), end=' ')
                update_property(solutions[i]['districts'].keys(), args, solutions[i]['districts'])
                solutions[i] = util.get_partition(solutions[i]['districts'], solutions[i]['district_ids'])

        print(' Final functional value: {:.3f}.'.format(solutions[i]['FuncVal']))
        print(' Saving solution ', i+1)
        solution = {'solution': solutions[i],
                    'mapping': mapping,
                    'seed': seeds[i]}
        try:
            with open(join(write_path, "sol{}_{}_{}.json".format(i+1, attributes['Level'], init_type[init])), 'w')\
                    as outfile:
                json.dump(solution, outfile)
                outfile.close()

        except Exception as e:
            print('Exception in make_sol(): {}'.format(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    parser.add_argument("-d", "--district", type=str, default="lcps")  # district: lcps, fcps
    parser.add_argument("-l", "--level", type=str, default="ES")  # school levels: ES, MS, HS
    parser.add_argument("-r", "--runs", type=int, default=25)  # number of runs to be simulated
    parser.add_argument("-e", "--seed", type=int, default=17)  # integer seed for random number generator
    parser.add_argument("-i", "--initialization", default=1, type=int)  # 1: seeded, 2: distance 3: existing
    # parser.add_argument("-y", "--year", type=int, default=2020)  # school year: 2020-21
    options = parser.parse_args()
    make_solutions(options)

    return 0


if __name__ == "__main__":
    print(sys.platform)
    sys.exit(main())
