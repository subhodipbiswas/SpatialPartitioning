#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import json
import time
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from os.path import isfile, join
from utils import Utils
from get_inputs import GetInputs
from make_sol import make_solutions
from functions import set_params, \
    update_property, \
    find_change, \
    parameters, obj_func


class SPATIAL:
    """
    Base class for SPATIAL algorithm. For more details, refer to the publication:
    Biswas S, Chen F, Chen Z, Lu CT, Ramakrishnan N. Incorporating domain knowledge into Memetic Algorithms for solving
    Spatial Optimization problems. InProceedings of the 28th International Conference on Advances in Geographic
    Information Systems 2020 Nov 3 (pp. 25-35).
    """

    def __init__(self, args=None, seed=0, district='lcps', initialization='', pop_size=10, iter_max=1000, min_stag=2):
        self.args = args
        self.seed = seed
        self.district = district
        self.initialization = initialization
        # Hyper-parameters of SPATIAL
        self.pop_size = pop_size
        self.iter_max = iter_max
        self.min_stag = min_stag

    @staticmethod
    def get_partition(districts, district_ids):
        partition = {'districts' : copy.deepcopy(districts),
                     'district_ids': copy.deepcopy(district_ids),
                     'FuncVal': obj_func(districts.keys(), districts)}
        return partition

    def get_neighbors(self, district_id, districts, district_ids):
        """
        Get the list of areas adjacent to the base districts
        """
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)
        adjacency, polygons, polygons_nbr, schools, attributes = self.args[2], self.args[3], self.args[4], self.args[5],\
                                                                 self.args[6]

        centers_list = {_ for _ in schools[attributes['Location']]}
        neighbors = []

        for location in districts[district_id]['MEMBERS']:
            neighbors = neighbors + [a for a in polygons_nbr[location]
                                     if district_ids[a] != district_id and a not in centers_list]

        neighbors = list(set(neighbors))  # get unique values

        # Check which areas break contiguity on being swapped
        to_remove = list()

        for location in neighbors:

            donor_district_id = district_ids[location]
            donor_district_members = [m for m in districts[donor_district_id]['MEMBERS']]

            if location in donor_district_members:
                donor_district_members.remove(location)

            if len(donor_district_members) > 1:  # If the cluster is not a singleton
                donor_districts_adj = adjacency.loc[donor_district_members, donor_district_members].values
                adjacent_mat = csr_matrix(donor_districts_adj)
                num_connect_comp = connected_components(adjacent_mat,
                                                        directed=False,
                                                        return_labels=False)

                if num_connect_comp != 1:  # Not 1 means disconnected
                    to_remove.append(location)
            else:
                to_remove.append(location)

        # Remove those areas that break contiguity
        neighbors = [a for a in neighbors if a not in to_remove]

        return neighbors

    def stop_algo(self, stagnate=0, iter=0):
        """
        Checks if the exit condition is satisfied or not.
        """
        termination = None
        condition = True

        if stagnate > self.min_stag:
            termination = 'Solution stagnation'
            condition = False

        elif iter >= self.iter_max:
            condition = False
            termination = 'Maximum iterations reached'

        return condition, termination

    @staticmethod
    def prob_selection(weights):
        """
        Probabilistic selection
        :param weights: List of values based on which the selection is made
        :return: The selected list element
        """
        total = 0
        winner = 0
        for i, w in enumerate(weights):
            total += w
            if random.random() * total < w:
                winner = i

        return winner

    @staticmethod
    def if_connected(adjacency, district):
        """
        Determines if the spatial units forming a district are connected or not.

        :param adjacency: Adjacency matrix of the spatial units
        :param district: Set of spatial units forming the district
        :return: boolean value 'True' if the district is connected else 'False'
        """
        connected = False

        if len(district) > 0:  # If the cluster is not a singleton
            district_adj = adjacency.loc[district, district].values
            adj_mat = csr_matrix(district_adj)
            num_connect_comp = connected_components(adj_mat,
                                                    directed=False,
                                                    return_labels=False)
            connected = num_connect_comp == 1

        return connected

    def check_move(self, cids, area, districts):
        """
        Check if moving a unit  between the spatial units preserves contiguity
        """
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        adjacency = self.args[2]
        donor_id, recip_id = cids[0], cids[1]
        donor_district = [m for m in districts[donor_id]['MEMBERS']]
        recip_district = [m for m in districts[recip_id]['MEMBERS']]
        donor_connected, recip_connected = False, False

        try:
            # Move 'area' from donor_district to recipient_district
            donor_district.remove(area)
            recip_district.append(area)

            donor_connected = self.if_connected(adjacency, donor_district)
            recip_connected = self.if_connected(adjacency, recip_district)

        except Exception as e:
            pass

        return donor_connected, recip_connected

    def repair(self, zids, districts, district_ids, connecteds):
        """
        Apply the repair operation if the spatial contiguity of a district if broken
        """
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)

        adjacency, polygons_nbr, schools, attributes = self.args[2], self.args[4], self.args[5], self.args[6]
        centers_list = {_ for _ in schools[attributes['Location']]}
        involved_ids = []

        for z in range(len(zids)):
            zid = zids[z]
            involved_ids.append(zid)

            if not connecteds[z]:
                district = [m for m in districts[zid]['MEMBERS']]
                district_adj = adjacency.loc[district, district].values
                adj_mat = csr_matrix(district_adj)
                num_connect_comp, labels = connected_components(adj_mat, directed=False, return_labels=True)
                assert num_connect_comp > 1  # if it is not connected

                # Perform repair process by scanning each connected component
                for c in range(num_connect_comp):
                    connect_sub_district = [district[i] for i in range(len(district)) if labels[i] == c]
                    good = False

                    for area in connect_sub_district:
                        if area in centers_list:
                            good = True  # A 'good' connected component should be kept as it is
                            break

                    if not good:
                        # Perform reassignment to neighboring districts
                        for area in connect_sub_district:
                            nbr_ids = [district_ids[nbr] for nbr in polygons_nbr[area] if district_ids[nbr] != zid]
                            if len(nbr_ids) > 0:
                                nbr_id = nbr_ids[random.randrange(len(nbr_ids))]
                                # Make the move
                                districts[zid]['MEMBERS'].remove(area)
                                districts[nbr_id]['MEMBERS'].append(area)
                                district_ids[area] = nbr_id
                                involved_ids.append(nbr_id)

        # Make sure that the 'involved ids' are connected
        involved_ids = list(set(involved_ids))
        for zid in involved_ids:
            district = [m for m in districts[zid]['MEMBERS']]
            district_adj = adjacency.loc[district, district].values
            adj_mat = csr_matrix(district_adj)
            num_connect_comp = connected_components(adj_mat, directed=False, return_labels=False)
            assert num_connect_comp == 1, "Disconnected components"  # it is not connected

    def make_move(self, zids, area, districts, district_ids, do_repair=True):
        """
        Moves a spatial unit between two districts
        """
        moved = False
        try:
            donor_id, recip_id = zids[0], zids[1]
            assert donor_id != recip_id, "Donor and recipient districts are same!!"
            assert len(districts[donor_id][
                           'MEMBERS']) > 1, "Will result in empty \'donor\' district!!"  # to prevent empty districts
            donor_connected, recipient_connected = self.check_move(zids, area, districts)
            districts[donor_id]['MEMBERS'].remove(area)
            districts[recip_id]['MEMBERS'].append(area)
            district_ids[area] = recip_id
            # Perform repair if either of districts are not connected as a result of the move
            if do_repair and (not donor_connected or not recipient_connected):
                self.repair(zids, districts, district_ids, [donor_connected, recipient_connected])

            moved = True
        except Exception as e:
            pass  # Exception won't effect final solution

        return moved

    def get_borderareas(self, zid, districts=None, district_ids=None):
        """
        Get the list of areas on the border of the base district 'zid'
        """

        args = self.args
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)
        polygons_nbr, schools, attributes = args[4], args[5], args[6]

        if districts is None and district_ids is None:
            districts, district_ids = args[7], args[8]

        centers_list = {_ for _ in schools[attributes['Location']]}

        border_areas = []

        for area in districts[zid]['MEMBERS']:
            for x in polygons_nbr[area]:
                if district_ids[x] != zid and area not in centers_list:
                    border_areas += [area]
                    break

        border_areas = list(set(border_areas))  # get unique list
        return border_areas

    def make_swap(self, i, j, zi, zj, solutions):
        """
        Swap nodes between solutions i and j
        :param i: ID for solution i
        :param j: ID for solution j (more fit)
        :param zi: district ID of a given spatial unit in solution i
        :param zj: district ID of the same spatial unit in solution j
        :param solutions: Copy of the solutions to the spatial optimization problem
        :return: Update solution if successful swap has happened
        """
        districts1, district_ids1 = copy.deepcopy(solutions[i]['districts']),\
                                    copy.deepcopy(solutions[i]['district_ids'])
        districts2, district_ids2 = copy.deepcopy(solutions[j]['districts']),\
                                    copy.deepcopy(solutions[j]['district_ids'])
        swap, solution = False, solutions[i]
        polygons_nbr = self.args[4]

        try:
            # Find mutually exclusive set of spatial units for swapping
            # Set operation: Zj - Zi
            areas_inzj_notzi = [
                m for m in districts2[zj]['MEMBERS']
                if m not in districts1[zi]['MEMBERS']
            ]
            # Set operation: Zi - Zj
            areas_inzi_notzj = [
                m for m in districts1[zi]['MEMBERS']
                if m not in districts2[zj]['MEMBERS']
            ]

            # The spatial units should not be located in the center of the respective districts, otherwise it will lead to holes
            neighbors_areas = self.get_neighbors(zi, districts1, district_ids1)
            border_areas = self.get_borderareas(zi, districts1, district_ids1)
            # Check for overlap
            incoming_areas = [m for m in areas_inzj_notzi if m in neighbors_areas]    # areas for moving into Zi
            outgoing_areas = [m for m in areas_inzi_notzj if m in border_areas]       # areas for moving out of Zi

            # Simultaneously move an (incoming) area into Zi and an (outgoing) area out of Zj
            if len(incoming_areas) > 0 and len(outgoing_areas):  # both sets should be non-empty
                in_area = incoming_areas[random.randrange(len(incoming_areas))]

                # Determine the incoming spatial unit
                in_id = district_ids1[in_area]
                incoming = self.make_move([in_id, zi], in_area, districts1, district_ids1)

                # Determine the outgoing spatial unit
                outgoing = False
                c = 0
                while c < 10 and not outgoing:
                    out_area = outgoing_areas[random.randrange(len(outgoing_areas))]
                    c += 1
                    if in_area == out_area:
                        continue
                    else:
                        out_ids = [district_ids1[a] for a in polygons_nbr[out_area] if district_ids1[a] != zi]
                        if len(out_ids) > 0:
                            out_id = out_ids[random.randrange(len(out_ids))]
                            outgoing = self.make_move([zi, out_id], out_area, districts1, district_ids1)

                swap = incoming and outgoing
        except Exception as e:
            pass

        if swap:
            # print('Swap happened!!')
            update_property(districts1.keys(), self.args, districts1)
            solution = self.get_partition(districts1, district_ids1)

        return swap, solution

    def spatial_recombination(self, solutions, i):
        """
        Perform spatially-aware recombination to modify solution 'i'
        :param solutions: Copy of the solutions to the spatial optimization problem
        :param i: ID for solution i
        :return: Updated solution 'i' by recombination operation
        """
        pop_size = len(solutions)
        # Fitness function for minimization
        fitness = [1.0/(1 + abs(solutions[p]['FuncVal'])) for p in range(pop_size)]
        weights = [fitness[p]/sum(fitness) for p in range(pop_size)]
        # Select another solution
        j = i
        while j == i:
            j = self.prob_selection(weights)
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        schools, attributes = self.args[5], self.args[6]
        center_areas = [_ for _ in schools[attributes['Location']]]
        solution = copy.deepcopy(solutions[i])

        # Recombination operator using the two solutions
        swap = False

        while not swap and len(center_areas) > 0:
            area = center_areas[random.randrange(len(center_areas))]
            # Find the respective districts containing 'area'
            zi, zj = solutions[i]['district_ids'][area], solutions[j]['district_ids'][area]

            if zi == zj:
                swap, solution = self.make_swap(i, j, zi, zj, solutions)

            if not swap:
                center_areas.remove(area)

        return solution

    def local_improvement(self, solutions, i):
        """
        Perform local improvement to modify solution 'i'
        :param solutions: Copy of the solutions to the spatial optimization problem
        :param i: ID for solution i
        :return: Updated solution 'i' by local search
        """
        solution, args = copy.deepcopy(solutions[i]), self.args
        districts, district_ids = solution['districts'], solution['district_ids']
        district_list = [x for x in districts.keys()]
        # utils = Utils(args)
        moved = False
        try:
            random.shuffle(district_list)

            while not moved and len(district_list) > 0:
                recipient = district_list[random.randrange(len(district_list))]
                neighbors = self.get_neighbors(recipient, districts, district_ids)

                while not moved and len(neighbors) > 0:
                    # Randomly select areas until there is a local improvement
                    area = neighbors[random.randrange(len(neighbors))]
                    donor = district_ids[area]  # Define the donor district
                    neighbors.remove(area)  # remove it

                    # Compute the objfunc before and after switch
                    change, possible = find_change([donor, recipient], area, args, districts)
                    # Then make the move, update the list of candidate districts, and return to 4
                    if possible and change <= 0:
                        moved = self.make_move([donor, recipient], area, districts, district_ids, False)

                district_list.remove(recipient)  # remove it so that new districts can be picked
        except Exception as e:
            print('Exception in local_improvement(): {}'.format(e))
        if moved:
            # print('Updating fitness')
            update_property([donor, recipient], args, districts)
            solution = self.get_partition(districts, district_ids)

        return solution

    @staticmethod
    def find_best_sol(solutions, best_f, stagnate):
        """
        Determine the best solution in the population.
        :param solutions: Copy of the solutions to the spatial optimization problem
        :param best_f: The functional value of the present best solution (to be updated)
        :param stagnate: Stagnation counter
        :return: New best solution and its fitness value, updated value of stagnation counter
        """
        new_best_f = math.inf
        new_best_sol = None
        for p in range(len(solutions)):
            if solutions[p]['FuncVal'] < new_best_f:
                new_best_f, new_best_sol = solutions[p]['FuncVal'], p

        # Check if solutions have converged to a local optimum
        if best_f - new_best_f < pow(10, -4):
            stagnate += 1
        else:
            stagnate = 0

        return new_best_f, new_best_sol, stagnate

    def run_module(self, r, solutions, util):
        """
        Run the improvement modules
        :param r: The num number
        :param solutions: Copy of the solutions to the spatial optimization problem
        :param util: Object of the utility class
        :return: Updated solutions
        """

        # Find the best solution and its functional value
        best_func_val, best_sol, stagnation = self.find_best_sol(solutions, math.inf, 0)
        initial = copy.deepcopy(solutions[best_sol])
        print('Iter: 0 \t Best func_val: {:.4f}'.format(best_func_val))

        iteration = 0
        t_start = time.time()
        fval_iter = {0: best_func_val}
        time_iter = {0: (time.time() - t_start) / 60.0}

        # Iteratively improve the solutions using the two search operators
        # Run parallel processing to update the solutions
        num_process = min(self.pop_size, mp.cpu_count())
        pool = mp.Pool(processes=num_process)
        run_results, condition, termination = None, True, " "

        # try:
        for it in tqdm(range(self.iter_max)):
            iteration = it + 1

            for module in range(2):
                output = [
                    (p, pool.apply_async((self.local_improvement, self.spatial_recombination)[module == 1],
                                         args=(solutions, p)
                                         )
                     )
                    for p in range(self.pop_size)
                ]
                # Fitness based selection
                for p, o in output:
                    solution = o.get()
                    if solution['FuncVal'] < solutions[p]['FuncVal']:
                        solutions[p] = solution

            best_func_val, best_sol, stagnation = self.find_best_sol(solutions, best_func_val, stagnation)
            # Print the best result
            if iteration % 20 == 0:
                print('\nIter: {} \t Best func_val: {:.3f}'.format(iteration, best_func_val))
                fval_iter[iteration] = best_func_val
                time_iter[iteration] = (time.time() - t_start) / 60.0
                condition, termination = self.stop_algo(stagnation, iteration)
                if not condition:
                    break

        # Printing the results
        t_elapsed = (time.time() - t_start) / 60.0  # measures in minutes
        best_solution = solutions[best_sol]

        print("Run: {} took {:.2f} min to execute {} iterations...\n"
              " Obtained FuncVal: {:.3f} ".format(r, t_elapsed, iteration, best_func_val))

        # Save the results
        print(' Checking the correctness of the solution...')
        correct = util.check_solution(best_solution['districts'])
        if correct:
            print('\n Correct solution.. Saving results .... ')
            w1, w2, epsilon, _ = parameters()

            prop, info = util.get_alg_params(r,
                                             util.get_params(w1=w1, w2=w2, Iter=iteration, MaxIter=self.iter_max,
                                                             PopSize=self.pop_size),
                                             iteration, t_elapsed, termination, self.seed,
                                             self.initialization, self.district,
                                             initial, best_solution)
            info['fval_vs_iter'] = fval_iter
            info['time_vs_iter'] = time_iter
            run_results = {'properties': prop, 'info': info}
        else:
            print('\n Incorrect solution. Has disconnected districts... \n')

        # except Exception as e:
        #     print("Error: {} in run_module()!!".format(e))

        pool.close()

        return run_results


def make_runs(options):
    """
    Simulate runs for the local search methods.
    """
    init = options.initialization
    district = options.district
    level = options.level
    runs = options.runs
    algo = 'SPATIAL'

    # SPATIAL hyperparameters
    pop_size = (10, 20)[district == 'fcps']
    iter_max = (1000, 2000)[district == 'fcps']

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

    trials = dict()
    util = Utils(args, seeds)
    print('\n Reading existing solutions (if any) \n')
    read_sol_path = util.create_dir(district, 'solutions', init_type[init])
    found = True
    try:
        for i in range(25):
            file_name = join(read_sol_path, "sol{}_{}_{}.json".format(i+1, attributes['Level'], init_type[init]))
            trials[i] = inputs.read_json_file(file_name)
    except Exception as e:
        print("Didn't find existing solutions!! Need to generate them \n")
        found = False

    if not found:
        make_solutions(options)

        for i in range(9999):
            try:
                file_name = join(read_sol_path, "sol{}_{}_{}.json".format(i+1, attributes['Level'], init_type[init]))
                trials[i] = inputs.read_json_file(file_name)
            except Exception as e:
                break

    if trials:
        print('\nRunning {} algorithm\n'.format(algo))
        # try:
        write_path = util.create_dir(district, 'results', init_type[init], algo)

        for r in range(runs):
            print('\n Run {} of {}\n'.format(r + 1, algo))
            # Generate random seeds and use them for instantiating initial trial solutions
            np.random.seed(trials[r]['seed'])
            sequence = np.random.permutation([_ for _ in range(len(trials))]).tolist()
            solutions = {i: trials[s]['solution'] for i, s in enumerate(sequence[:pop_size])}
            # Improve the solutions using SPATIAL algorithm
            spatial = SPATIAL(args, trials[r]['seed'], district, init_type[init], pop_size, iter_max)
            results = spatial.run_module(r + 1, solutions, util)

            with open(join(write_path, "run{}_{}_{}.json".format(r + 1, attributes['Level'], algo)), 'w') as outfile:
                json.dump(results, outfile)
                outfile.close()
        # except Exception as e:
        #     print('Exception in make_runs(): {}'.format(e))
    else:
        print('Couldn\'t run local search. Exiting the program!!')


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
    make_runs(options)

    return 0


if __name__ == "__main__":
    print(sys.platform)
    sys.exit(main())
