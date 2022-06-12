import math
import os
import copy
import time
import random
from pprint import pprint
import geopandas as gpd
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from functions import obj_func,\
    update_property,\
    find_change,\
    parameters,\
    target_compact, target_balance


class Utils:
    """
    Utility class containing necessary functions..
    """

    def __init__(self, args=None, seeds=None):
        self.args = args
        self.seeds = seeds

    def gen_solutions(self, init=1, num_sol=1, args=None, seeds=None):
        """Generate solutions using initialize()"""

        if args is None:
            args = self.args
        if seeds is None:
            seeds = self.seeds

        solutions = dict()
        try:
            if num_sol > 1:
                # Parallel (asynchronous) solution initialization
                pool = mp.Pool(processes=min(mp.cpu_count() - 1, num_sol))
                output = [(i,
                           pool.apply_async(self.initialize,
                                            args=(init,
                                                  args,
                                                  (None, seeds[i])[seeds is not None]
                                                  )
                                            )
                           )
                          for i in range(num_sol)
                          ]
                pool.close()

                for i, p in output:
                    districts, district_ids = p.get()
                    solutions[i] = self.get_partition(districts, district_ids)
            else:
                # Serial execution applies for 'existing' solution
                districts, district_ids = self.initialize(init, args)
                solutions[0] = self.get_partition(districts, district_ids)

        except Exception as e:
            print(e)
            print("Couldn\'t generate solution(s)!!")

        return solutions

    def initialize(self, init, args=None, seed=None):
        """Initializes districts with different schemes"""

        if args is None:
            args = self.args
        # 'seeded' initialization based on the centroids
        if init == 1:
            districts, district_ids = self.seeded_init(args, seed)
        # 'distance' initialization
        elif init == 2:
            districts, district_ids = self.dist_init(args, seed)
        # 'existing' initialization
        elif init == 3:
            districts, district_ids = self.exist_init(args)
        else:
            pass

        # This is call to an outside method, needs to be resolved
        update_property(districts.keys(), args, districts)
        # pprint(districts)

        return districts, district_ids

    def seeded_init(self, args=None, seed=None):
        """
        Seeded initialization starts with assigning the centers to each district and then build on them
        using the adjacency relationship between the polygons.
        Args:
            args:
            seed:

        Returns:

        """
        if args is None:
            args = self.args

        random.seed(seed)

        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        polygons, polygons_nbr, schools, attributes = args[3], args[4], args[5], args[6]
        districts, district_ids = dict(), dict()
        # Enumerate districts and set status
        polygons_list = [s for s in polygons_nbr.keys()]
        for location in polygons_list:
            district_ids[location] = -1  # -1 means not assigned to a districts

        # Initialize the districts with center polygons
        district_list = []
        for index, school in schools.iterrows():
            location, district_id = schools[attributes['Location']][index], 'District{}'.format(index)
            district_ids[location] = district_id
            polygons_list.remove(location)      # Remove polygons that have already been assigned to a district
            district_list.append(district_id)
            districts[district_id] = self.get_params(MEMBERS=[location],
                                                     DISTRICT=district_id)

        while len(polygons_list) > 0:
            # Pick a random districts
            district_id = district_list[random.randrange(len(district_list))]
            members = [x for x in districts[district_id]['MEMBERS']]
            # Get list of free areas around it
            neighbors = self.get_adj_areas(members, polygons_nbr, district_ids)

            if len(neighbors) > 0:
                location = neighbors[random.randrange(len(neighbors))]
                district_ids[location] = district_id
                districts[district_id]['MEMBERS'].append(location)
                polygons_list.remove(location)
            else:
                district_list.remove(district_id)

        if list(district_ids.values()).count(-1) > 0:
            print('There are unassigned polygons present. Error!!')

        return districts, district_ids

    def exist_init(self, args=None):
        """
        Extracts the existing partition for evaluation
        """
        if args is None:
            args = self.args

        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        polygons, polygons_nbr, schools, attributes = args[3], args[4], args[5], args[6]

        districts, district_ids = dict(), dict()
        # Enumerate districts and set status
        polygons_list = [s for s in polygons_nbr.keys()]
        for location in polygons_list:
            district_ids[location] = -1  # -1 means not assigned to a districts

        # Map the existing districts to our district numbering
        mapping = {}
        # Initialize the districts with school-containing polygons
        district_list = []
        for index, school in schools.iterrows():
            location, district_id = schools[attributes['Location']][index], 'District{}'.format(index)
            district_ids[location] = district_id
            polygons_list.remove(location)      # Remove polygons that have already been assigned to a district
            district_list.append(district_id)
            districts[district_id] = self.get_params(MEMBERS=[location],
                                                     DISTRICT=district_id)
            school_id = school['SCH_CODE']
            mapping[school_id] = district_id

        # Get the existing partition from the data
        for index, polygon in polygons.iterrows():
            location = polygon[attributes['Location']]

            if location in polygons_list:
                school_id = polygon['{}_CODE'.format(attributes['Level'])]
                district_id = mapping[school_id]
                district_ids[location] = district_id
                districts[district_id]['MEMBERS'].append(location)
                polygons_list.remove(location)

        return districts, district_ids

    def dist_init(self, args=None, seed=None):
        """
        Distance-based initialization to grow compact districts/clusters initially.
        Args:
            args:
            seed:

        Returns:

        """
        if args is None:
            args = self.args

        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        polygons, polygons_nbr, schools, attributes = args[3], args[4], args[5], args[6]

        districts, district_ids = dict(), dict()
        # Enumerate districts and set status
        polygons_list = [s for s in polygons_nbr.keys()]
        for location in polygons_list:
            district_ids[location] = -1  # -1 means not assigned to a districts

        # Map the existing districts to our district numbering
        mapping = {}
        # Initialize the districts with school-containing polygons
        district_list = []
        for index, school in schools.iterrows():
            location, district_id = schools[attributes['Location']][index], 'District{}'.format(index)
            district_ids[location] = district_id
            polygons_list.remove(location)      # Remove polygons that have already been assigned to a district
            district_list.append(district_id)
            districts[district_id] = self.get_params(MEMBERS=[location],
                                                     DISTRICT=district_id)
            school_id = school['SCH_CODE']
            mapping[school_id] = district_id

        num_districts = len(district_list)     # No. of schools

        while len(polygons_list) > 0:
            # Pick a random districts
            district_id = district_list[random.randrange(num_districts)]
            members = [x for x in districts[district_id]['MEMBERS']]
            # Get list of free areas around it
            neighbors = self.get_adj_areas(members, polygons_nbr, district_ids)

            # Find the nearest unassigned polygon and assign it to District 'district_id' for the sake of compactness
            center_member = polygons[polygons[attributes['Location']] == districts[district_id]['MEMBERS'][0]]
            best_neighbor, best_distance = None, math.inf

            for n in neighbors:
                neighbor = polygons[polygons[attributes['Location']] == n]

                for ind, loc in neighbor.iterrows():
                    for index, center in center_member.iterrows():
                        dist = int(loc.geometry.centroid.distance(center.geometry.centroid))
                        if dist <= best_distance:
                            best_distance = dist
                            best_neighbor = n

            if best_neighbor:
                district_ids[best_neighbor] = district_id
                districts[district_id]['MEMBERS'].append(best_neighbor)
                polygons_list.remove(best_neighbor)

        if list(district_ids.values()).count(-1) > 0:
            print('There are unassigned polygons present. Error!!')

        return districts, district_ids

    @staticmethod
    def get_neighbors(district_id, args):
        """
        Get the list of areas adjacent to the base districts
        """
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)
        adjacency, polygons, polygons_nbr, schools, attributes = args[2], args[3], args[4], args[5], args[6]
        districts, district_ids = args[7], args[8]

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

    @staticmethod
    def get_adj_areas(locations, polygons_nbr, district_ids):
        """Returns adjacent unassigned polygons to a cluster"""
        adj_locations = []
        if len(locations) > 0:
            for location in locations:
                adj_locations = adj_locations + [a for a in polygons_nbr[location]
                                                 if district_ids[a] == -1]

            adj_locations = list(set(adj_locations))

        return adj_locations

    def get_partition(self, districts, district_ids):
        """

        """
        r = copy.deepcopy(districts)
        i = copy.deepcopy(district_ids)
        f = obj_func(r.keys(), r)

        partition = self.get_params(districts=r,
                                    district_ids=i,
                                    FuncVal=f)
        return partition

    @staticmethod
    def make_move(cids, area, args):
        """Moving polygon between clusters"""

        donor_district_id = cids[0]
        recip_district_id = cids[1]
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)
        districts, district_ids = args[7], args[8]

        donor_members = [m for m in districts[donor_district_id]['MEMBERS']]
        recip_members = [m for m in districts[recip_district_id]['MEMBERS']]

        moved = False
        try:
            if len(donor_members) > 1:
                # Make the move
                donor_members.remove(area)
                recip_members.append(area)

                # Update the districts
                districts[donor_district_id]['MEMBERS'] = donor_members
                districts[recip_district_id]['MEMBERS'] = recip_members
                district_ids[area] = recip_district_id

                update_property(cids, args)
                moved = True

        except Exception as e:
            print('Exception in make_move():{}'.format(e))

        return moved

    def get_alg_params(self, run, alg_params, *argv):
        # argv = (iteration, t_elapsed, termination, seed, initialization, state, initial, final)
        iteration = argv[0]
        t_elapsed = argv[1]
        terminate = argv[2]
        seed = argv[3]
        initialize = argv[4]
        state = argv[5]

        """Consolidating all the attributes"""
        params = self.get_params(Seed=seed,
                                 State=state,
                                 AlgParams=alg_params,
                                 Iteration=iteration,
                                 TimeElapsed=t_elapsed,
                                 Termination=terminate,
                                 Initialization=initialize)

        initial, final = argv[6], argv[7]
        existing = self.gen_solutions(3)[0]

        print("Run: {}\t Iterations/Flips: {}\t Time: {:.4f} min\t"
              "  Initial CV: {:.4f} Final CV {:.4f} Existing CV: {:.4f}".format(run,
                                                                                iteration,
                                                                                t_elapsed,
                                                                                initial['FuncVal'],
                                                                                final['FuncVal'],
                                                                                existing['FuncVal'])
              )
        run_info = {'Existing': existing,
                    'Initial': initial,
                    'Final': final}
        return params, run_info

    def repair(self, zids, districts, district_ids):
        """
        Check if the spatial contiguity of a district if broken and apply repair operation
        """
        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        population, capacity = self.args[0], self.args[1]
        adjacency, polygons_nbr, schools, attributes = self.args[2], self.args[4], self.args[5], self.args[6]
        centers_list = {_ for _ in schools[attributes['Location']]}

        while len(zids) > 0:
            zid = zids[random.randrange(len(zids))]

            district = [m for m in districts[zid]['MEMBERS']]
            district_adj = adjacency.loc[district, district].values
            adj_mat = csr_matrix(district_adj)
            num_connect_comp, labels = connected_components(adj_mat, directed=False, return_labels=True)
            assert num_connect_comp > 1  # if it is not connected
            # print('{} \t Num disconnected components: {}'.format(zid, num_connect_comp))

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
                        nbr_ids = [district_ids[nbr] for nbr in polygons_nbr[area]
                                   if district_ids[nbr] not in zids]
                        if len(nbr_ids) > 0:
                            # Make the move
                            nbr_id = nbr_ids[random.randrange(len(nbr_ids))]
                            districts[zid]['MEMBERS'].remove(area)
                            districts[nbr_id]['MEMBERS'].append(area)
                            district_ids[area] = nbr_id

            zids.remove(zid)

        # Make sure that all the districts are connected and valid
        return self.check_solution(districts)

    def check_solution(self, districts, district_ids=None):
        """
        Checks if all the individual regions are geographically connected.
        """
        if district_ids is None:
            district_ids = districts.keys()

        # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes)
        population, capacity, adjacency = self.args[0], self.args[1], self.args[2]
        valid, disconnections = True, []

        for district_id in district_ids:
            members = [m for m in districts[district_id]['MEMBERS']]
            pop = sum(population[m] for m in members)
            cap = sum(capacity[m] for m in members)

            if pop <= 0 or cap <= 0:
                valid = False
            if not self.if_connected(adjacency, members):
                disconnections.append(district_id)

        return valid, disconnections

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

    @staticmethod
    def create_dir(*args):
        """
        Function to create directories and generate paths.
        """
        write_path = "./"
        for name in args:
            try:
                write_path = write_path + "{}/".format(name)
                os.mkdir(write_path)
            except Exception as e:
                print(".")

        return write_path

    @staticmethod
    def get_params(**argsv):
        # Returns a dictionary
        return argsv

    @staticmethod
    def get_args(*args):
        # Returns a tuple
        return args
