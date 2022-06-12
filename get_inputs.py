import os
import json
import copy
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, Rook


class GetInputs:
    """
    Helps in reading the geospatial data files and returning the processed data.
    """
    # sy = "SY2019_20"    # school year of the corresponding datasets
    
    def __init__(self, district='lcps', level='ES', p=2):
        self.read_path = './{}/data/'.format(district)
        self.district = district
        self.level = level
        self.p = p

    def get_schools(self, polygons, attributes):
        """ Read the geospatial data corresponding to centers and filter them based on the school type. """
        schools = gpd.read_file(self.read_path + 'Schools.geojson')
        level = {'ES': 'ELEMENTARY',
                 'MS': 'MIDDLE',
                 'HS': 'HIGH'
        }
        schools = schools[schools['CLASS'] == level[self.level]]
        print('Number of {} schools: {}'.format(level[self.level], len(schools)))
        schools.index = [i for i in range(len(schools))]
        '''
        school_polys = set(schools[attributes['Location']])
        polys = set(polygons[attributes['Location']])
        remove_polys = polys - school_polys

        to_remove = [i for i, s in polygons.iterrows() if s[attributes['Location']] in remove_polys]
        centers = polygons.drop(to_remove, axis=0)
        '''
        return schools

    def get_attr(self):
        """
        Get the attribute corresponding to the district
        """
        level = {'ES': 'ELEM',
                 'MS': 'MID',
                 'HS': 'HIGH'
        }
        if self.district == 'lcps':
            attr = {
                'Location': 'STDYAREA',    # Polygon identifier attribute
                'Level': level[self.level],    # Polygon population attribute
                'Capacity': 'CAPACITY',    # Capacity of district attribute
                'district': 'lcps',
                'weight': 7
            }

        elif self.district == 'fcps':
            attr = {
                'Location': 'SPA',    # Polygon identifier attribute
                'Level': level[self.level],    # Polygon population attribute
                'Capacity': 'CAPACITY',    # Capacity of district attribute
                'district': 'fcps',
                'weight': 8
                }

        else:
            attr = None

        return attr

    @staticmethod
    def get_adj_list(polygons_nbrlist):
        """Returns adjacency relations as a data frame"""
        polygons_list = [x for x in polygons_nbrlist.keys()]
        adjacency = pd.DataFrame(0, index=polygons_list, columns=polygons_list)

        for spa in polygons_list:
            for nbr in polygons_nbrlist[str(spa)]:
                adjacency.at[spa, nbr] = 1

        return adjacency

    @staticmethod
    def get_pop_cap(polygons, schools, attributes):
        """
        Get attending student population and capacity for centers in a school district.
        """
        population, capacity = {}, {}

        try:
            for index, polygon in polygons.iterrows():
                location = polygon[attributes['Location']]
                population[location] = polygon['{}_POP'.format(attributes['Level'])]
                capacity[location] = 0

            for index, school in schools.iterrows():
                location = school[attributes['Location']]
                capacity[location] = school[attributes['Capacity']]
        except Exception as e:
            print(e)

        return population, capacity

    def get_inputs(self):
        """ Read geo-spatial data corresponding to school districts """
        # Read the data: centers and Student Planning Areas (polygons)
        polygons = gpd.read_file(self.read_path + "Polygons.geojson")
        attributes = self.get_attr()
        schools = self.get_schools(polygons, attributes)

        # Get the adjacency list corresponding to the polygons
        rW_polygons = Rook.from_dataframe(polygons)
        polygons_nbrlist = dict()

        for i in range(len(polygons)):
            location = polygons[attributes['Location']][i]
            polygons_nbrlist[location] = [polygons[attributes['Location']][k]
                                          for k, v in rW_polygons[i].items()]

        # Read the population and capacity info the centers
        population, capacity = self.get_pop_cap(polygons, schools, attributes)
        # Get the adjacency matrix of the underlying graph structure of the polygons
        adjacency = self.get_adj_list(polygons_nbrlist)

        return population, capacity, adjacency, polygons, polygons_nbrlist, schools, attributes

    @staticmethod
    def read_json_file(file_loc):
        """ Reads a json file given the file location """
        with open(file_loc) as inpfile:
            data = json.load(inpfile)
            inpfile.close()

        return data
