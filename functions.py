import math
from shapely.ops import unary_union

w1 = 0.0
w2 = 0.0
max_iter = 0
epsilon = pow(10, 0)


def print_params():
    print(' w1 (Capacity) : {:.2}\n w2 (Compactness) : {:.2}'.format(w1, w2))
    print(' Epsilon : {}\n MaxIter : {}'.format(epsilon,
                                                max_iter
                                                )
          )


def set_params(w):
    global w1, w2, epsilon, max_iter
    w1 = 0.1 * w
    w2 = 1 - w1
    epsilon = pow(10, -5)  # global constant
    max_iter = 1000

    print_params()


def parameters():
    return w1, w2, epsilon, max_iter


def obj_func(ids, districts):
    """
    Sum of the F-values
    """

    return sum(districts[i]['F'] for i in ids)


def show_stat(args):
    # args = (population, capacity, adjacency, polygons, polygons_nbr, centers, attributes, districts, district_ids)
    districts = args[7]

    for district_id in districts.keys():

        # Get population and capacity statistics
        cap = districts[district_id]['Capacity']
        pop = districts[district_id]['Population']

        # Get perimeter and area statistics
        area = districts[district_id]['Area']
        peri = districts[district_id]['Perimeter']

        print("%s : Area: %10.5f Perimeter: %10.5f Capacity: %5d Population: %5d" % (district_id, area, peri, cap, pop))


def find_change(ids, area, args, districts=None):
    """Compute the change in objective function for adding 'area' to district 'district_id'."""
    # args = (population, capacity, adjacency, polygons, polygons_nbr, centers, attributes, districts, district_ids)
    population, capacity, polygons, attributes = args[0], args[1], args[3], args[6]
    if districts is None:
        districts = args[7]

    donor_id, recip_id = ids[0], ids[1]

    donor_members = [x for x in districts[donor_id]['MEMBERS']]
    recip_members = [x for x in districts[recip_id]['MEMBERS']]

    change, possible = None, False
    try:

        # Compute the change in functional value
        initial = sum([districts[i]['F'] for i in ids])

        donor_members.remove(area)
        recip_members.append(area)
        new_districts = [donor_members, recip_members]

        possible = True
        global w1, w2
        final = 0

        for s in new_districts:
            members = [m for m in s]
            _, _, f1 = target_balance(members, population, capacity)
            shape_list = [polygons['geometry'][index] for index, polygon in polygons.iterrows()
                          if polygon[attributes['Location']] in members]
            _, _, f2 = target_compact(shape_list)
            final += w1 * f1 + w2 * f2

            if not members:
                possible = False

        change = final - initial

    except Exception as e:
        print(e)

    return change, possible


def target_balance(members, pop, cap):
    """Balance of population with school capacity of the district"""
    p = c = None
    try:
        p = sum([pop[m] for m in members])
        c = sum([cap[m] for m in members])
        score = (p + 0.001) / (c + epsilon)

    except Exception as e:
        print('Exception raised in target_balance(): {}'.format(e))
        score = 0

    f1 = abs(1 - score)
    return p, c, f1


def target_compact(shape_list):
    """ Get perimeter, area and the target compactness of the district"""

    try:
        total = unary_union(shape_list)
        area = total.area
        peri = total.length
        score = (4*math.pi*area)/(peri**2)    # IPQ score or Polsby Popper score
        # score = peri**2/(4*math.pi*area)      # Schwartzberg's index
    except Exception as e:
        print('Exception raised in target_compactness(): {}'.format(e))
        area, peri, score = 0, 0, 0

    f2 = 1 - score
    return area, peri, f2


def computation(i, args, districts=None):
    # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)
    d = dict()

    if districts is None:
        districts = args[7]

    members = [m for m in districts[i]['MEMBERS']]

    '''Get population and capacity statistics'''
    pop, cap, f1 = target_balance(members=members, pop=args[0], cap=args[1])
    d['Capacity'] = cap
    d['Population'] = pop
    d['F1'] = f1

    '''Get area and perimeter statistics'''
    polygons, attributes = args[3], args[6]
    shape_list = [polygons['geometry'][index] for index, polygon in polygons.iterrows()
                  if polygon[attributes['Location']] in members]
    area, peri, f2 = target_compact(shape_list)
    d['Area'] = area
    d['Perimeter'] = peri
    d['F2'] = f2

    global w1, w2, epsilon
    try:
        F = w1 * f1 + w2 * f2
    except Exception as e:
        print('Error in computation(): {}'.format(e))
        F = 1

    d['F'] = F

    return i, d


def update_property(ids, args, districts=None):
    """
    Update the properties of districts (clusters) contained in district_id_list
    """
    # args = (population, capacity, adjacency, polygons, polygons_nbr, schools, attributes, districts, district_ids)

    if districts is None:
        districts = args[7]

    for i in ids:
        t, d = computation(i, args, districts)
        districts[i]['Capacity'] = d['Capacity']
        districts[i]['Population'] = d['Population']

        districts[i]['Area'] = d['Area']
        districts[i]['Perimeter'] = d['Perimeter']

        districts[i]['F1'] = d['F1']
        districts[i]['F2'] = d['F2']
        districts[i]['F'] = d['F']
