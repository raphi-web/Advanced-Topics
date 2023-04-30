import osmnx as ox
import numpy as np
import math

def get_with_tags(place, params):
    """
    gets OSM Attributes as Geodataframe
    :param place: String, name of place
    :param params: Dict, key-value pair of osm attributes
    :return: Geodataframe
    """
    geo_df = ox.geometries_from_place(place, params)
    return geo_df


def get_bus_stops(city):
    """
    Gets the Busstops of a city from osm
    :param city: String, name of city
    :return: Geodataframe
    """
    params = {"highway": ["bus_stop"]}
    return get_with_tags(city, params)


def get_buildings(city):
    """
    gets all buildings of a city from osm
    :param city: String, name of city
    :return: Geodataframe
    """
    params = {"building": True}
    return get_with_tags(city, params)


def get_network_austria():
    """
    Gets the routable drivingnetwork of austria
    :return: Networx.Graph and Ox.Graphproject
    """
    G = ox.graph_from_place("Austria", network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    Gp = ox.project_graph(G)
    return G, Gp


def orientation_analysis(G):
    """
    Performs the orientation analysis for a city.
    !!! I could not calculate the weights properly so I left them out!
    """
    GS = ox.simplify_graph(G)
    G = ox.add_edge_bearings(G,2)
    GS = ox.add_edge_bearings(GS,2)
    bearings = []
    for e in GS.edges:
        attributes = GS.get_edge_data(*e)
        if "bearing" in attributes:
            bear = attributes["bearing"]
            if bear == 90:
                bear += 270

            bearings.append(bear)

    weighted_bearings = []
    lengths = []
    for e in G.edges:
        attributes = G.get_edge_data(*e)
        if "bearing" in attributes:
            bear  = attributes["bearing"]
            if bear == 90:
                bear += 270
        
            weighted_bearings.append(bear)
            lengths.append(attributes["length"])


    #paper does not specify how the weights are calculated
    # I tried a lot but nothing worked
    #length_weights = sigmoidScaling(lengths,k=1,x0=-2)
    #max_length = max(lengths)
    #min_length = min(lengths)
    #length_weights = [l/max_length for l in lengths]
    #length_weights = [l/(max_length - min_length) for l in lengths]
    #length_weights = lengths
    #weighted_bearings = [b * w for b,w in zip(weighted_bearings,length_weights)]
    
    bins_no_weight = __center_sort_to_bins(bearings)
    bins_with_weight = __center_sort_to_bins(weighted_bearings)

    n_bearing = len(bearings)
    n_weighted = len(weighted_bearings)

    def shannon(n, m): return n/m * np.log(n/m)
    h0 = -np.sum([shannon(len(b), n_bearing) for b in bins_no_weight if len(b) != 0])
    hw = -np.sum([shannon(len(b), n_weighted) for b in bins_with_weight if len(b) != 0])

    hmax = np.log(36)
    hg = 1.386

    def calc_phi(h): return 1 - ((h - hg)/(hmax - hg))**2
    phi = calc_phi(h0)
    phi_w = calc_phi(hw)

    return (phi, phi_w, h0, hw, bins_no_weight,bins_with_weight)


def __center_sort_to_bins(input_values, min_val=0, max_val=360, step=10):
    copy_input_values = input_values.copy()
    bin_values = np.arange(min_val, max_val, step)
    bins = []
    for bin in bin_values:
        bin_list = []
        to_remove = []
        for i, val in enumerate(copy_input_values):
            delta = abs(bin - val)
            if delta < step/2:
                bin_list.append(val)
                to_remove.append(i)

        for idx in sorted(to_remove, reverse=True):
            del copy_input_values[idx]

        bins.append(bin_list)
    return bins

import math 

def sigmoidScaling(arr,k=1,x0=0):
    def genSigmoid(L,k,x0):
        return lambda x: L/(1+math.e**(-k * (x -x0)))
    
    def genScaler(new_min,new_max, min_value, max_value):
        a = (new_max - new_min)
        b = max_value - min_value
        return lambda value: (a * ((value - min_value)/b)) + new_min

    sigmoidF = genSigmoid(1,k=k,x0=x0)
    k = np.mean(arr)
    new_min = -6
    new_max = 6 + x0
    min_value = min(arr)
    max_value = max(arr)
    
    scale_f = genScaler(new_min,new_max,min_value,max_value)
    scaled_values = [scale_f(x) for x in arr]
    sigmoid_mapped = [sigmoidF(x) for x in scaled_values]
    return sigmoid_mapped

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bus_stops = get_bus_stops("Graz")
    buildings = get_buildings("Graz")
    x = np.arange(-6,6,0.1)
    y = sigmoidScaling(x,1,0)
    plt.plot(x,y)
    plt.ylim(0,1)
    plt.show()
