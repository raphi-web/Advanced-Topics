import osmnx as ox



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


if __name__ == "__main__":
    bus_stops = get_bus_stops("Graz")
    buildings = get_buildings("Graz")
