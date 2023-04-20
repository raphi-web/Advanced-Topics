import osmnx as ox
from osmnx import downloader
import geopandas as gpd
import matplotlib.pyplot as plt


def get_with_tags(place, params):
    geo_df = ox.geometries_from_place("Graz", params)
    return geo_df


def get_bus_stops(city):
    params = {"highway": ["bus_stop"]}
    return get_with_tags(city, params)


def get_buildings(city):
    params = {"building": True}
    return get_with_tags(city, params)


if __name__ == "__main__":
    bus_stops = get_bus_stops("Graz")
    buildings = get_buildings("Graz")
