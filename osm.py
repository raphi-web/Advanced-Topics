import osmnx as ox
from osmnx import downloader
import geopandas as gpd
import matplotlib.pyplot as plt





def get_with_tags(place,params):
    geo_df = ox.geometries_from_place("Graz", params)
    return geo_df


def get_bus_stops(city):
    params = {"highway":["bus_stop"]}
    get_with_tags(city,params)

