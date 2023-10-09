#!/usr/bin/python
import networkx as nx
import math
import fiona
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import geopandas as gpd
from shapely import affinity
import get_thermal_data as gtd
#from scipy.spatial import Voronoi
# https://gis.stackexchange.com/questions/295874/getting-polygon-breadth-in-shapely
# https://networkx.org/documentation/stable/auto_examples/geospatial/plot_polygons.html#sphx-glr-auto-examples-geospatial-plot-polygons-py
# could convert each square of the env raster to a polygon then convert to a network
# https://stackoverflow.com/questions/63876018/cut-a-shapely-polygon-into-n-equally-sized-polygons
# https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations sample points form a raster
# https://networkx.org/documentation/stable/tutorial.html

##################
## TO DO:
## 1) Stats describing number of points, distance between points etc
## 2) connect thermal data to point data
## 3) Build landscape network!
## 40 build out simulate landscape class

class Landscape_Attributes_From_Polygon(object):
    '''
    Args:
        - Landscape polygon should be a .shp file
    '''
    def __init__(self, landscape_polygon_fp):
        self.Landscape_polygon = self._get_shape_w_fiona(raster_fp = landscape_polygon_fp)
        self.ls_df = self._get_shape_w_geopandas(landscape_polygon_fp)
        self.points = self.get_points(num = 40, smaller_versions = 40)

    def _get_shape_w_fiona(self, raster_fp):
        '''Enter the shape file file path to be read with fiona'''
        with fiona.open(raster_fp, "r") as shapefile:
            landscape_shp = [shape(feature["geometry"]) for feature in shapefile]
        return landscape_shp

    def _get_shape_w_geopandas(self, shp_fp):
        return gpd.read_file(shp_fp)

    def _shape_to_points(self, shape, num = 10, smaller_versions = 10):
        points = []

        for shrink_factor in range(0, smaller_versions + 1):
            shrink_factor = smaller_versions - shrink_factor
            shrink_factor = shrink_factor / float(smaller_versions)

            smaller_shape = affinity.scale(shape, shrink_factor, shrink_factor)

            # Use the boundary of the shape for interpolation
            boundary = smaller_shape.boundary

            for i in range(num):
                distance_along_perimeter = i / float(num)
                point = boundary.interpolate(distance_along_perimeter, normalized=False)
                x, y = point.xy[0][0], point.xy[1][0]
                points.append(geom.Point([x, y]))

        # Add the centroid of the smallest shape  
        centroid = smaller_shape.centroid
        points.append(geom.Point([centroid.x, centroid.y]))

        #points = np.array(points)
        return points

    def get_points(self, num = 10, smaller_versions = 10):
        total_points = []
        for index, row in self.ls_df.iterrows():
                #shape = affinity.translate(row['geometry'], -row['geometry'].bounds[0], -row['geometry'].bounds[1])
                points = self._shape_to_points(row['geometry'], num=num, smaller_versions=10)
                total_points.append(points)
        return total_points

    def _load_thermal_data(self, **env_file_paths):
        env_data_shp = gtd.GetThermalData()
        for env_data_type, env_fp in env_file_paths:
            output_therma_raster, meta_data = env_data_shp.extract_environmental_data_shape(env_data_tiff_fp = '/home/mremington/Documents/therma_sim/Raster-Layers/bioclim_data/wc2.1_30s_bio_1.tif',
                                                                                            landscape_shp = ls_shp)

    def plot_points(self):
        ls.ls_df['geometry'].plot()
        xs = [point.x for point in self.points[0]]
        ys = [point.y for point in self.points[0]]
        plt.scatter(xs, ys,color='r')
        plt.show()

    def landscape_stats(self):
        return len(points)


class Simulate_Landscape(object):
    '''
    Args:
        width - Estimated width of a landscape in meters
        length - Estimated length of a landscape in meters
        size_of_node - Hoe large the sub territories are
    Attributes:
    '''
    def __init__(self, height, width, centroid=(0,0), size_of_node=(25, 25)):
        self.centroid = geom.Point(centroid)
        self.height = height
        self.width = width
        self.size_of_node = size_of_node
        self._earths_radius = 6371000.0 # Radius of the Earth in meters

    def generate_global_landscape_network_object(self, directed_graph=False):
        if directed_graph:
            global_landscape = nx.DiGraph()
        else:
            global_landscape = nx.Graph()
        return global_landscape

    def calculate_new_lat_lng(initial_lat, initial_lng, distance_meters, bearing_radians):
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(initial_lat)
        lng1 = math.radians(initial_lng)
        # Calculate new latitude
        lat2 = math.asin(math.sin(lat1) * math.cos(distance_meters / self._earths_radius) +
                         math.cos(lat1) * math.sin(distance_meters / self._earths_radius) * math.cos(bearing_radians))
        # Calculate new longitude
        lng2 = lng1 + math.atan2(math.sin(bearing_radians) * math.sin(distance_meters / self._earths_radius) * math.cos(lat1),
                                 math.cos(distance_meters / self._earths_radius) - math.sin(lat1) * math.sin(lat2))
        # Convert new latitude and longitude from radians to degrees
        new_lat = math.degrees(lat2)
        new_lng = math.degrees(lng2)

        return new_lat, new_lng

    def simulate_landscape_topography(self):
        lat_max = self.calculate_new_lat_lng(initial_lat = self.centroid.y, initial_lng = self.centroid.x,
                                             distance_meters = self.width/2, bearing_radians=0) #0 for west
        lat_min = self.calculate_new_lat_lng(initial_lat = self.centroid.y, initial_lng = self.centroid.x,
                                             distance_meters = self.width/2, bearing_radians=math.pi) #pi for east
        lng_max = self.calculate_new_lat_lng(initial_lat = self.centroid.y, initial_lng = self.centroid.x,
                                             distance_meters = self.length/2, bearing_radians=math.pi/2) #pi/2 for north
        lng_min = self.calculate_new_lat_lng(initial_lat = self.centroid.y, initial_lng = self.centroid.x,
                                             distance_meters = self.length/2, bearing_radians=(3*math.pi)/2) #3pi/2 for south



if __name__ ==  "__main__":
    landscape_polygon_fp = '/home/mremington/Documents/therma_sim/Raster-Layers/landscape-polygons/canada/Canada.shp'
    ls = Landscape_Attributes_From_Polygon(landscape_polygon_fp)
    #print(ls.ls_df.head())
    #print(ls.Landscape_polygon)
    #ls.plot_points()
    #print(ls.points)
    #plt.show()
    print(ls.landscape_stats)
