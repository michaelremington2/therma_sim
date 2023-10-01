#!/usr/bin/python
import networkx as nx
import fiona
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, shape, Point
import geopandas as gpd
from shapely import affinity
from scipy.spatial import Voronoi
# https://gis.stackexchange.com/questions/295874/getting-polygon-breadth-in-shapely
# https://networkx.org/documentation/stable/auto_examples/geospatial/plot_polygons.html#sphx-glr-auto-examples-geospatial-plot-polygons-py
# could convert each square of the env raster to a polygon then convert to a network
# https://stackoverflow.com/questions/63876018/cut-a-shapely-polygon-into-n-equally-sized-polygons
# https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations sample points form a raster


class Landscape_From_Polygon(object):
    '''
    Args:
        - Landscape polygon should be a .shp file
    '''
    def __init__(self, landscape_polygon_fp, **kwargs):
        self.Landscape_polygon = self._get_shape_w_fiona(raster_fp = landscape_polygon_fp)
        self.ls_df = self._get_shape_w_geopandas(landscape_polygon_fp)
        self.points = self._get_points(num = 40, smaller_versions = 40)

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
                points.append(Point([x, y]))

        # Add the centroid of the smallest shape
        centroid = smaller_shape.centroid
        points.append(Point([centroid.x, centroid.y]))

        #points = np.array(points)
        return points

    def _get_points(self, num = 10, smaller_versions = 10):
        total_points = []
        for index, row in self.ls_df.iterrows():
                #shape = affinity.translate(row['geometry'], -row['geometry'].bounds[0], -row['geometry'].bounds[1])
                points = self._shape_to_points(row['geometry'], num=num, smaller_versions=10)
                total_points.append(points)
        return total_points

    def plot_points(self):
        ls.ls_df['geometry'].plot()
        xs = [point.x for point in self.points[0]]
        ys = [point.y for point in self.points[0]]
        plt.scatter(xs, ys,color='r')
        plt.show()


class Simulate_Landscape(object):
    '''
    Args:
        width - Estimated width of a landscape in meters
        length - Estimated length of a landscape in meters
    Attributes:
    '''
    def __init__(self, centroid, height, width):
        self.global_landscape = self.generate_global_landscape_network_object()

    def generate_global_landscape_network_object(self, directed_graph=False):
        if directed_graph:
            global_landscape = nx.DiGraph()
        else:
            global_landscape = nx.Graph()
        return global_landscape

    def simulate_landscape_topography(self):
        pass



if __name__ ==  "__main__":
    landscape_polygon_fp = '/home/mremington/Documents/therma_sim/Raster-Layers/landscape-polygons/canada/Canada.shp'
    ls = Landscape_From_Polygon(landscape_polygon_fp)
    #print(ls.ls_df.head())
    #print(ls.Landscape_polygon)
    ls.plot_points()
    #print(ls.points)
    plt.show()
