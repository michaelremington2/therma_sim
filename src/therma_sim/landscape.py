#!/usr/bin/python
import networkx as nx
# https://gis.stackexchange.com/questions/295874/getting-polygon-breadth-in-shapely
# https://networkx.org/documentation/stable/auto_examples/geospatial/plot_polygons.html#sphx-glr-auto-examples-geospatial-plot-polygons-py
# could convert each square of the env raster to a polygon then convert to a network
# https://stackoverflow.com/questions/63876018/cut-a-shapely-polygon-into-n-equally-sized-polygons
class Landscape_From_Polygon(object):
	def __init__(self, landscape_polygon, **kwargs):
		self.Landscape_polygon = landscape_polygon



class Simulate_Landscape(object):
	'''
    Args:
    	width - Estimated width of a landscape in meters
    	length - Estimated length of a landscape in meters
    Attributes:
    '''
	def __init__(self):
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
    pass