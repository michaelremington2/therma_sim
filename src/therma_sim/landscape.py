#!/usr/bin/python
import networkx as nx

class Landscape(object):
    '''
    Args:
    	width - Estimated width of a landscape in meters
    	length - Estimated length of a landscape in meters
    Attributes:
    '''
	def __init__(self, width, length):
		self.global_landscape = self.generate_global_landscape_network_object()

	def generate_global_landscape_network_object(self, directed_graph=False):
		if directed_graph:
			global_landscape = nx.DiGraph()
		else:
			global_landscape = nx.Graph()
		return global_landscape

if __name__ ==  "__main__":
    pass