#!/usr/bin/python

class Organism(object):
	# Metabolism
	# Thermal Conditions
	def __init__(self, sim_object, species_label=''):
		self.sim = sim_object

	def __hash__(self):
        return id(self)
