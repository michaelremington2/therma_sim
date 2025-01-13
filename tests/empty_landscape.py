from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
import numpy as np
import pandas as pd



class ContinuousLandscapeModel(Model):
    """A model with agents on a continuous landscape."""
    def __init__(self, width, height):
        self.landscape = ContinuousSpace(x_max=width, y_max=height, torus=False)
        self.landscape.thermal_profile = pd.read_csv("Data/thermal_db.csv", header=0)
        self.landscape.microhabitats = ['Burrow', 'Open']
        self.step_id = 1
        self.month=9
        self.schedule = RandomActivation(self)
        

    def add_agents(self, agent_list: list):
        # Create agents
        for agent in agent_list:
            self.schedule.add(agent)
            self.landscape.place_agent(agent, agent.pos)

    def remove_agents(self, agent_list: list):
        for agent in agent_list:
            self.schedule.remove(agent)
    
    def step(self):
        """Advance the model by one step."""
        self.schedule.step()


if __name__ == "__main__":
    pass