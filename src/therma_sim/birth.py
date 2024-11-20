#!/usr/bin/python


class birth_module(object):
    def __init__(self, agent_type, frequency: str, litter_size: float, partuition_months: int):
        self.frequency = frequency
        self.agent_type = agent_type
        self.litter_size = litter_size
        self.partuition_months = partuition_months