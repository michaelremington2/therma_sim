#!/usr/bin/python
import numpy as np
from scipy.special import softmax

class SoftmaxLookupTable:
    def __init__(self, min_utility=0, max_utility=1, step=0.05, temperature=1.0, decimal_places=2):
        """
        Initializes the Softmax lookup table for behavioral preferences.

        Parameters:
        - min_utility (float): Minimum utility value.
        - max_utility (float): Maximum utility value.
        - step (float): Granularity of the lookup table.
        - temperature (float): Controls exploration vs. exploitation.
        - decimal_places (int): Number of decimal places for rounding.
        """
        self.min_utility = min_utility
        self.max_utility = max_utility
        self.step = step
        self.temperature = temperature
        self.decimal_places = decimal_places
        self.utilities = np.arange(min_utility, max_utility + step, step)
        self.lookup_table = self._build_lookup_table()

    def round_to_step(self, value):
        """Rounds a value to the nearest multiple of the step size with fixed decimal places."""
        return round(round(value / self.step) * self.step, self.decimal_places)

    def mask_zero_utilities(self, utilities):
        """Masks 0 utility values by replacing them with -inf to prevent selection."""
        return np.where(utilities == 0, -np.inf, utilities)  # Ensures zero-utility behaviors are never chosen

    def _build_lookup_table(self):
        """Creates a lookup table mapping utility tuples to softmax probabilities."""
        table = {}
        for u1 in self.utilities:
            for u2 in self.utilities:
                for u3 in self.utilities:
                    key = (self.round_to_step(u1), self.round_to_step(u2), self.round_to_step(u3))
                    utilities = np.array([u1, u2, u3])
                    masked_utilities = self.mask_zero_utilities(utilities)  # Apply masking
                    probabilities = softmax(masked_utilities / self.temperature)  # Apply temperature scaling
                    table[key] = probabilities
        return table

    def get_probabilities(self, utilities):
        """
        Retrieves softmax probabilities from the lookup table.

        Parameters:
        - utilities (tuple): A tuple of three utilities (rest, thermoregulate, forage).

        Returns:
        - np.array: Softmax probabilities corresponding to the given utilities.
        """
        key = tuple(self.round_to_step(u) for u in utilities)  # Ensure key aligns with lookup table
        masked_utilities = self.mask_zero_utilities(np.array(utilities))  # Mask before softmax if not found
        return self.lookup_table.get(key, softmax(masked_utilities / self.temperature))  # Fallback if not found
