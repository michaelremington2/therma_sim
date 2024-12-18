# Interaction Module

### Ecology



### Algorithm

Key Components

    Agents
        Predator (e.g., snake): A sit-and-wait predator that relies on thermal performance and spatial proximity to hunt.
        Prey (e.g., kangaroo rat): An active agent that moves through the landscape, seeking resources while avoiding predators.

    Environment
        Landscape: Represented as a spatial grid where agents interact within defined distances (meters).
        Thermal Context: Body temperature influences predator strike performance.

    Key Parameters
        Interaction Distance: Maximum distance within which predator-prey interactions can occur.
        Caloric Gain: Energy obtained by the predator from consuming prey.
        Digestion Efficiency: Proportion of prey calories metabolized by the predator.

Core Processes

    Interaction Checking
        Predators assess spatial proximity to prey.
        Only active predators and prey are considered for interactions.

    Strike Decision
        Predators attempt strikes based on:
            Spatial proximity: Prey must be within a critical interaction distance.
            Thermal performance: Probability of a successful strike is influenced by the predator's body temperature.

    Energy Dynamics
        Predator: Gains energy from successful strikes, accounting for prey mass and digestion efficiency.
        Prey: Dies if struck successfully, removing it from the landscape.

    Thermal Performance
        Predator strike success depends on body temperature and thermal performance limits (e.g., minimum, maximum, and optimal temperatures).
        Performance peaks at optimal temperature and declines beyond critical thresholds.

Model Goals

    Behavioral Dynamics: Simulate predator-prey interactions in realistic environmental and physiological contexts.
    Thermal Constraints: Explore how predator performance varies with temperature.
    Energy Budgets: Track predator energy gains and losses to assess survival and fitness outcomes.