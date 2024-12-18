# Krat Movement

### Ecology

- Nocternal
- Kangaroo rats are patch specialists
- 2 movement types: Beeline and short, meandering 
- During a foraging bout, they initially visit four or five patches be-
fore returning to the burrow and depositing collected seeds;
then they exit again to visit another four or five patches be-
fore returning once more (Thompson, personal communi-
cation to C.M.H., June 24, 1992), a multiple patch form of
central-place foraging.
- visit a set of patches for a period
of several months, supporting the importance of memory
for spatial locations.
- The small (ca. 35 g) Merriam's
kangaroo rat (D. merriami) tolerates broad overlap (or at least interdigita-
tion) of its large home ranges, changes day burrows frequently, and scatters
the food it collects in numerous caches, often many meters apart (Behrends
et al. 1986a, b; Daly et al. 1992b).
- Core home ranges of radio-collared kangaroo rats average 1750 ± 620 m2 (± 1 standard error, n = 28; Gummer and Robertson 2003c). 
- However, Ord’s kangaroo rats occasionally move beyond this range at night, with overall home range size averaging 7830 ± 2930 m2 (n = 38). 
- The average maximum home range width is 130 ± 35 m (n = 38).

### Algorithm
  
- Random walk in patch
  - Within the hour, simulate a trajectory of within patch movement
- beeline between patches (shortist euclidean distance)
  - Each hour kangaroo rats can switch patches at random
- each agent is assigned 4~5 patches in memory within their home range and switch between them through the night
- open vs burrow states = active vs inactive
- We might not need to explicitely model seeds or metabolism of a kangaroo rat, but just model the movement patterns.
  - Simulate a trajectory every hour


### Papers

1)  How does the ecological foraging behavior of desert kangaroo rats (Dipodomys deserti) relate to their behavior on radial mazes?
   - https://link.springer.com/content/pdf/10.3758/BF03195959.pdf
2) Activity Patterns of Kangaroo Rats - Granivores in a Desert Habitat
   - https://link.springer.com/chapter/10.1007/978-3-642-18264-8_10
3) Canada official Dipodomys ordii report
   - https://www.canada.ca/en/environment-climate-change/services/species-risk-public-registry/cosewic-assessments-status-reports/ord-kangaroo-rat/chapter-8.html#bio_4