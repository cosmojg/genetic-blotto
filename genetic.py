# Game Description: In a distant, war-torn land, there are 10 castles. There are two
# warlords: you and your archenemy. Each castle has its own strategic value for a would-
# be conqueror. Specifically, the castles are worth 1, 2, 3, … , 9 and 10 victory
# points. You and your enemy each have 100 soldiers to distribute, any way you like, to
# fight at any of the 10 castles. Whoever sends more soldiers to a given castle conquers
# that castle and wins its victory points. If you each send the same number of troops,
# you split the points. You don’t know what distribution of forces your enemy has chosen
# until the battles begin. Whoever wins the most points wins the war.

# Objective: Submit a plan distributing your 100 soldiers among the 10 castles. Once I
# receive all your battle plans, I will adjudicate all the possible one-on-one matchups.
# A victory will be worth one “victory point,” while a tie will be worth 0.5 victory
# points. After all the one-on-one matchups are complete, whoever has accumulated the
# fewest victory points will be eliminated from the tournament, after which the battle
# will recommence with one fewer competitor. If two warlords are tied with the fewest
# number of victory points, the first tiebreaker will be whoever has more wins (and
# fewer ties) and the second tiebreaker will be performance in the preceding round (and
# then the round before that, etc.). If two or more strategies on the chopping block are
# precisely the same, I will randomly pick which one to eliminate.

# Source: https://fivethirtyeight.com/features/the-final-battle-for-riddler-nation/

import numpy as np
import polars as pl
from tqdm import tqdm
from typing import Any, Tuple

rng = np.random.default_rng()

# Parameters
POPULATION_SIZE = 100
P_CROSSOVER = 0.8
P_MUTATION = 0.05
MAX_GENERATIONS = 500
points = np.arange(1.,11.,1.)

def fitness_function(sample: np.ndarray[Any, Any], individual: np.ndarray[Any, Any]) -> float:
    score = float(0.0)
    for row in sample:
        comparison = individual - row
        comparison[comparison > 0] = 1
        comparison[comparison == 0] = 0.5
        comparison[comparison < 0] = 0
        total_points = (comparison * points).sum()
        if total_points > 27.5:
            score += 1
        elif total_points == 27.5:
            score += 0.5

    return score / float(sample.shape[0])

def sum_correction(individual: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    while np.sum(individual) < 100:
        individual[rng.integers(10)] += 1
    while np.sum(individual) > 100:
        castle = rng.integers(10)
        if individual[castle] > 0:
            individual[castle] -= 1

    return individual

def create_initial_population(popsize: int) -> np.ndarray[Any, Any]:
    population = np.ones((popsize, 10))
    for i in range(popsize):
        individual = np.rint(rng.dirichlet(np.ones(10)/(rng.random()*10))*100)
        if individual.sum() != 100:
            individual = sum_correction(individual)
        assert(individual.sum() == 100)
        rng.shuffle(individual) # This is probably unnecessary
        population[i] = individual

    return population

def roulette_wheel(population: np.ndarray[Any, Any], fitness_scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    total_fit = sum(fitness_scores)
    if total_fit == 0:
        return np.array(rng.choice(population))
    probabilities = [fit / total_fit for fit in fitness_scores]

    return np.array(rng.choice(population, p=probabilities))

def uniform_crossover(parent1: np.ndarray[Any, Any], parent2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    mask = rng.random(10) < P_CROSSOVER
    offspring = parent1.copy()
    offspring[mask] = parent2[mask]

    return offspring

def mutation(individual: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    for i in range(individual.shape[0]):
        if rng.random() < P_MUTATION:
            diff = rng.integers(-5, 6)
            if abs(diff) < individual[i] < (100 - abs(diff)):
                individual[i] += diff
                unbalanced = True
                for j in range(individual.shape[0]):
                    if j > 0 and abs(diff) < individual[(i+j) % 10] < (100 - abs(diff)) and unbalanced:
                        individual[(i+j) % 10] -= diff
                        unbalanced = False

    return individual

def main(sample: np.ndarray[Any, Any], popsize: int, maxgen: int) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    population = create_initial_population(popsize)
    fitness_scores = np.array([fitness_function(sample, ind) for ind in population])

    for generation in tqdm(range(maxgen)):
        new_population = np.ones(population.shape)

        for i in range(new_population.shape[0]):
            # Selection
            parent1 = roulette_wheel(population, fitness_scores)
            parent2 = roulette_wheel(population, fitness_scores)

            # Crossover
            offspring = uniform_crossover(parent1, parent2)

            # Mutation
            individual = mutation(offspring)

            if individual.sum() != 100:
                individual = sum_correction(individual)
            assert(individual.sum() == 100)
            
            new_population[i] = individual

        population = new_population
        fitness_scores = np.array([fitness_function(sample, ind) for ind in population])

    return population, fitness_scores

if __name__ == "__main__":
    popsize = POPULATION_SIZE
    maxgen = MAX_GENERATIONS
    # Run genetic algorithm on sample data
    sample = pl.concat([pl.read_ipc("castle-solutions-1.arrow"),
                             pl.read_ipc("castle-solutions-2.arrow"),
                             pl.read_ipc("castle-solutions-3.arrow"),
                             pl.read_ipc("castle-solutions-4.arrow"),
                             pl.read_ipc("castle-solutions-5.arrow")]).to_numpy()
    print(main(sample, popsize, maxgen))
