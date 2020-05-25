################################################################################
## Imports

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from random import random
from pyeasyga import pyeasyga

################################################################################
## Common code

# Data
data = [0] * 5

# Define fitness function (invftrap5)
def fitness_function(individual, data=None):
  u = sum(individual)
  if u > 0:
    return u - 1
  return 5

# Define the key field for sorting
def get_key(obj):
  return obj.fitness

################################################################################
## Simple Genetic Algorithm (sGA)

# Initialize genetic algorithm
sga = pyeasyga.GeneticAlgorithm(data, population_size=40, generations=100)

# Set fitness function
sga.fitness_function = fitness_function

# Covariance matrices
sga_cm = {}

# Set evolution function
def run(self, sga_cm):
  self.create_first_generation()
  # Initial Covariance Matrix
  arrs = [numpy.transpose(i.genes) for i in self.current_generation]
  sga_cm['icm'] = numpy.cov(arrs)
  for i in range(1, self.generations):
    self.create_next_generation()
    # Intermediary
    if i == int(self.generations / 2):
      # Intermediary Covariance Matrix
      arrs = [numpy.transpose(i.genes) for i in self.current_generation]
      sga_cm['tcm'] = numpy.cov(arrs)
  # Final Covariance Matrix
  arrs = [numpy.transpose(i.genes) for i in self.current_generation]
  sga_cm['fcm'] = numpy.cov(arrs)
sga.run = run

# Run sGA
sga.run(sga, sga_cm)
# Get best individual
result = sga.best_individual()
# Print result
print('The sGA best solution is: {}'.format(result))

################################################################################
## Compact Genetic Algorithm (cGA)

# Initialize genetic algorithm
cga = pyeasyga.GeneticAlgorithm(data, population_size=40, generations=100)

# Set fitness function
cga.fitness_function = fitness_function

# Covariance matrices
cga_cm = {}

# Update probability vector
def update_prob(winner, loser, prob, popsize):
  for i in range(0, len(prob)):
    if winner[i] != loser[i]:
      if winner[i] == 1:
        prob[i] += 1.0 / float(popsize)
      else:
        prob[i] -= 1.0 / float(popsize)

# Create a new individual
def create_individual(prob):
  individual = []
  for p in prob:
    if random() < p:
      individual.append(1)
    else:
      individual.append(0)
  return pyeasyga.Chromosome(individual)
cga.create_individual = create_individual

# Make competition between two individuals
def compete(a, b):
  if a.fitness > b.fitness:
    return a, b
  else:
    return b, a

# Set evolution function
def run(self):
  # Initialize probability vector
  prob = [0.5] * len(self.seed_data)
  # Initialize best solution
  best = None
  # Population
  population = []
  # Initial population
  arrs = []
  for _ in range(self.population_size):
    individual = self.create_individual(prob)
    population.append(individual)
    arrs.append(numpy.transpose(individual.genes))
  cga_cm['icm'] = numpy.cov(arrs)
  # Run `i` generations
  for i in range(0, self.generations):
    # Create individuals
    a = self.create_individual(prob)
    b = self.create_individual(prob)
    population.append(a)
    population.append(b)
    # Calculate fitness for each individual
    a.fitness = self.fitness_function(a.genes)
    b.fitness = self.fitness_function(b.genes)
    # Get the best and worst individual
    winner, loser = compete(a, b)
    # Update best solution
    if best:
      if winner.fitness > best.fitness:
        best = winner
    else:
      best = winner
    # Intermediary
    # Update best individuals population
    population.sort(key=get_key, reverse=True)
    population = population[:self.population_size]
    if i == int(self.generations / 2):
      # Intermediary Covariance Matrix
      arrs = [numpy.transpose(i.genes) for i in population]
      cga_cm['tcm'] = numpy.cov(arrs)
    # Update the probability vector based on the success of each bit
    update_prob(winner.genes, loser.genes, prob, self.population_size)
  # Add final solution
  self.current_generation.append(best)
  # Update best individuals population
  population.sort(key=get_key, reverse=True)
  population = population[:self.population_size]
  # Final Covariance Matrix
  arrs = [numpy.transpose(i.genes) for i in population]
  cga_cm['fcm'] = numpy.cov(arrs)
cga.run = run

# Run evolution
cga.run(cga)
# Get best individual
result = cga.best_individual()
# Print result
print('The cGA best solution is: {}'.format(result))
