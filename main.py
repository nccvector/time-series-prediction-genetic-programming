import numpy as np 
import argparse
import math

import random
import operator
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import networkx as nx
import matplotlib.pyplot as plt


# Making parser object
parser = argparse.ArgumentParser(description='Process some integers.')

# Adding arguments
parser.add_argument('-population', type=int, help='population size')
parser.add_argument('-iterations', type=int, help='number of iterations to perform')
parser.add_argument('-data', type=str, help='data file path')

# Getting args
args = parser.parse_args()

# Parsing the input vector
dims = 1
data = np.genfromtxt(args.data, delimiter=',')

population_size = args.population
iterations = args.iterations

# Seperating input vector and output vector
in_vector = data[:,:dims]
y_vector = data[:,-1]

# Mean square of the result with the y_vector
def mean_square_error(res):
    
    global y_vector

    return np.sum((y_vector - res)**2) / res.shape[0]


# Genetic programming code
def Divide(left, right):
    if right == 0:
        return 1

    return left / right


# Creating an initial symbolic tree and provinding the selected operators
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(Divide, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-10,10))
pset.addEphemeralConstant("rand102", lambda: random.randint(-10,10))
pset.addEphemeralConstant("rand103", lambda: random.randint(-10,10))
pset.addEphemeralConstant("rand104", lambda: random.randint(-10,10))
pset.addEphemeralConstant("rand105", lambda: random.randint(-10,10))

# Negative weights of fitness for minimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Creating optimization toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Evaluation function for symbolic trees
def evalSymbReg(individual):

    global in_vector

    # Creating a function from expression
    func = toolbox.compile(expr=individual)

    # Finding result from input vectors
    result = np.array([func(*in_) for in_ in in_vector])

    return mean_square_error(result),
    
# Generation parameters
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Seed for repeatability
random.seed(318)

# Init population
pop = toolbox.population(n=population_size)
hof = tools.HallOfFame(10)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, iterations, stats=mstats,
                                halloffame=hof, verbose=True)

# pop, log, hof
# print(hof)

# Best fit function
opt_func = toolbox.compile(expr=hof[0])

# Evaluatng the input vector using the opt_func
result = np.array([opt_func(*in_) for in_ in in_vector])

expr = hof[0]
nodes, edges, labels = gp.graph(expr)

g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = nx.kamada_kawai_layout(g)

nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, labels)

# Comparison plots
plt.figure()
plt.plot(in_vector, y_vector, label='Ground truth')
plt.plot(in_vector, result, label='Function output')
plt.legend(loc='best')

plt.show()