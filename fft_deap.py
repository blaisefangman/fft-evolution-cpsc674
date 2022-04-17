import copy
import random
import sys

import numpy

import operator

from functools import partial
import functools

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from inspect import isclass

from chopsticks import ChopsticksSimulator

import multiprocessing

def fft_generate(pset, type_=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    stack = [type_]
    while len(stack) != 0:
        type_ = stack.pop()
        node = random.choice(pset.terminals[type_] + pset.primitives[type_])

        try:
            node in pset.primitives[type_]
            for arg in reversed(node.args):
                stack.append(arg)
        except:
            if isclass(node):
                node = node()
        expr.append(node)

    return expr

def if_then_else(cond, out1, out2):
    return lambda: out1() if cond() else out2()

def if2_then_else(cond1, cond2, out1, out2):
    return lambda: out1() if cond1() and cond2() else out2()

def if3_then_else(cond1, cond2, cond3, out1, out2):
    return lambda: out1() if cond1() and cond2() and cond3() else out2()

def if4_then_else(cond1, cond2, cond3, cond4, out1, out2):
    return lambda: out1() if cond1() and cond2() and cond3() and cond4() else out2()

class HandValue(): pass

class HandName(): pass

class HandAction(): pass

class Null(): pass

class Bool(): pass

class MyHandBool(): pass

class OppHandBool(): pass

def generateHandName():
    return random.randint(0, 1)

def generateHandValue():
    return random.randint(0, 4)

cs = ChopsticksSimulator()

pset = gp.PrimitiveSetTyped("MAIN", [], Null)

# Nested If then elses
pset.addPrimitive(if_then_else, [MyHandBool, HandAction, Null], Null)
pset.addPrimitive(if_then_else, [OppHandBool, HandAction, Null], Null)

pset.addPrimitive(if2_then_else, [MyHandBool, MyHandBool, HandAction, Null], Null)
pset.addPrimitive(if2_then_else, [MyHandBool, OppHandBool, HandAction, Null], Null)

pset.addPrimitive(if3_then_else, [MyHandBool, MyHandBool, OppHandBool, HandAction, Null], Null)
pset.addPrimitive(if3_then_else, [MyHandBool, OppHandBool, OppHandBool, HandAction, Null], Null)

pset.addPrimitive(if4_then_else, [MyHandBool, MyHandBool, OppHandBool, OppHandBool, HandAction, Null], Null)

# Terminal If then elses
pset.addPrimitive(if_then_else, [MyHandBool, HandAction, HandAction], Null)
pset.addPrimitive(if_then_else, [OppHandBool, HandAction, HandAction], Null)

pset.addPrimitive(if2_then_else, [MyHandBool, MyHandBool, HandAction, HandAction], Null)
pset.addPrimitive(if2_then_else, [MyHandBool, OppHandBool, HandAction, HandAction], Null)

pset.addPrimitive(if3_then_else, [MyHandBool, MyHandBool, OppHandBool, HandAction, HandAction], Null)
pset.addPrimitive(if3_then_else, [MyHandBool, OppHandBool, OppHandBool, HandAction, HandAction], Null)

pset.addPrimitive(if4_then_else, [MyHandBool, MyHandBool, OppHandBool, OppHandBool, HandAction, HandAction], Null)

# Conditions
pset.addPrimitive(cs.hand_greater, [HandName, HandValue], MyHandBool)
pset.addPrimitive(cs.hand_equal, [HandName, HandValue], MyHandBool)
pset.addPrimitive(cs.hand_less, [HandName, HandValue], MyHandBool)

pset.addPrimitive(cs.opp_hand_greater, [HandName, HandValue], OppHandBool)
pset.addPrimitive(cs.opp_hand_equal, [HandName, HandValue], OppHandBool)
pset.addPrimitive(cs.opp_hand_less, [HandName, HandValue], OppHandBool)

# Actions
pset.addPrimitive(cs.attack, [HandName, HandName], HandAction)
pset.addPrimitive(cs.generate_transfer, [HandValue], HandAction)

# Constants
pset.addEphemeralConstant(name="hand_name", ephemeral=generateHandName, ret_type=HandName)
pset.addEphemeralConstant(name="hand_value", ephemeral=generateHandValue, ret_type=HandValue)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def evalChopsticks(individual):
    NUM_EVALS = 25
    # Transform the tree expression to functional Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine
    my_wins, my_rounds = 0, 0
    for i in range(NUM_EVALS):
        result = cs.run(routine)
        my_wins += int(result[0] == 0)
        my_rounds += result[1]

    return (my_wins / NUM_EVALS,)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", fft_generate, pset=pset)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalChopsticks)
toolbox.register("select", tools.selTournament, tournsize=300)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", fft_generate)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=60))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=60))

def evolve_players(ngen=40, pop=None):
    if pop is None:
        pop = toolbox.population(n=20000)
    hof = tools.HallOfFame(25)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("med", numpy.median)
    stats.register("std", numpy.std)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, ngen, stats, halloffame=hof)

    return pop, hof, stats

if __name__ == '__main__':
    filename = sys.argv[1]
    f = open(filename, 'w')

    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)

    pop, hof, stats = evolve_players(ngen=20)
    for i in range(5):
        f.write(str(hof[0]))
        f.write('\n\n')
        new_strats = [gp.compile(s, pset) for s in hof]
        cs.set_opp_strat(new_strats)
        for ind in pop:
            del ind.fitness.values
        pop, hof, stats = evolve_players(ngen=20, pop=pop)
    f.write("FINAL TOP 10\n\n")
    for i in range(10):
        f.write(str(hof[i]))
        f.write('\n\n')
