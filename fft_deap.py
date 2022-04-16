import copy
import random

import numpy

from functools import partial
import functools

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from inspect import isclass

from chopsticks import ChopsticksSimulator

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

def generateHandName():
    return random.randint(0, 1)

def generateHandValue():
    return random.randint(0, 4)

cs = ChopsticksSimulator()

pset = gp.PrimitiveSetTyped("MAIN", [], Null)

# Nested If then elses
pset.addPrimitive(if_then_else, [Bool, HandAction, Null], Null)
pset.addPrimitive(if2_then_else, [Bool, Bool, HandAction, Null], Null)
pset.addPrimitive(if3_then_else, [Bool, Bool, Bool, HandAction, Null], Null)
pset.addPrimitive(if4_then_else, [Bool, Bool, Bool, Bool, HandAction, Null], Null)

# Terminal If then elses
pset.addPrimitive(if_then_else, [Bool, HandAction, HandAction], Null)
pset.addPrimitive(if2_then_else, [Bool, Bool, HandAction, HandAction], Null)
pset.addPrimitive(if3_then_else, [Bool, Bool, Bool, HandAction, HandAction], Null)
pset.addPrimitive(if4_then_else, [Bool, Bool, Bool, Bool, HandAction, HandAction], Null)

# Conditions
pset.addPrimitive(cs.hand_greater, [HandName, HandValue], Bool)
pset.addPrimitive(cs.hand_equal, [HandName, HandValue], Bool)
pset.addPrimitive(cs.hand_less, [HandName, HandValue], Bool)
pset.addPrimitive(cs.opp_hand_greater, [HandName, HandValue], Bool)
pset.addPrimitive(cs.opp_hand_equal, [HandName, HandValue], Bool)
pset.addPrimitive(cs.opp_hand_less, [HandName, HandValue], Bool)

# Actions
pset.addPrimitive(cs.attack, [HandName, HandName], HandAction)
pset.addPrimitive(cs.generate_transfer, [HandValue], HandAction)

# Constants
pset.addEphemeralConstant(name="hand_name", ephemeral=generateHandName, ret_type=HandName)
pset.addEphemeralConstant(name="hand_value", ephemeral=generateHandValue, ret_type=HandValue)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", fft_generate, pset=pset)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

toolbox.register("evaluate", evalChopsticks)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", fft_generate)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def evolve_players(ngen=50):
    pop = toolbox.population(n=800)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, ngen, stats, halloffame=hof)

    return pop, hof, stats

if __name__ == "__main__":
    cs.set_opp_strat([cs.heuristic_random_move])
    pop, hof, stats = evolve_players()
    print(hof[0])
