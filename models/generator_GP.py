#imports
import numpy
import random
import gymnasium as gym
import operator
import math
from operator import attrgetter

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt

import multiprocessing
import numpy as np
import pandas as pd
import zlib, ast


#values that can be modified for testing 
MAX_STEPS_G = 14 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 50
EXTRA_TIME_STEPS  = 10 #number of wanted generated values 
NUM_EP = 50 #number of episodes for RL 
STARTING_STEP = 0 #starting step

data1 = pd.read_csv("input.csv", header = 0)

MAX_PITCHES = data1['pitch'].apply(ast.literal_eval).apply(len).max() #max pitches played in a step for whole piece
 
#memory size
memory_size = 15


#reward function, using Normalized Compression Distance
def calculate_ncd(seq1, seq2):
    seq1 = np.hstack(seq1)
    seq2 = np.hstack(seq2)
    # Serialize sequences
    seq1_bytes = str(seq1).encode()
    seq2_bytes = str(seq2).encode()

    # Compress individual sequences and concatenated sequence
    c_x = len(zlib.compress(seq1_bytes))
    c_y = len(zlib.compress(seq2_bytes))
    c_xy = len(zlib.compress(seq1_bytes + seq2_bytes))

    ncd = (c_xy - min(c_x, c_y)) / max(c_x, c_y)
    return ncd

#environment
class TimeSeriesEnvironment:
    def __init__(self, max_steps = MAX_STEPS_G, current_step_g=STARTING_STEP):

        self.max_generated_steps = max_steps
        self.current_step = current_step_g

        #starts off at 0, getting the 0th row in the dataset
        self.last_state = self._get_state()

    def reset(self):
        # Resets to the starting step we want
        self.current_step = STARTING_STEP

        self.last_state = self._get_state() 

        return self.last_state


#get predicted step and compare with the actual value
    
#first step: 0, last step:  19. len: 20
    def step(self, action_type):

        #reset values 
        done = False
        reward = 0

        #get the state we are on
        self.last_state = self._get_state()

        modified_offset = action_type[0] #+ self.last_state[0]
        modified_duration = action_type[1] #+ self.last_state[1]
        modified_pitch_array = action_type[2:] #+ self.last_state[2]
        #clipping the values now: 
        modified_offset = np.clip(modified_offset, action_type[0], None)  # Clip the first action
        modified_duration = np.clip(modified_duration, 0, None)  # Second clipping if it's larger than previous offset value
        modified_pitch_array = np.clip(modified_pitch_array, -1, 127)  # Clip the second action

        # modified pitch array should be rounded
        modified_pitch_array = [round(p) for p in modified_pitch_array]
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        #check if we reached the end, calculate the reward 
                #0 - 19           #20 
        #get the next value to reward based on the predicted value
        if self.current_step+1 < len(data1):
            actual_row = data1.iloc[self.current_step+1]
            self.real_state = (actual_row['offset'], actual_row['duration_ppq'], np.array(ast.literal_eval(actual_row['pitch'])))
            reward = -calculate_ncd(predicted_state, self.real_state)
        else:
            #end of dataset
            done = True

        self.current_step +=1
        return self.last_state, reward, done

    def step_simulation(self, action_type):
        
         # Apply continuous actions to the state 
        modified_offset = action_type[0]
        modified_duration = action_type[1]
        modified_pitch_array = np.array(action_type[2:])
        
        #clipping the values now: 
        modified_offset = np.clip(modified_offset, action_type[0], None)  # Clip the first action
        modified_duration = np.clip(modified_duration, 0, None)  # Second clipping if it's larger than previous offset value
        modified_pitch_array = np.clip(modified_pitch_array, -1, 127)  # Clip the second action
        
        # modified pitch array should be rounded
        modified_pitch_array = [round(p) for p in modified_pitch_array]
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        self.current_step +=1
        return predicted_state
    
    # Access the row corresponding to the current step
    def _get_state(self):
        row = data1.iloc[self.current_step]
        state = (row['offset'], row['duration_ppq'], np.array(ast.literal_eval(row['pitch'])))
        return state



#primitives
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

# helper function to limit decimal places
def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)
    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def valid_index(index, size):
    if index < 0:
        return 0
    elif index >= size:
        return size - 1
    return int(index)

def read(memory, index):
    index = valid_index(index, len(memory))
    return memory[index]

def write(memory, value, index):
    index = valid_index(index, len(memory))
    memory[index] = value
    return memory[index]

def exp_of_num(input):
    try:
        result = input * input
        if math.isinf(result):
            return float('inf') if input > 0 else float('-inf')
        return truncate(result, 8)
    except OverflowError:
        return float('inf')


def sqrt_pos(input):
    return math.sqrt(abs(input))

def dummy_pri(input):
    return input


obs_size = MAX_PITCHES+2
print("observation size: ", obs_size)


pset = gp.PrimitiveSetTyped("MAIN", [list, float, float, float, float, float, float, float, float], 
                            [list, float, float, float, float, float, float, float, float])
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
#pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(math.sin, [float], float)

for i in range(0, memory_size):
    pset.addTerminal(float(i), float)


pset.addPrimitive(read, [list, float], float)  # 'read' needs an integer index
pset.addPrimitive(write, [list, float, float], float)
pset.addPrimitive(dummy_pri, [list], list)

#added primitives 
pset.addPrimitive(exp_of_num, [float], float)  # 'read' needs an integer index
pset.addPrimitive(sqrt_pos, [float], float)  # 'read' needs an integer index
#pset.addPrimitive(math.cos, [float], float)

# evaluates the fitness of an individual policy
def evalRL(policy):
    env = TimeSeriesEnvironment()
    # transform expression tree to functional Python code
    get_action = gp.compile(policy, pset)
    fitness = 0
    memory = [0.0] * 2
    for _ in range(0, NUM_EP):
        done = False
        truncated = False
        # reset environment and get first observation
        observation = env.reset()
        observation = np.hstack((observation[:2],observation[2]))
        print("OBSERVATION:", observation)
        
        episode_reward = 0
        num_steps = 0
        # evaluation episode
        while not (done):
            # use the expression tree to compute action
            #change so it doesn't accept velocity
            action = get_action(memory, observation)
            
            print("our action: ", action)
            observation, reward, done = env.step(action)
            episode_reward += reward

        fitness += episode_reward
    return (fitness / NUM_EP,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalRL)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(42)
    # set to the number of cpu cores available
    num_parallel_evals = multiprocessing.cpu_count()

    
    population_size = 360
    num_generations = GENERATIONS
    prob_xover = 0.9
    prob_mutate = 0.1

    pop = toolbox.population(n=population_size)

    # HallOfFame archives the best individuals found so far,
    # even if they are deleted from the population.
    hof = tools.HallOfFame(1)  # We keep the single best.

    # configues what stats we want to track
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #setup parallel evaluations
    pool = multiprocessing.Pool(processes=num_parallel_evals)
    toolbox.register("map", pool.map)

    # run the evolutionary algorithm
    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        prob_xover,
        prob_mutate,
        num_generations,
        stats=mstats,
        halloffame=hof,
        verbose=True,
    )

    pool.close()

    best_fits = log.chapters["fitness"].select("max")
    best_fit = truncate(hof[0].fitness.values[0], 0)

    print("Best fitness: " + str(best_fit))
    print(hof[0])

if __name__ == '__main__':
    main()
