import numpy as np
import pandas as pd
from tpg.trainer import Trainer, loadTrainer
from tpg.agent import Agent
import multiprocessing as mp
from multiprocessing import Barrier
#from threading import Barrier
import time
import ast, pickle
import zlib
from pathlib import Path
from tpg.utils import getLearners, getTeams, learnerInstructionStats, actionInstructionStats, pathDepths
import extras.utils as utils
#UNIVARIATE TESTS
#cauchyhalf
#cauchy
#default
#cauchy for 100
#for 5 lets try without resetting every generation
MAX_STEPS_G = 960 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 20
EXTRA_TIME_STEPS  = 150 #number of wanted generated values 
STARTING_STEP = 0 #starting step

PRIME_STEPS = 25
TRAINING_STEPS = 25
VALIDATION_STEPS = TESTING_STEPS = 50
ACTION_WINDOW = 25 #window of steps it will look at during action

original_data = pd.read_csv("input_uni.csv")
changed_data = pd.read_csv("input_uni.csv")

min_pitch = changed_data['pitch'].min()
max_pitch = changed_data['pitch'].max()

changed_data['pitch'] = utils.min_max_scale(changed_data['pitch'])

#add columns to new data
new_data = pd.DataFrame(changed_data, columns=['pitch'])   

training_data = pd.DataFrame(changed_data.iloc[:MAX_STEPS_G], columns=['pitch'])

def produce_smooth(window_size):
    #Produces a trailing moving average of the input data.
    smoothed_data = changed_data.rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

#environment
class TimeSeriesEnvironment:
    def __init__(self, data, max_steps = MAX_STEPS_G, current_step_g=STARTING_STEP):
        self.data = data
        self.max_generated_steps = max_steps
        self.current_step = current_step_g
        self.total_states = []
        self.total_true_states = []
        #starts off at 0, getting the 0th row in the dataset
        self.last_state = self._get_state()

    def reset(self, episodenum, window_size):
        # Resets to the starting step we want
        self.total_states = []
        self.total_true_states = []
        self.current_step = episodenum * window_size 

        self.max_generated_steps = (self.current_step + window_size)
        self.last_state = self._get_state() 

        return self.last_state


#get predicted step and compare with the actual value
    
#first step: 0, last step:  19. len: 20s
    def step(self, action_type, predicted_state_window, reward_func):
        #reset values 
        done = False
        reward = 0
            
        _, action_values = action_type
        modified_pitch_array = action_values
        
        modified_pitch_array = utils.sigmoid(modified_pitch_array)  # Assuming modified_pitch_array can be directly used with np.clip
        
        predicted_state_window = predicted_state_window[1:]  # Remove the first element
        predicted_state_window = np.append(predicted_state_window, modified_pitch_array)  # Add the newest predicted_state

        #check if we reached the end, calculate the reward 
                #0 - 20           #20

        if self.current_step+1 <= self.max_generated_steps:

            #max steps arent reached so we keep adding each state to a string
            self.total_true_states = np.append(self.total_true_states, self._get_state()) 
            self.total_states = np.append(self.total_states, modified_pitch_array)  # Append the new state as a tuple
        else:
        
            #end of window for episode
            #calculate the normalized compression distance with envelope to the whole song 
                #TODO conversion needed?
            self.total_states = np.array(self.total_states, dtype=np.float64)
            self.total_true_states = np.array(self.total_true_states, dtype=np.float32)
            
            if reward_func == 'ncd':
                reward = -utils.ncd(self.total_states.ravel(), self.total_true_states.ravel()) * 100
            elif reward_func == 'mse':
                reward = -utils.mse(self.total_states.ravel(), self.total_true_states.ravel())
            elif reward_func == 'correlation':
                reward = -utils.calc_correlation_2(self.total_states.ravel(), self.total_true_states.ravel())
            elif reward_func == 'theils':
                reward = -utils.theil_u_statistic_u(self.total_true_states.ravel(), self.total_states.ravel())
            elif reward_func == 'hybrid':
                reward = (-utils.theil_u_statistic_u(self.total_true_states.ravel(), self.total_states.ravel())*0.7) + (-utils.ncd_u(self.total_true_states.ravel(), self.total_states.ravel())*0.3)
            #print("states: ", self.total_states, "reward: ", reward)
            done = True

        self.current_step +=1
        return self.last_state, predicted_state_window, reward, done

    def step_simulation(self, action_type):
        # stuff print(action_type)
        _, action_values = action_type

        # Clip the values to ensure they are within the valid range
        modified_pitch_array = utils.sigmoid(action_values)  # Assuming modified_pitch_array can be directly used with np.clip
        
        self.current_step +=1
        return modified_pitch_array
    
    def get_window(self): 
        return self._update_window(np.ones(ACTION_WINDOW))

    def _update_window(self,window):
        window = np.hstack(window)[1:]
        next_row = self.data.iloc[self.current_step]
        window = np.append(window, next_row['pitch'])
        return window
    
    # Access the row corresponding to the current step
    def _get_state(self):
        row = self.data.iloc[self.current_step]
        return row['pitch']


def runAgent(args):
    agent, scoreList, data, task_todo = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()
    env = TimeSeriesEnvironment(data)
    scoreTotal = 0
    episode_length = PRIME_STEPS + TRAINING_STEPS
    numEpisodes = int(MAX_STEPS_G / episode_length) # 500 / 5 = 10
    reward = 0

    for ep in range(numEpisodes):
        isDone = False
        _ = env.reset(ep, episode_length) #resets at next 100 window (based on episode)

        action_state = env.get_window()  #makes it a window from the reset
        predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        i = 0
        while True:
            #prime first half of the episode
            #if current step is less than the second half (0-49)
            if  env.current_step <= ((ep * episode_length) + PRIME_STEPS - 1):
                action_value = (agent.act(np.hstack(action_state))) #can ignore the action_value since we are priming
                env.current_step += 1 #increase step in environment
                action_state = env._update_window(action_state)
                predicted_state = action_state #updating predicted          
            else:
                action_value = (agent.act(np.hstack(predicted_state))) #takes first correct one then loops the steps 
                #stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, predicted_state, reward_func='theils') #now fix step 
                scoreEp += reward
                
            # Apply clipping to continuous actions : action_value 
            #action_state gives you the next step
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                # stuff print("we finished")
                break
            i+=1
        scoreTotal += scoreEp
        #print("ScoreTotal:", scoreTotal)

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task=task_todo)

    scoreList.append((agent.team.id, agent.team.outcomes))
    return agent, scoreList


def RunBestAgent(args):
    agent, scoreList, changed_data = args
    # Initialize the environment but starting from the end of the data

    simulation_results = []

    env = TimeSeriesEnvironment(changed_data)
    window = np.ones(ACTION_WINDOW) 
    
    for x in range(MAX_STEPS_G - PRIME_STEPS -1, MAX_STEPS_G):
        env.current_step = x 
        row = changed_data.iloc[x]
        window = np.hstack(window)[1:]
        
        window = np.append(window, row['pitch'])
        action_value = (agent.act(np.hstack(window)))
        

    print("Priming complete")
    for x in range(EXTRA_TIME_STEPS-1): 
        #print("action value:", action_value)
        action_state = env.step_simulation(action_value)
        action_value = (agent.act(np.hstack(window)))
        window = np.hstack(window)[1:]
        window = np.append(window, action_state)
        simulation_results.append(action_state)
    print("Simulation complete..")

    simulated_data = pd.DataFrame(simulation_results, columns=['pitch'])

    #invert the scale for offset
    
    simulated_data['pitch'] = simulated_data['pitch'].apply(lambda x: utils.invert_min_max_scale(x, min_pitch, max_pitch))

    simulated_data = pd.concat([original_data[:MAX_STEPS_G+1], simulated_data], ignore_index=True)
    simulated_data['pitch'] = simulated_data['pitch'].round()
    simulated_data.to_csv('checkpoints/Simulation_5.csv', index=False)

if __name__ == '__main__':
    data_20 = produce_smooth(20)
    data_15 = produce_smooth(15)
    data_10 = produce_smooth(10)
    data_8= produce_smooth(5)
    data_1 = changed_data
    
    tStart = time.time()
    trainer_checkpoint_path = Path("checkpoints/trainer_savepoint_5.pkl")
    gen_checkpoint_path = Path("checkpoints/gen_savepoint_5.txt")

    if trainer_checkpoint_path.exists():
        trainer = loadTrainer(trainer_checkpoint_path)
        print("LOADED TRAINER")
    else:
        trainer = Trainer(actions=[1], teamPopSize=200, initMaxTeamSize=10, initMaxProgSize=100, pActAtom=1.0, memType="default", operationSet="def")
        gen_start = 0
    
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, 'r') as file:
            gen_start = int(file.read().strip())  # Read the number and convert it to an integer
        print("LOADED GEN NUMBER: ", gen_start)
    else:
        gen_start = 0

    # Open a text file to write output
    with open('checkpoints/results_5.txt', 'a' if gen_start > 0 else 'wb') as file:
        #file.write('Trainer done:'.encode()+str(trainer).encode())
        processes = mp.cpu_count()

        man = mp.Manager() 
        pool = mp.Pool(processes=processes)
            
        allScores = []
        prev_score_avg = -10000000000
        
        for gen in range(gen_start, GENERATIONS): 
            #-------------------------------------------------
            print("GENERATION ", gen)

            #frued newman
            #-------------------------------------------------
            task_todo = ['smoothing_20','smoothing_15','smoothing_10','smoothing_8','smoothing_1']
            
            for task_val in task_todo:
                scoreList = man.list()
                agents = trainer.getAgents()
                data_number = task_val.split('_')[1]
                # Access the corresponding data variable dynamically
                data_variable_name = locals()[f'data_{data_number}']
                
                pool.map(runAgent, [(agent, scoreList, data_variable_name, task_val) for agent in agents])
                
                teams = trainer.applyScores(scoreList)  
                
                champ = trainer.getEliteAgent(task=task_val)
                champ.saveToFile("checkpoints/"+task_val)
                trainer.evolve(tasks=[task_val])

        #file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
        #file.write('Final Results:\nMin, Max, Avg\n')
        #for score in allScores:
         #   file.write(f"{score}\n")

        champ = pickle.load(open("checkpoints/smoothing_1", 'rb'))
        champ.configFunctionsSelf()
        print(champ.team)
        print(champ.team.fitness)
        print(champ.team.learners)
        print(champ.team.outcomes)
        print("---------------")
        #champ.configFunctions()

        # Assuming RunBestAgent is a function you have defined earlier
        #empty array is: scorelist
        RunBestAgent((champ, [], changed_data))