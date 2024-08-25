import numpy as np
import pandas as pd
from tpg.trainer import Trainer, loadTrainer
from tpg.agent import Agent
import multiprocessing as mp
import time
import ast, pickle
import zlib
from pathlib import Path
from tpg.utils import getLearners, getTeams, learnerInstructionStats, actionInstructionStats, pathDepths
from ..extras import utils
#changes i made for _18, 1200 gen, operations def, and 200 pop (ALSO CHANGED TO LIST FOR PITCHES) SONG IS FUR ELISE
#changes i made for _19, 2500 gen, operations def, and 150 pop (ALSO CHANGED TO LIST FOR PITCHES) SONG IS FUR ELISE 
#changes i made for _20, 2500 gen, operations full, and 150 pop (ALSO CHANGED TO LIST FOR PITCHES) SONG IS FUR ELISE
#changes i made for _21, 2500 gen, operations def, and 150 pop initMaxTeamSize=5 CHANGE TO 10 initMaxProgSize=128 change to 100 SONG IS FUR ELISE
#changes i made for _22, same as 21 but weighted rmse now
#changes i made for _23, same as 21 but more generations
#changes i made for _24, using new fitness function (theil)
#changes i made for _25, using MSE function with direction composition (gen4)
#changes i made for _26, using NCD with envelope and compare it with entire training data (gen3)
#changes i made for _27, using NCD with envelope and compare it with the 50 values (gen3) (it wasn't good)
#changes i made for _28, using mean absolute error
#changes i made for _29, combine NCD and MSE (TPG 3)
#changes i made for _30, changed population size for team from 150 to 360, probability for atomic action from 1.0 to 0.95. max steps is 800->950. training =ncd, validation and testing mse
#changes i made for _31, swap NCD and MSE
#changes i made for _32, change back population to 150 (from _30) and use 1.0 probability
#changes i made for _33, change back population to 150 (from _30) and use 1.0 probability
#changes i made for _34, ncd all, with corrections made for validation and testing agent output tpg4
#changes i made for _35, mse all, with corrections made for validation and testing agent output tpg5
#changes i made for _36, mse for everything but testing, with corrections made for validation and testing agent output tpg6
#changes i made for _37, ncd for everything but testing, with corrections made for validation and testing agent output tpg7
#_38 short test for all ncd
#_39 check for ncd for training and testing and validation are mse (BAD)
#_40 flipped from 39 (BAD)
#_41 use correlation (BAD) too little generations maybe
#_42 NCD2
#_43 NCD 
#_44 theils but each separate and then added. (not based on sequence) (not good)
#_45 change windows from 5 to 50 and 10 to 100 NCD
#_46 actually use theils now
#_47 actually use theils but shorter everrythigns 5 5 10
#_48 FIX WINDOW

#_53 run theils and run ncd  4
#_53 run ncd for whole thing 5
#_53 run theils for every thing 6
#_53 run NCD for the whole thing 7 but use interdependencies
#_53 theils but parameters changed
#_54 theils 2 but parameters changed
#values that can be modified for testing 
MAX_STEPS_G = 960 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 0
EXTRA_TIME_STEPS  = 150 #number of wanted generated values 
STARTING_STEP = 0 #starting step

PRIME_STEPS = 50
TRAINING_STEPS = 50
VALIDATION_STEPS = TESTING_STEPS = 100
ACTION_WINDOW = 25 #window of steps it will look at during action

original_data = pd.read_csv("input.csv")
changed_data = pd.read_csv("input.csv")

starting_offset = changed_data['offset'].iloc[MAX_STEPS_G] #the starting offset will be the last as we forecast beyond this value (for reversing)
#MAX_PITCHES =  1  #changed_data['pitch'].apply(ast.literal_eval).apply(len).max() #max pitches played in a step for whole piece, will be one
#change offset to first order difference (FOD)
changed_data['offset'] = utils.compute_first_order_difference(changed_data['offset'])

# original values prescaled
prescaled_offset_min = changed_data['offset'].min()
prescaled_offset_max = changed_data['offset'].max() 
prescaled_duration_min = changed_data['duration_ppq'].min()  
prescaled_duration_max = changed_data['duration_ppq'].max() 
min_pitch = 0 
max_pitch = 127

#scaled
changed_data['offset'] = utils.min_max_scale(changed_data['offset'])
changed_data['duration_ppq'] = utils.min_max_scale(changed_data['duration_ppq'])
changed_data['pitch'] = utils.min_max_scale(changed_data['pitch'])

#add columns to new data
new_data = pd.DataFrame(changed_data, columns=['offset', 'duration_ppq', 'pitch'])   

training_data = pd.DataFrame(changed_data.iloc[:MAX_STEPS_G], columns=['offset', 'duration_ppq', 'pitch'])

#environment
class TimeSeriesEnvironment:
    def __init__(self, max_steps = MAX_STEPS_G, current_step_g=STARTING_STEP):

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
        window_size_act = ACTION_WINDOW * 3
        #reset values 
        done = False
        reward = 0
            
        _, action_values = action_type
        modified_offset, modified_duration, modified_pitch_array = action_values

        # Clip the values to ensure they are within the valid range
        modified_offset = utils.sigmoid(modified_offset)
        modified_duration = utils.sigmoid(modified_duration)
        modified_pitch_array = utils.sigmoid(modified_pitch_array)  # Assuming modified_pitch_array can be directly used with np.clip
        
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)

        if len(predicted_state_window) >= window_size_act:
            predicted_state_window = predicted_state_window[3:]
            predicted_state_window = np.append(predicted_state_window, predicted_state)  # Add the newest predicted_state

        #check if we reached the end, calculate the reward 
                #0 - 20           #20
        if self.current_step+1 <= self.max_generated_steps:
            #max steps arent reached so we keep adding each state to a string
            self.total_true_states.extend(self._get_state()) 
            self.total_states.extend(predicted_state)  # Append the new state as a tuple
        else:
            #end of window for episode
            #calculate the normalized compression distance with envelope to the whole song 
            self.total_states = np.array(self.total_states, dtype=np.float64)
            self.total_true_states = np.array(self.total_true_states, dtype=np.float64)
            if reward_func == 'ncd':
                reward = -utils.ncd(self.total_states.ravel(), self.total_true_states.ravel()) * 100
            elif reward_func == 'mse':
                reward = -utils.mse(self.total_states.ravel(), self.total_true_states.ravel())
            elif reward_func == 'correlation':
                reward = -utils.calc_correlation_2(self.total_states.ravel(), self.total_true_states.ravel())
            elif reward_func == 'theils':
                row = changed_data.iloc[self.current_step+1]
                row = (row['offset'], row['duration_ppq'], row['pitch']) #dk if i need this, change later
                self.total_true_states = np.append(self.total_true_states, row)
                reward = -utils.theil_u_statistic(self.total_true_states.ravel(), self.total_states.ravel())
            #print("states: ", self.total_states, "reward: ", reward)
            done = True

        self.current_step +=1
        return self.last_state, predicted_state_window, reward, done

    def step_simulation(self, action_type):
        # stuff print(action_type)
        _, action_values = action_type
        modified_offset, modified_duration, modified_pitch_array = action_values

        # Clip the values to ensure they are within the valid range
        modified_offset = utils.sigmoid(modified_offset)
        modified_duration = utils.sigmoid(modified_duration)
        modified_pitch_array = utils.sigmoid(modified_pitch_array)  # Assuming modified_pitch_array can be directly used with np.clip
        
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        self.current_step +=1
        return predicted_state
    
    
    def get_window(self):
        window_size = ACTION_WINDOW * 3
        state = np.ones(window_size) 
        return self._update_window(state)

    def _update_window(self,state):
        state = np.hstack(state)[3:]
        next_row = changed_data.iloc[self.current_step]
        state = np.append(state, (next_row['offset'], next_row['duration_ppq'], next_row['pitch']))
        return np.hstack(state)
    
    # Access the row corresponding to the current step
    def _get_state(self):
        row = changed_data.iloc[self.current_step]
        state = (row['offset'], row['duration_ppq'], row['pitch'])
        return state


def runAgent(args):
    agent, scoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()
    env = TimeSeriesEnvironment()
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
        while True:
            #prime first half of the episode
            #if current step is less than the second half (0-49)
            if  env.current_step <= ((ep * episode_length) + PRIME_STEPS - 1):
                action_value = (agent.act(np.hstack(action_state)))
                env.current_step += 1 #increase step in environment
                action_state = env._update_window(action_state)
                predicted_state = action_state #updating predicted
            else:
                action_value = (agent.act(np.hstack(predicted_state))) #takes first correct one then loops the steps 
                #stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, predicted_state, reward_func='ncd') #now fix step 
                scoreEp += reward
                
            # Apply clipping to continuous actions : action_value 
            #action_state gives you the next step
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                # stuff print("we finished")
                break
        scoreTotal += scoreEp
        #print("ScoreTotal:", scoreTotal)

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task='main')

    scoreList.append((agent.team.id, agent.team.outcomes))
    return agent, scoreList

def RunValidationAgents(args):
    agent, validscoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()

    env = TimeSeriesEnvironment()
    scoreTotal = 0
    episode_length = PRIME_STEPS + VALIDATION_STEPS #50 + 100 = 150
    numEpisodes = int((MAX_STEPS_G - PRIME_STEPS) / episode_length) # 750 / 150 = 5
    reward = 0

    for ep in range(numEpisodes):
        isDone = False
        #memory array is returned as this is the action state
        _ = env.reset(ep, episode_length) #resets at next 25 window (based on episode)
        action_state = env.get_window()  #makes it a window from the reset
        predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        while True:
            #prime first half of the episode
            #if current step is less than the second half (0-48)
            if  env.current_step <= ((ep * episode_length) + PRIME_STEPS - 1):
                action_value = (agent.act(np.hstack(action_state)))
                env.current_step += 1 #increase step in environment
                action_state = env._update_window(action_value[1])
                predicted_state = action_state #updating predicted
            else:
                action_value = (agent.act(np.hstack(predicted_state))) #takes first correct one then loops the steps 
                #stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, predicted_state, reward_func='theils') #now fix step 
                scoreEp += reward
                
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                break
        scoreTotal += scoreEp

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task='validation')

    validscoreList.append((agent.team.id, agent.team.outcomes))
    return agent, validscoreList

def RunTestingAgents(args):
    agent, validscoreList = args

    #function for handling parallelism
    agent.configFunctionsSelf()

    env = TimeSeriesEnvironment(current_step_g=PRIME_STEPS-1)
    scoreTotal = 0
    episode_length = PRIME_STEPS + TESTING_STEPS #50 + 100 = 150
    numEpisodes = int((MAX_STEPS_G - PRIME_STEPS) / episode_length) # 750 / 150 = 5
    reward = 0

    for ep in range(numEpisodes):
        isDone = False
        #memory array is returned as this is the action state
        action_state = env.reset(ep, episode_length) #resets at next 25 window (based on episode)
        #predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        action_state = env.get_window()  #makes it a window from the reset
        predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        while True:
            #prime first half of the episode
            #if current step is less than the second half (0-48)
            if  env.current_step <= ((ep * episode_length) + PRIME_STEPS - 1):
                action_value = (agent.act(np.hstack(action_state)))
                env.current_step += 1 #increase step in environment
                action_state = env._update_window(action_value[1])
                predicted_state = action_state #updating predicted
            else:
                action_value = (agent.act(np.hstack(predicted_state))) #takes first correct one then loops the steps 
                #stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, predicted_state, reward_func='theils') #now fix step 
                scoreEp += reward
                
            
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                break
        scoreTotal += scoreEp

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task='testing')

    validscoreList.append((agent.team.id, agent.team.outcomes))
    return agent, validscoreList

def RunBestAgent(args):
    agent, scoreList = args
    # Initialize the environment but starting from the end of the data

    simulation_results = []

    env = TimeSeriesEnvironment()
    # stuff print("Priming")
    # 0 to 799 (inclusive)
    window = np.ones(ACTION_WINDOW * 3) 
    print("PRE WINDOW: ", window)
    
    for x in range(MAX_STEPS_G - PRIME_STEPS -1, MAX_STEPS_G):
        env.current_step = x 
        print("current step: ", env.current_step)
        row = changed_data.iloc[x]
        window = np.hstack(window)[3:]
        
        window = np.append(window, (row['offset'], row['duration_ppq'], row['pitch']))
        action_value = (agent.act(np.hstack(window)))
        print("step "+str(x)+" for state: ", np.hstack(window))
        

    print("Priming complete")
    for x in range(EXTRA_TIME_STEPS-1): 
        #print("action value:", action_value)
        action_state = env.step_simulation(action_value)
        action_value = (agent.act(np.hstack(window)))
        window = np.hstack(window)[3:]
        window = np.append(window, action_state)
        simulation_results.append(action_state)
        last_state = action_state
    print("Simulation complete..")

    simulated_data = pd.DataFrame(simulation_results, columns=['offset', 'duration_ppq', 'pitch'])

    #invert the scale for offset
    
    #change this to use final step 
    original_offsets = np.cumsum(np.insert(simulated_data['offset'].values, 0, starting_offset))[1:]
    simulated_data['offset'] = simulated_data['offset'].apply(lambda x: round(utils.invert_min_max_scale(x, prescaled_offset_min, prescaled_offset_max), 7))
    simulated_data['offset'] = original_offsets

    #invert duration and pitches
    simulated_data['duration_ppq'] = simulated_data['duration_ppq'].apply(lambda x: round(utils.invert_min_max_scale(x, prescaled_duration_min, prescaled_duration_max), 3))
    simulated_data['pitch'] = simulated_data['pitch'].apply(lambda x: utils.invert_min_max_scale(x, min_pitch, max_pitch))

    simulated_data = pd.concat([original_data[:MAX_STEPS_G+1], simulated_data], ignore_index=True)
    simulated_data['pitch'] = simulated_data['pitch'].round()
    simulated_data.to_csv('Simulation_53.csv', index=False)

if __name__ == '__main__':
    tStart = time.time()
    trainer_checkpoint_path = Path("trainer_savepoint_53.pkl")
    gen_checkpoint_path = Path("gen_savepoint_53.txt")

    if trainer_checkpoint_path.exists():
        trainer = loadTrainer(trainer_checkpoint_path)
        print("LOADED TRAINER")
    else:
        trainer = Trainer(actions=[3], teamPopSize=150, initMaxTeamSize=10, initMaxProgSize=100, pActAtom=1.0, memType="default", operationSet="def")
        gen_start = 0
    
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, 'r') as file:
            gen_start = int(file.read().strip())  # Read the number and convert it to an integer
        print("LOADED GEN NUMBER: ", gen_start)
    else:
        gen_start = 0

    # Open a text file to write output
    with open('results_53.txt', 'a' if gen_start > 0 else 'w') as file:
        file.write(f"Trainer done: {trainer}\n")
        processes = mp.cpu_count()

        man = mp.Manager() 
        pool = mp.Pool(processes=processes)
            
        allScores = []

        for gen in range(gen_start, GENERATIONS): 
            scoreList = man.list()
            
            agents = trainer.getAgents()

            pool.map(runAgent, [(agent, scoreList) for agent in agents])
            
            teams = trainer.applyScores(scoreList)  
            
            champ = trainer.getEliteAgent(task='main')
            champ.saveToFile("best_agent_53")

            trainer.evolve(tasks=['main'])
            
            validation_champion_path = Path("validation_champion_53")
            testing_champion_path = Path("testing_champion_53")
        
                
            if gen % 10 == 0 and gen != 0 and gen % 100 != 0:  # Validation phase every 10 generations but not on 100th
                if validation_champion_path.exists():
                    with open(validation_champion_path, 'rb') as f:
                        best_validation_agent = pickle.load(f)
                        best_validation_score = best_validation_agent.team.outcomes['validation']
                else:
                    best_validation_agent = None
                    best_validation_score = float('-inf')
            
                best_relative_validation_score = float('-inf')
                looper = True
                start_validation_time = time.time()
                print("Values")
                while looper:
                    validationScores = man.list()
                    agents = trainer.getAgents()
                    
                    pool.map(RunValidationAgents, [(agent, validationScores) for agent in agents])
                    rootteams_ = trainer.applyScores(validationScores)
                    
                    #the current best of this evolution
                    current_best_validation = trainer.getEliteAgent(task='validation')
                    current_score = current_best_validation.team.outcomes['validation']
                    print("Validation Generation Score: ", current_best_validation.team.outcomes['validation'])
                    #save and retrieve best validation agent (since the best of gen != best)
                        
                    if current_best_validation.team.outcomes['validation'] > best_validation_score:
                        best_validation_score = current_best_validation.team.outcomes['validation']
                        best_validation_agent = current_best_validation
                        with open(validation_champion_path, 'wb') as f:
                            pickle.dump(best_validation_agent, f)
                    
                    #lower than previous: early stoppage
                    if current_best_validation.team.outcomes['validation'] < best_relative_validation_score:
                        looper= False
                    else: 
                        best_relative_validation_score = current_best_validation.team.outcomes['validation'] 
                        
                    print("Best agent of all time score: ", best_validation_score)
                    print(f"Validation champ with the best test score with {best_validation_score} on test data.")
        
                    with open("final_validation_scores_53.txt", 'w') as f:
                        f.write(str(best_validation_score))
                         
                            
                    if time.time() - start_validation_time > (3600*4):  # Check if 4 hour has passed
                        print("Time limit for finding a better validation champ exceeded.")
                        looper= False

                    if looper:
                        trainer.evolve(tasks=['validation'])
            
                
            if gen % 100 == 0 and gen != 0:  # testing phase every 100 generations
                if testing_champion_path.exists():
                    with open(testing_champion_path, 'rb') as f:
                        best_testing_agent = pickle.load(f)
                        best_testing_score = best_testing_agent.team.outcomes['testing']
                else:
                    best_testing_agent = None
                    best_testing_score = float('-inf')
                    
                best_relative_testing_score = float('-inf')
                looper = True
                start_testing_time = time.time()
                print("Values")
                while looper:
                    testingScores = man.list()
                    agents = trainer.getAgents()
                    
                    pool.map(RunTestingAgents, [(agent, testingScores) for agent in agents])
                    teams1 = trainer.applyScores(testingScores)
                    
                    #the current best of this evolution
                    current_best_testing = trainer.getEliteAgent(task='testing')
                    current_score = current_best_testing.team.outcomes['testing']
                    print("Testing Generation Score: ", current_best_testing.team.outcomes['testing'])
                    #save and retrieve best validation agent (since the best of gen != best)
                        
                    if current_best_testing.team.outcomes['testing'] > best_testing_score:
                        best_testing_score = current_best_testing.team.outcomes['testing']
                        best_testing_agent = current_best_testing
                        with open(testing_champion_path, 'wb') as f:
                            pickle.dump(best_testing_agent, f)
                    
                    #lower than previous: early stoppage
                    if current_best_testing.team.outcomes['testing'] < best_relative_testing_score:
                        looper= False
                    else: 
                        best_relative_testing_score = current_best_testing.team.outcomes['testing'] 
                        
                    print("Best agent of all time score: ", best_testing_score)
                    print(f"Testing champ with the best test score with {best_testing_score} on test data.")
        
                    with open("final_testing_scores_53.txt", 'w') as f:
                        f.write(str(best_testing_score))
                         
                            
                    if time.time() - start_testing_time > (3600*4):  # Check if 4 hour has passed
                        print("Time limit for finding a better validation champ exceeded.")
                        looper= False

                    if looper:
                        trainer.evolve(tasks=['testing'])
                    
            scoreStats = trainer.fitnessStats
            allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
            print(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}")
            file.write(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}\n")

            trainer.saveToFile("trainer_savepoint_53.pkl")
            with open("gen_savepoint_53.txt", 'w') as gen_file:
                gen_file.write(str(gen))
            
            #to keep the champ saved in a file for evaluation later on 
        
        file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
        file.write('Final Results:\nMin, Max, Avg\n')
        for score in allScores:
            file.write(f"{score}\n")

        champ = pickle.load(open("testing_champion_53", 'rb'))
        champ.configFunctionsSelf()
        print(champ.team)
        print(champ.team.fitness)
        print(champ.team.learners)
        print(champ.team.outcomes)
        print("---------------")
        #champ.configFunctions()

        # Assuming RunBestAgent is a function you have defined earlier
        #empty array is: scorelist
        RunBestAgent((champ, []))