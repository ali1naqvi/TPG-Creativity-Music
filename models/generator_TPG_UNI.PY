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
GENERATIONS = 100
EXTRA_TIME_STEPS  = 150 #number of wanted generated values 
STARTING_STEP = 0 #starting step

PRIME_STEPS = 50
TRAINING_STEPS = 50
VALIDATION_STEPS = TESTING_STEPS = 100
ACTION_WINDOW = 25 #window of steps it will look at during action

original_data = pd.read_csv("input_uni.csv")
changed_data = pd.read_csv("input_uni.csv")

min_pitch = changed_data['pitch'].min()
max_pitch = changed_data['pitch'].max()

changed_data['pitch'] = utils.min_max_scale(changed_data['pitch'])

#add columns to new data
new_data = pd.DataFrame(changed_data, columns=['pitch'])   

training_data = pd.DataFrame(changed_data.iloc[:MAX_STEPS_G], columns=['pitch'])

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
        #reset values 
        done = False
        reward = 0
            
        _, action_values = action_type
        modified_pitch_array = action_values
        
        modified_pitch_array = utils.sigmoid(modified_pitch_array)  # Assuming modified_pitch_array can be directly used with np.clip

        predicted_state_window = predicted_state_window[1:]  # Remove the first element
        predicted_state_window += modified_pitch_array  # Add the newest predicted_state

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
            self.total_true_states = np.array(self.total_true_states, dtype=np.float64)
            
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
        next_row = changed_data.iloc[self.current_step]
        window = np.append(window, next_row['pitch'])
        return window
    
    # Access the row corresponding to the current step
    def _get_state(self):
        row = changed_data.iloc[self.current_step]
        return row['pitch']


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
                action_state = env._update_window(action_state)
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
                action_state = env._update_window(action_state)
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
        prev_score_avg = -(np.inf)

        for gen in range(gen_start, GENERATIONS): 
            scoreList = man.list()
            
            agents = trainer.getAgents()
            pool.map(runAgent, [(agent, scoreList) for agent in agents])
            
            teams = trainer.applyScores(scoreList)  
            
            champ = trainer.getEliteAgent(task='main')
            champ.saveToFile("checkpoints/best_agent_5")
            trainer.evolve(tasks=['main'])
            
            validation_champion_path = Path("checkpoints/validation_champion_5")
            testing_champion_path = Path("checkpoints/testing_champion_5")
        
                
            #if gen % 10 == 0 and gen != 0 and gen % 100 != 0:  # Validation phase every 10 generations but not on 100th
            if validation_champion_path.exists():
                with open(validation_champion_path, 'rb') as f:
                    best_validation_agent = pickle.load(f)
                    best_validation_score = best_validation_agent.team.outcomes['validation']
            else:
                best_validation_agent = None
                best_validation_score = float('-inf')

            validationScores = man.list()
            agents = trainer.getAgents()
            
            pool.map(RunValidationAgents, [(agent, validationScores) for agent in agents])
            rootteams_ = trainer.applyScores(validationScores)
            
            #the current best of this evolution
            current_best_validation = trainer.getEliteAgent(task='validation')
            current_score = current_best_validation.team.outcomes['validation']
            #save and retrieve best validation agent (since the best of gen != best)
                
            if current_best_validation.team.outcomes['validation'] > best_validation_score:
                best_validation_score = current_best_validation.team.outcomes['validation']
                best_validation_agent = current_best_validation
                with open(validation_champion_path, 'wb') as f:
                    pickle.dump(best_validation_agent, f)
                
                
            print(f"Validation champ with the best test score with {best_validation_score} on test data.")

            with open("checkpoints/final_validation_scores_5.txt", 'w') as f:
                f.write(str(best_validation_score))
                    
            trainer.evolve(tasks=['validation'])
            
            #if gen % 100 == 0 and gen != 0:  # testing phase every 100 generations
            if testing_champion_path.exists():
                with open(testing_champion_path, 'rb') as f:
                    best_testing_agent = pickle.load(f)
                    best_testing_score = best_testing_agent.team.outcomes['testing']
            else:
                best_testing_agent = None
                best_testing_score = float('-inf')
                
            testingScores = man.list()
            agents = trainer.getAgents()
            
            pool.map(RunTestingAgents, [(agent, testingScores) for agent in agents])
            teams1 = trainer.applyScores(testingScores)
            
            #the current best of this evolution
            current_best_testing = trainer.getEliteAgent(task='testing')
            current_score = current_best_testing.team.outcomes['testing']
            #save and retrieve best validation agent (since the best of gen != best)
                
            if current_best_testing.team.outcomes['testing'] > best_testing_score:
                best_testing_score = current_best_testing.team.outcomes['testing']
                best_testing_agent = current_best_testing
                with open(testing_champion_path, 'wb') as f:
                    pickle.dump(best_testing_agent, f)
                
            print(f"Testing champ with the best test score with {best_testing_score} on test data.")

            with open("checkpoints/final_testing_scores_5.txt", 'w') as f:
                f.write(str(best_testing_score))

            trainer.evolve(tasks=['testing'])
                    
            scoreStats = trainer.fitnessStats
            allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
            print(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}")
            file.write('Gen: '.encode() + str(gen).encode() + "\n".encode())
            file.write('Results so far: '.encode() + str(allScores).encode())
            
            trainer.saveToFile("checkpoints/trainer_savepoint_5.pkl")
            with open("checkpoints/gen_savepoint_5.txt", 'w') as gen_file:
                gen_file.write(str(gen))
            #reset memory if the average goes down
            #if scoreStats['average'] < prev_score_avg: #first 20 we reset and then every 10th subsequently 
                #for agent in agents:
                    #agent.actVars['memMatrix'].fill(0)
            prev_score_avg = scoreStats['average']
            
            #to keep the champ saved in a file for evaluation later on 
        
        #file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
        #file.write('Final Results:\nMin, Max, Avg\n')
        #for score in allScores:
         #   file.write(f"{score}\n")

        champ = pickle.load(open("checkpoints/testing_champion_5", 'rb'))
        champ.configFunctionsSelf()
        print(champ.team)
        print(champ.team.fitness)
        print(champ.team.learners)
        print(champ.team.outcomes)
        print("---------------")
        ##complexity check: 
        #total_instruc = 0
        #for learner_x in champ.team.learners:
         #   total_instruc += len(learner_x.program.instructions)
        #champ.configFunctions()

        # Assuming RunBestAgent is a function you have defined earlier
        #empty array is: scorelist
        RunBestAgent((champ, []))