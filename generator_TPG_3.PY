import numpy as np
import pandas as pd
from tpg.trainer import Trainer
from tpg.agent import Agent
import multiprocessing as mp
import time
import ast, pickle
import zlib
from pathlib import Path
from tpg.utils import getLearners, getTeams, learnerInstructionStats, actionInstructionStats, pathDepths


#values that can be modified for testing 
MAX_STEPS_G = 2600 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 2000
EXTRA_TIME_STEPS  = 450 #number of wanted generated values 
STARTING_STEP = 0 #starting step
DATA_DIVISION = 0.5
EP_LENGTH = 100

data1 = pd.read_csv("input.csv", header = 0)

MAX_PITCHES = data1['pitch'].apply(ast.literal_eval).apply(len).max() #max pitches played in a step for whole piece

def root_mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("The length of actual and predicted lists must be the same.")
    
    sum_squared_error = 0
    for a, p in zip(actual, predicted):
        sum_squared_error += (a - p) ** 2
    rmse = (sum_squared_error / len(actual)) ** 0.5
    return rmse 

#environment
class TimeSeriesEnvironment:
    def __init__(self, max_steps = MAX_STEPS_G, current_step_g=STARTING_STEP):

        self.max_generated_steps = max_steps
        self.current_step = current_step_g

        #starts off at 0, getting the 0th row in the dataset
        self.last_state = self._get_state()

    def reset(self, episodenum, window_size):
        # Resets to the starting step we want
        
        self.current_step = episodenum * window_size 

        #0 to 20, subtract 1  
        # stuff print("current step: ", self.current_step)
        self.max_generated_steps = (self.current_step + window_size)
        # stuff print("max generated step: ", self.max_generated_steps)
        self.last_state = self._get_state() 

        return self.last_state


#get predicted step and compare with the actual value
    
#first step: 0, last step:  19. len: 20s
    def step(self, action_type, forecasting=False):

        #reset values 
        done = False
        reward = 0
            
        #intuition is that if its the first atomic action, it will directly use the value, else (second atomic action), it will apply the change to the previous
        if action_type[0] == 0:
            modified_offset = action_type[1][0] #+ self.last_state[0]
            modified_duration = action_type[1][1] #+ self.last_state[1]
            modified_pitch_array = action_type[1][2:] #+ self.last_state[2]
            
        else: 
            modified_offset = action_type[1][0] + self.last_state[0]
            modified_duration = action_type[1][1] + self.last_state[1]
            modified_pitch_array = action_type[1][2:] + self.last_state[2]
        #clipping the values now: 
        modified_offset = max(modified_offset, self.last_state[0])
        modified_duration = max(modified_duration, 0)  # Second clipping if it's larger than previous offset value
        modified_pitch_array = np.clip(modified_pitch_array, -1, 127)  # Clip the second action

        # modified pitch array should be rounded
        modified_pitch_array = [round(p) for p in modified_pitch_array]
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        #check if we reached the end, calculate the reward 
                #0 - 19           #20 
        #get the next value to reward based on the predicted value
        if self.current_step+1 < self.max_generated_steps:
            actual_row = data1.iloc[self.current_step+1]
            self.real_state = (actual_row['offset'], actual_row['duration_ppq'], np.array(ast.literal_eval(actual_row['pitch'])))
            #reward = -calculate_ncd(self.real_state, predicted_state)
            reward = -root_mean_squared_error(np.hstack(self.real_state), np.hstack(predicted_state))
        else:
            #end of dataset
            done = True

        if forecasting == True:
            self.last_state = predicted_state
        else:
            self.last_state = self._get_state()
        self.current_step +=1
        return self.last_state, predicted_state, reward, done

    def step_simulation(self, action_type):
        # stuff print(action_type)
        
        if action_type[0] == 0:
            modified_offset = action_type[1][0] #+ self.last_state[0]
            modified_duration = action_type[1][1] #+ self.last_state[1]
            modified_pitch_array = action_type[1][2:] #+ self.last_state[2]
        else: 
            modified_offset = action_type[1][0] + self.last_state[0]
            modified_duration = action_type[1][1] + self.last_state[1]
            modified_pitch_array = action_type[1][2:] + self.last_state[2]
        
        #clipping the values now:
        modified_offset = max(modified_offset, self.last_state[0])
        modified_duration = max(modified_duration, 0)  # Second clipping if it's larger than previous offset value
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


def runAgent(args):
    agent, scoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()
    env = TimeSeriesEnvironment()
    scoreTotal = 0
    episode_length = EP_LENGTH
    numEpisodes = int(MAX_STEPS_G / episode_length) # 500 / 50 = 10
    reward = 0

    priming_steps = round(episode_length * DATA_DIVISION) # 100 * 0.5 = 50 priming steps
    for ep in range(numEpisodes):
        isDone = False
        #memory array is returned as this is the action state
        action_state = env.reset(ep, episode_length) #resets at next 25 window (based on episode)
        
        #starts on the second rather than the first so the first order difference be computed.
        #env.current_step += 1
        
        predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        while True:
            # change the action_state returning. Either use prediction (recursion) or direct approach 
            #prime first half of the episode
            # stuff  print("modifying: ", ((ep * episode_length) + priming_steps), "with episode: ", ep)
            if ((ep * episode_length) + priming_steps) > env.current_step:
                action_value = (agent.act(np.hstack((action_state[:2],action_state[2]))))
                env.current_step += 1 #increase step in environment
                action_state = env._get_state()
                predicted_state = action_state #updating predicted
            #second half of episode
            else:
                #print("forecasting with step: ", env.current_step, "with state: ", predicted_state)
                #after direct steps, it will take the predicted value produced then 
                predicted_state = np.hstack((predicted_state[:2],predicted_state[2]))
                action_value = (agent.act(predicted_state))
                # stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, forecasting=True) #now fix step 
                
            # Apply clipping to continuous actions : action_value 
            #action_state gives you the next step
            scoreEp += reward
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                # stuff print("we finished")
                break
        scoreTotal += scoreEp

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal)

    scoreList.append((agent.team.id, agent.team.outcomes))
    return agent, scoreList

def RunValidationAgents(args):
    agent, validscoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()

    env = TimeSeriesEnvironment()
    scoreTotal = 0
    episode_length = EP_LENGTH * 2 #100 
    numEpisodes = int(MAX_STEPS_G / episode_length) # 500 / 100 = 5
    reward = 0

    priming_steps = round(episode_length * DATA_DIVISION) # 100 * 0.5 = 50 PRIMING
    for ep in range(numEpisodes):
        counter_direct = 0
        #memory array is returned as this is the action state
        action_state = env.reset(ep, episode_length) #resets at next 25 window (based on episode)
        #predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        isDone = False
        while True:
            #action state is current value, memory based on this will return from get_action_mem (includes it)
            # change the action_state returning. Either use prediction (recursion) or direct approach 
            #prime first half of the episode
            #intuition: multiplication of episode and length gives us the starting step + priming steps gives us where the forecasting should start
            if ((ep * episode_length) + priming_steps) > env.current_step:
                action_value = (agent.act(np.hstack((action_state[:2],action_state[2]))))
                env.current_step +=1 #increase step in environment
                action_state = env._get_state()
            #second half of episode
            else:
                if counter_direct < (priming_steps / 2):
                    action_value = (agent.act(np.hstack((action_state[:2],action_state[2]))))
                    env.current_step +=1 #increase step in environment
                    action_state = env._get_state()
                    predicted_state = action_state
                    counter_direct +=1
                else:
                    predicted_state = np.hstack((predicted_state[:2],predicted_state[2]))
                    action_value = (agent.act(predicted_state))
                    action_state, predicted_state, reward, isDone = env.step(action_value, forecasting=True) #now fix step 
                
            # Apply clipping to continuous actions : action_value 
            #action_state gives you the next step
            scoreEp += reward
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                break
        scoreTotal += scoreEp

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task='validation')

    validscoreList.append((agent.team.id, agent.team.outcomes))
    return agent, validscoreList

def RunBestAgent(args):
    agent, scoreList = args
    # Initialize the environment but starting from the end of the data
    env = TimeSeriesEnvironment(current_step_g=MAX_STEPS_G-1)

    simulation_results = []

    # stuff print("Priming")
    for x in range(MAX_STEPS_G):
        row = data1.iloc[x]
        state = (row['offset'], row['duration_ppq'], np.array(ast.literal_eval(row['pitch'])))
        action_value = (agent.act(np.hstack((state[:2],state[2]))))

    for x in range(EXTRA_TIME_STEPS-1): 

        action_value = (agent.act(np.hstack((env.last_state[:2],env.last_state[2]))))

        action_state = env.step_simulation(action_value)
        # stuff print("action_state: ", action_state)

        new_last_state = (action_state[0], action_state[1], np.array2string(np.array(action_state[2:]), separator=', '))

        simulation_results.append(new_last_state)
        env.last_state = action_state

    new_data = pd.DataFrame(simulation_results, columns=['offset', 'duration_ppq', 'pitch'])
    new_data['pitch'] = new_data['pitch'].apply(str)    
    updated_data = pd.concat([data1[:MAX_STEPS_G], new_data], ignore_index=True)
    updated_data.to_csv('Simulation_13.csv', index=False)
    
    new_data = pd.DataFrame(simulation_results, columns=['offset', 'duration_ppq', 'pitch'])
    new_data['pitch'] = new_data['pitch'].apply(str)

if __name__ == '__main__':
    tStart = time.time()
    trainer_checkpoint_path = Path("trainer_savepoint_13.pkl")
    gen_checkpoint_path = Path("gen_savepoint_13.txt")

    if trainer_checkpoint_path.exists():
        trainer = pickle.load(open(trainer_checkpoint_path, 'rb'))
        trainer.configFunctions()
        print("LOADED TRAINER")
    else:
        #-159.676
        trainer = Trainer(actions=[MAX_PITCHES+2], teamPopSize=360, pActAtom=1.0, memType="default", operationSet="def")
        gen_start = 0
    
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, 'r') as file:
            gen_start = int(file.read().strip())  # Read the number and convert it to an integer
        print("LOADED GEN NUMBER: ", gen_start)
    else:
        gen_start = 0

    # Open a text file to write output
    with open('results_13.txt', 'a' if gen_start > 0 else 'w') as file:
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
            
            champ = trainer.getEliteAgent()
            champ.saveToFile("best_agent_13")

            trainer.evolve()
            
            if gen % 10 == 0 and gen != 0:  # Validation phase every 10 generations
                prevbestscore = float('-inf') #starting value of negative infinity
                looper = True
                start_validation_time = time.time()
                while looper == True:
                    validationScores = man.list()
                    agents = trainer.getAgents()
                    
                    pool.map(RunValidationAgents, [(agent, validationScores) for agent in agents])
                    teams1 = trainer.applyScores(validationScores)
                      
                    # Update: Do not break the loop immediately
                    if prevbestscore >= validationScores[0][1]['validation']:
                        validationChamp = trainer.getEliteAgent(task='validation')
                        finalTestScore = max(validationScores)
                        
                        print(f"Validation champ met/exceeded the best test score with {finalTestScore} on test data.")
                        validationChamp.saveToFile("validation_champion_13")
                        with open("final_validation_scores_13.txt", 'w') as f:
                            f.write(str(finalTestScore))
                        looper= False
                            
                    if time.time() - start_validation_time > (3600*4):  # Check if 4 hour has passed
                        print("Time limit for finding a better validation champ exceeded.")
                        looper= False

                    prevbestscore = validationScores[0][1]['validation']
                    trainer.evolve(tasks=['validation'])
                    
            scoreStats = trainer.fitnessStats
            allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
            print(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}")
            file.write(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}\n")

            trainer.saveToFile("trainer_savepoint_13.pkl")
            with open("gen_savepoint_13.txt", 'w') as gen_file:
                gen_file.write(str(gen))
            
            #to keep the champ saved in a file for evaluation later on 
        
        file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
        file.write('Final Results:\nMin, Max, Avg\n')
        for score in allScores:
            file.write(f"{score}\n")
        
        #champ = pickle.load(open("./output_files/best_agent", 'rb'))
        champ = trainer.getEliteAgent(task='validation')
        champ.configFunctionsSelf()

        # Assuming RunBestAgent is a function you have defined earlier
        #empty array is: scorelist
        
        RunBestAgent((champ, []))
