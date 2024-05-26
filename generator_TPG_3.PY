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
import utils
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

#values that can be modified for testing 
MAX_STEPS_G = 951 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 52900
EXTRA_TIME_STEPS  = 300 #number of wanted generated values 
STARTING_STEP = 0 #starting step

PRIME_STEPS = 50
TRAINING_STEPS = 50
VALIDATION_STEPS = TESTING_STEPS = 100

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

# full_song_bytes = training_data.to_numpy().ravel().tobytes()
# c_full_song = len(zlib.compress(full_song_bytes))

epsilon = 1e-10#small number near zero

#reward function, using Normalized Compression Distance

def ncd(sample, target):
    # Normalize the sample
    sample = np.clip(sample, 0, 1)
    
    # Convert sample and target to bytes
    sample_bytes = sample.tobytes()
    target_bytes = target.tobytes()
    
    # Compress individual sequences and concatenated sequence
    #c_sample = len(compress_ppmz(sample_bytes))
    #c_target = len(compress_ppmz(target_bytes))
    c_sample = len(zlib.compress(sample_bytes))
    c_target = len(zlib.compress(target_bytes))
    
    #c_concatenated = len(compress_ppmz(sample_bytes + target_bytes))
    c_concatenated = len(zlib.compress(sample_bytes + target_bytes))
    
    # Calculate NCD value
    ncd_value = (c_concatenated - min(c_sample, c_target)) / max(c_sample, c_target)
    
    return ncd_value + 100

def mse(sample, target):
    sample = np.clip(sample, 0, 1)
    sum_squared_error = 0
    for a, p in zip(target, sample):
        sum_squared_error += (a - p) ** 2
    mse = (sum_squared_error / len(sample))
    return mse

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
    def step(self, action_type, reward_func):

        #reset values 
        done = False
        reward = 0
            
        _, action_values = action_type
        modified_offset, modified_duration, modified_pitch_array = action_values

        # Clip the values to ensure they are within the valid range
        modified_offset = np.clip(modified_offset, 0, 1)
        modified_duration = np.clip(modified_duration, 0, 1)
        modified_pitch_array = np.clip(modified_pitch_array, 0, 1)  # Assuming modified_pitch_array can be directly used with np.clip
        
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)

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
                reward = -ncd(self.total_states.ravel(), self.total_true_states.ravel()) * 100
            elif reward_func == 'mse':
                reward = -mse(self.total_states.ravel(), self.total_true_states.ravel())
            #print("states: ", self.total_states, "reward: ", reward)
            done = True

        self.current_step +=1
        return self.last_state, predicted_state, reward, done

    def step_simulation(self, action_type):
        # stuff print(action_type)
        _, action_values = action_type
        modified_offset, modified_duration, modified_pitch_array = action_values

        # Clip the values to ensure they are within the valid range
        modified_offset = np.clip(modified_offset, 0, 1)
        modified_duration = np.clip(modified_duration, 0, 1)
        modified_pitch_array = np.clip(modified_pitch_array, 0, 1)  # Assuming modified_pitch_array can be directly used with np.clip
        
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        self.current_step +=1
        return predicted_state
    
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
    numEpisodes = int(MAX_STEPS_G / episode_length) # 500 / 100 = 5
    reward = 0

    for ep in range(numEpisodes):
        isDone = False
        action_state = env.reset(ep, episode_length) #resets at next 100 window (based on episode)
        
        predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        while True:
            #prime first half of the episode
            #if current step is less than the second half (0-49)
            if  env.current_step < ((ep * episode_length) + PRIME_STEPS):
                action_value = (agent.act(np.hstack(action_state)))
                env.current_step += 1 #increase step in environment
                action_state = env._get_state()
                predicted_state = action_state #updating predicted
            else:
                action_value = (agent.act(np.hstack(predicted_state)))
                #stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, reward_func='ncd') #now fix step 
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
            if ((ep * episode_length) + PRIME_STEPS) > env.current_step:
                action_value = (agent.act(np.hstack(action_state)))
                env.current_step +=1 #increase step in environment
                action_state = env._get_state()
            #second half of episode
            else:
                predicted_state = (agent.act(action_state))
                action_state, predicted_state, reward, isDone = env.step(predicted_state, reward_func='mse') #now fix step
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

    #funcion for handling parallelism
    agent.configFunctionsSelf()

    env = TimeSeriesEnvironment(current_step_g=49)
    scoreTotal = 0
    episode_length = PRIME_STEPS + TESTING_STEPS #50 + 100 = 150
    numEpisodes = int((MAX_STEPS_G - PRIME_STEPS) / episode_length) # 750 / 150 = 5
    reward = 0

    for ep in range(numEpisodes):
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
            if ((ep * episode_length) + PRIME_STEPS) > env.current_step:
                action_value = (agent.act(np.hstack(action_state)))
                env.current_step +=1 #increase step in environment
                action_state = env._get_state()
            #second half of episode
            else:
                predicted_state = (agent.act(action_state))
                action_state, predicted_state, reward, isDone = env.step(predicted_state, reward_func='mse') #now fix step
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

    # stuff print("Priming")
    # 0 to 799 (inclusive)
    for x in range(MAX_STEPS_G):
        row = changed_data.iloc[x]
        last_state = (row['offset'], row['duration_ppq'], row['pitch'])
        action_value = (agent.act(np.hstack(last_state)))
        print("step "+str(x)+" for state: ", np.hstack(last_state))

    #start at 800
    env = TimeSeriesEnvironment()
    print("what the last state is:", env.last_state)
    print("Priming last state (offset) to start from: ", utils.invert_min_max_scale(last_state[2], min_pitch, max_pitch))
    print("Priming last state (duration) to start from: ", utils.invert_min_max_scale(last_state[1], prescaled_duration_min, prescaled_duration_max))

    print("Priming complete")
    for x in range(EXTRA_TIME_STEPS-1): 
        print("FORECASTING: step "+str(x)+" for state: ", np.hstack((last_state)))
        action_value = (agent.act(np.hstack(last_state)))
        #print("action value:", action_value)
        action_state = env.step_simulation(action_value)
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
    simulated_data.to_csv('Simulation_30.csv', index=False)

if __name__ == '__main__':
    tStart = time.time()
    trainer_checkpoint_path = Path("trainer_savepoint_30.pkl")
    gen_checkpoint_path = Path("gen_savepoint_30.txt")

    if trainer_checkpoint_path.exists():
        trainer = loadTrainer(trainer_checkpoint_path)
        print("LOADED TRAINER")
    else:
        trainer = Trainer(actions=[3], teamPopSize=360, initMaxTeamSize=10, initMaxProgSize=100, pActAtom=0.95, memType="default", operationSet="def")
        gen_start = 0
    
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, 'r') as file:
            gen_start = int(file.read().strip())  # Read the number and convert it to an integer
        print("LOADED GEN NUMBER: ", gen_start)
    else:
        gen_start = 0

    # Open a text file to write output
    with open('results_30.txt', 'a' if gen_start > 0 else 'w') as file:
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
            champ.saveToFile("best_agent_30")

            trainer.evolve(tasks=['main'])
            
            validation_champion_path = Path("validation_champion_30")
            testing_champion_path = Path("testing_champion_30")
            if gen % 10 == 0 and gen != 0 and gen % 100 != 0:  # Validation phase every 10 generations but not on 100th
                prevbestscore = float('-inf') #starting value of negative infinity
                looper = True
                start_validation_time = time.time()
                print("Values")
                while looper:
                    validationScores = man.list()
                    agents = trainer.getAgents()
                    
                    pool.map(RunValidationAgents, [(agent, validationScores) for agent in agents])
                    teams1 = trainer.applyScores(validationScores)
                    
                    #the current best of this evolution
                    current_best_validation = trainer.getEliteAgent(task='validation')
                    print("Validation Generation Score: ", current_best_validation.team.outcomes['validation'])
                    #save and retrieve best validation agent (since the best of gen != best)
                    if current_best_validation.team.outcomes['validation'] >= prevbestscore:
                        prevbestscore = current_best_validation.team.outcomes['validation']
                        validationChamp = current_best_validation
                        validationChamp.saveToFile("validation_champion_30")
                    else: 
                        #error check just in case file does not exist 
                        if validation_champion_path.exists():
                            validationChamp = pickle.load(open(validation_champion_path, 'rb'))
                            validationChamp.configFunctionsSelf()
                        else: 
                            validationChamp = trainer.getEliteAgent(task='validation')
                        print("validationchampion: ", validationChamp.team.outcomes)
                        print(f"Validation champ with the best test score with {validationChamp.team.outcomes['validation']} on test data.")
                        with open("final_validation_scores_30.txt", 'w') as f:
                            f.write(str(validationChamp.team.outcomes['validation']))
                        looper= False
                            
                    if time.time() - start_validation_time > (3600*4):  # Check if 4 hour has passed
                        print("Time limit for finding a better validation champ exceeded.")
                        looper= False

                    if looper:
                        trainer.evolve(tasks=['validation'])
            
            if gen % 100 == 0 and gen != 0:  # testing phase every 10 generations
                prevbestscore = float('-inf') #starting value of negative infinity
                looper = True
                start_testing_time = time.time()
                print("Values")
                while looper:
                    TestingScores = man.list()
                    agents = trainer.getAgents()
                    
                    pool.map(RunTestingAgents, [(agent, TestingScores) for agent in agents])
                    teams1 = trainer.applyScores(TestingScores)
                    
                    #the current best of this evolution
                    current_best_testing = trainer.getEliteAgent(task='testing')
                    print("testing Generation Score: ", current_best_testing.team.outcomes['testing'])
                    #save and retrieve best testing agent (since the best of gen != best)
                    if current_best_testing.team.outcomes['testing'] >= prevbestscore:
                        prevbestscore = current_best_testing.team.outcomes['testing']
                        testingChamp = current_best_testing
                        testingChamp.saveToFile("testing_champion_30")
                    else: 
                        #error check just in case file does not exist 
                        if testing_champion_path.exists():
                            testingChamp = pickle.load(open(testing_champion_path, 'rb'))
                            testingChamp.configFunctionsSelf()
                        else: 
                            testingChamp = trainer.getEliteAgent(task='testing')
                        print("testingchampion: ", testingChamp.team.outcomes)
                        print(f"testing champ with the best test score with {testingChamp.team.outcomes['testing']} on test data.")
                        with open("final_testing_scores_30.txt", 'w') as f:
                            f.write(str(testingChamp.team.outcomes['testing']))
                        looper= False
                            
                    if time.time() - start_testing_time > (3600*4):  # Check if 4 hour has passed
                        print("Time limit for finding a better testing champ exceeded.")
                        looper= False

                    if looper:
                        trainer.evolve(tasks=['testing'])
                    
            scoreStats = trainer.fitnessStats
            allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
            print(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}")
            file.write(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}\n")

            trainer.saveToFile("trainer_savepoint_30.pkl")
            with open("gen_savepoint_30.txt", 'w') as gen_file:
                gen_file.write(str(gen))
            
            #to keep the champ saved in a file for evaluation later on 
        
        file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
        file.write('Final Results:\nMin, Max, Avg\n')
        for score in allScores:
            file.write(f"{score}\n")

        champ = pickle.load(open("testing_champion_30", 'rb'))
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