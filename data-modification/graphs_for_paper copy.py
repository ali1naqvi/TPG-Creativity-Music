import pandas as pd
import matplotlib.pyplot as plt

# Load the three CSV files
csv_files = ["input.csv", "Simulation_49.csv", "Simulation_50.csv", "Simulation_51.csv", "Simulation_52.csv"]
df = [pd.read_csv(file) for file in csv_files]

#DO NOT TOUCH 21
#21 = RMSE OR MSE
#24 = THEIL
#31 = NCD 

# Plot the 'pitch' column from each dataframe
plt.figure(figsize=(10, 6))
df[0] = df[0].iloc[930:1050]
df[1] = df[1].iloc[930:1050]
df[2] = df[2].iloc[930:1050]
df[3] = df[3].iloc[930:1050]
df[4] = df[4].iloc[930:1050]

plt.plot(df[0]['pitch'], label=f'Target values')
plt.plot(df[1]['pitch'], label=f'49')
plt.plot(df[2]['pitch'], label=f'50')
plt.plot(df[3]['pitch'], label=f'51')
plt.plot(df[4]['pitch'], label=f'52')


plt.xlabel('Index')
plt.ylabel('Pitch')
plt.title('Validation Data of \'Fur Elise\'')
plt.legend()
plt.show()