import pandas as pd
import matplotlib.pyplot as plt

# Load the three CSV files
csv_files = ["simulation_24_check50.csv", "input.csv"]
df = [pd.read_csv(file) for file in csv_files]

#DO NOT TOUCH 21
#21 = RMSE OR MSE
#24 = THEIL
#31 = NCD 

# Plot the 'pitch' column from each dataframe
plt.figure(figsize=(10, 6))
df[0] = df[0].iloc[900:]
df[1] = df[1].iloc[900:]

plt.plot(df[1]['pitch'], label=f'Target values')
plt.plot(df[0]['pitch'], label=f'Var')


plt.xlabel('Index')
plt.ylabel('Pitch')
plt.title('Validation Data of \'Fur Elise\'')
plt.legend()
plt.show()