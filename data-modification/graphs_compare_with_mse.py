import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mse(sample, target):
    sample = np.clip(sample, 0, 1)
    sum_squared_error = 0
    for a, p in zip(target, sample):
        sum_squared_error += (a - p) ** 2
    mse = (sum_squared_error / len(sample))
    return mse


# Load the four CSV files
csv_files = ["input_uni.csv", "checkpoints/Simulation_5.csv"]
df = [pd.read_csv(file) for file in csv_files]

# Trim the dataframes to the specified range
df = [df[i].iloc[] for i in range(2)]

# Calculate MSE for each simulation compared to the input
mse_values = []
mse1 = mse(df[1]['pitch'], df[0]['pitch'])
mse_values.append(mse1)

# Print the MSE values
for i, mse1 in enumerate(mse_values):
    print(f"MSE between input and simulation_{csv_files[i].split('_')[1].split('.')[0]}: {mse1}")

# Plot the 'pitch' column from each dataframe
plt.figure(figsize=(10, 6))

plt.plot(df[0]['pitch'], label='Binary representation (new)')
plt.plot(df[1]['pitch'], label='Old representation')

plt.xlabel('Index')
plt.ylabel('Pitch')
plt.title('Testing on sample song')
plt.legend()
plt.show()
