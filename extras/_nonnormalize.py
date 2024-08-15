#FOR PROFESSORS FORMATTED FILES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the reference file
reference_file = pd.read_csv("input.csv")
original_min = reference_file['pitch'].min()
original_max = reference_file['pitch'].max()

def main_conversion(scaled_val, original_min, original_max):
    feature_range = (0, 1)
    scale = feature_range[1] - feature_range[0]
    return ((scaled_val - feature_range[0]) * (original_max - original_min) / scale) + original_min

if __name__ == '__main__':
    # Read the file to normalize
    to_normalize_file = pd.read_csv("tpg_15_test_t200.csv")
    
    # Apply the conversion to the 'x0' column
    #to_normalize_file['x0'] = to_normalize_file['x0'].apply(lambda x: main_conversion(x, original_min, original_max))
    #to_normalize_file['y0'] = to_normalize_file['y0'].apply(lambda x: main_conversion(x, original_min, original_max))

# Plot the 'pitch' column from each dataframe
plt.figure(figsize=(10, 6))

plt.plot(reference_file['pitch'][:1100])
#plt.plot(to_normalize_file['y0'], label='Guesses Values')

plt.xlabel('Index')
plt.ylabel('Pitch')
plt.title('\'Ava Maria\'')
plt.legend()
plt.show()


    # Use rows after 1000 from the reference file
    #print(len(to_normalize_file['x0']))
    #reference_file_2 = reference_file.iloc[1000:1149]
    
    # Create a new DataFrame with the required columns
   # new_df = pd.DataFrame({
    #    'offset': reference_file_2['offset'].values,
    #    'duration_ppq': reference_file_2['duration_ppq'].values,
    #    'pitch': round(to_normalize_file['x0']).values
    #})

    # Save the result to a new CSV file
    #new_df.to_csv('kelly_t_1000.csv', index=False)