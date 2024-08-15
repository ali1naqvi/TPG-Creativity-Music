import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def produce_smooth(data, window_size):
    #Produces a trailing moving average of the input data.
    
    smoothed_data = data.rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

if __name__ == '__main__':
    # Load the data from CSV file
    original_file = pd.read_csv("input_uni.csv")
    
    smoothed_file = produce_smooth(original_file, 1)
    
    
    # Plot the original and smoothed data for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(original_file['pitch'], label=f'Original')
    plt.plot(smoothed_file['pitch'], label=f'Smoothed 2')
    
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Original and Smoothed Data')
    plt.legend()
    plt.show()
