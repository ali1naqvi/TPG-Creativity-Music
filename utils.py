import numpy as np
import zlib
epsilon = 1e-10

def min_max_scale(series, feature_range=(0, 1)):
    #min range + scale *(x - x_min) / x_max - x_min
    min_val = series.min()
    max_val = series.max()
    scale = feature_range[1] - feature_range[0]
    series_scaled = feature_range[0] + (series - min_val) * scale / (max_val - min_val)
    return series_scaled

def invert_min_max_scale(scaled_val, original_min, original_max, feature_range=(0, 1)):
    scale = feature_range[1] - feature_range[0]
    return ((scaled_val - feature_range[0]) * (original_max - original_min) / scale) + original_min

def compute_first_order_difference(series):
    return series.diff().fillna(0)  # Using fillna(0) for the first element which doesn't have a previous value

def sigmoid(x):
    if x >= 0:  
        return 1 / (1 + np.exp(-x))
    elif x < 0:
        return np.exp(x) / (1 + np.exp(x))

def calc_correlation(y_true, y_pred):
    def correlation(y1, y2):
        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)
        numerator = np.sum((y1 - y1_mean) * (y2 - y2_mean))
        denominator = np.sqrt(np.sum((y1 - y1_mean) ** 2) * (np.sum((y2 - y2_mean) ** 2)))
        if denominator == 0:
            denominator = epsilon
        return numerator / denominator
    
    total_correlation = 0
    for start in range(3):
        y1_group = y_true[start::3]
        y2_group = y_pred[start::3]
        total_correlation += correlation(y1_group, y2_group)
    
    return total_correlation +100

def calc_correlation_2(y_true, y_pred):
    #here we focus on inter-dependencies where we capture the correlation between sequences rather than each task differently
    def correlation(y1, y2):
        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)
        numerator = np.sum((y1 - y1_mean) * (y2 - y2_mean))
        denominator = np.sqrt(np.sum((y1 - y1_mean) ** 2) * (np.sum((y2 - y2_mean) ** 2)))
        if denominator == 0:
            denominator = epsilon
        return numerator / denominator
    
    total_correlation = 0
    num_groups = 0
    
    for i in range(0, len(y_true), 3):
        if i + 3 <= len(y_true):
            y1_group = y_true[i:i+3]
            y2_group = y_pred[i:i+3]
            corr = correlation(y1_group, y2_group)
            total_correlation += corr
            num_groups += 1
    return (total_correlation / num_groups)

#FITNESS FUNCTIONS
def ncd(sample, target):
    #  #here we focus on inter-dependencies where we capture the correlation between sequences rather than each task differently
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
    
    return ncd_value

def ncd_2(y1, y2):
    
    def ncd_done(sample,target):# Normalize the sample
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

    ncd_val = 0 
    for start in range(3):
        y1_group = y1[start::3]
        y2_group = y2[start::3]
        ncd_val += ncd_done(y1_group, y2_group)
    
    return ncd_val

def mse(sample, target):
    sample = np.clip(sample, 0, 1)
    sum_squared_error = 0
    for a, p in zip(target, sample):
        sum_squared_error += (a - p) ** 2
    mse = (sum_squared_error / len(sample))
    return mse

def theil_u_statistic(actual_, predicted_):
    #actual has one more tuple (flattened tho) than predicted; since theils requires one future value. 
    def theils(actual, predicted): 
        numerator = 0
        denominator = 0 
        for p in range(len(predicted)):
            numerator += (actual[p] - predicted[p]) ** 2
            denominator += (actual[p] - actual[p+1]) ** 2
        return numerator/denominator
    
    total_theils = 0
    for start in range(3):
        y1_group = actual_[start::3]
        y2_group = predicted_[start::3]
        total_theils += theils(y1_group, y2_group)
        
    return total_theils


