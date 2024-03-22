def min_max_scale(series, feature_range=(0, 1)):
    #min range + scale *(x - x_min) / x_max - x_min
    min_val = series.min()
    max_val = series.max()
    scale = feature_range[1] - feature_range[0]
    series_scaled = feature_range[0] + (series - min_val) * scale / (max_val - min_val)
    return series_scaled

def invert_min_max_scale(scaled_val, original_min, original_max, feature_range=(0, 1)):
    print(scaled_val)
    scale = feature_range[1] - feature_range[0]
    return ((scaled_val - feature_range[0]) * (original_max - original_min) / scale) + original_min

def compute_first_order_difference(series):
    return series.diff().fillna(0)  # Using fillna(0) for the first element which doesn't have a previous value