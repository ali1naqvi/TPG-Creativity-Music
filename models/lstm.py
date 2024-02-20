import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def preprocess_data(data):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

if __name__ == '__main__':
    # Load your data
    data = pd.read_csv('./input.csv')

    # Convert pitch column from string to list
    data['pitch'] = data['pitch'].apply(ast.literal_eval)

    # Create separate columns for each pitch element
    pitch_columns = ['pitch_' + str(i) for i in range(6)]
    data[pitch_columns] = pd.DataFrame(data['pitch'].tolist(), index=data.index)

    # Drop the original pitch column
    data.drop('pitch', axis=1, inplace=True)

    # Preprocess the data
    scaled_data, scaler = preprocess_data(data)

    # Reshape data for LSTM model
    # This step depends on how you want to structure your time series data
    # For example, you could use a sliding window approach

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(scaled_data.shape[1], scaled_data.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model (example parameters - adjust as needed)
    model.fit(scaled_data, epochs=100, batch_size=32)

    # Forecasting
    # This also depends on how you've structured your data
    forecasted_values = model.predict(scaled_data)

    # Inverse scale the forecasted values if necessary
    forecasted_values = scaler.inverse_transform(forecasted_values)

    # Round the forecasted values
    rounded_forecasted_values = [
        [round(value[0], 6) if i < 2 else round(value[0]) for i, value in enumerate(row)]
        for row in forecasted_values
    ]

    # Print the rounded forecasted values
    print("Rounded Forecasted Values:")
    for row in rounded_forecasted_values:
        print(row)