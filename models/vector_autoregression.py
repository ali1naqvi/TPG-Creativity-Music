import pandas as pd
import ast  # To convert string representation of list to actual list
from statsmodels.tsa.api import ARIMA
from statsmodels.tsa.api import VAR

if __name__ == '__main__':
    # Load your data
    data = pd.read_csv('./input_uni.csv', nrows=1000)

    # Convert pitch column from string to list
    # data['pitch'] = data['pitch'].apply(ast.literal_eval)

    # Create separate columns for each pitch element
    # pitch_columns = ['pitch_1']
    # data[pitch] = pd.DataFrame(data['pitch'].tolist(), index=data.index)

    # Drop the original pitch column
    # data.drop('pitch', axis=1, inplace=True)

    # Now data is ready for VAR model
    print(data)

    #multivariate
    #constant_columns = [col for col in data.columns if data[col].nunique() == 1]
    #data = data.drop(columns=constant_columns)

    # Fitting the VAR model
    model = ARIMA(data, order=(5, 1, 0))
    fitted_model = model.fit()  # Using Akaike Information Criterion to find optimal lag

    forecasted_values = fitted_model.forecast(steps=160)
    
    # Round each forecasted value to 6 decimal points
    forecasted_values = [round(value) for value in forecasted_values]

    # Save the rounded forecasted values to a CSV file
    forecasted_df = pd.DataFrame(forecasted_values, columns=data.columns)

    combined_data = pd.concat([data, forecasted_df], ignore_index=True)

    # Save the combined data to a CSV file
    combined_data.to_csv('simulation_stats_model.csv', index=False)

    print("Forecasted values saved to simulation_var.csv")
