import pandas as pd
import ast  # To convert string representation of list to actual list
from statsmodels.tsa.api import VAR


if __name__ == '__main__':
    # Load your data
    data = pd.read_csv('test_files/csv_files/bach.csv', nrows=1000)

    # Convert pitch column from string to list
    #data['pitch'] = data['pitch'].apply(ast.literal_eval)

    # Create separate columns for each pitch element
    #pitch_columns = ['pitch_1']
    #data[pitch] = pd.DataFrame(data['pitch'].tolist(), index=data.index)

    # Drop the original pitch column
    #data.drop('pitch', axis=1, inplace=True)


    # Now data is ready for VAR model
    print(data)

    constant_columns = [col for col in data.columns if data[col].nunique() == 1]
    data = data.drop(columns=constant_columns)
    
    
        
    # Fitting the VAR model
    model = VAR(data)
    
    fitted_model = model.fit(maxlags=10, ic='aic')  # Using Akaike Information Criterion to find optimal lag

    # Forecasting the next 10 values
    forecasted_values = fitted_model.forecast(data.values[-fitted_model.k_ar:], steps=300)
        # Round each forecasted value to 6 decimal points
    forecasted_values = [
    [round(value, 6) if i < 2 else round(value) for i, value in enumerate(row)]
    for row in forecasted_values
    ]

    # Print the rounded forecasted values
    print("Rounded Forecasted Values:")
    for row in forecasted_values:
        print(row)