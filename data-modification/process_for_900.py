#this is for looking at sections and seeing what it forecasted (with Dr.Kelly implementation)
import pandas as pd

def process_csv(input_file, output_file, limit=900):
    data = pd.read_csv(input_file)

    processed_data = []

    # Iterate through the data in chunks
    for start in range(0, limit, 150):
        end = start + 150
        if end <= len(data):
            chunk = data.iloc[start + 50:end]
            processed_data.append(chunk)
        else:
            break

    # Concatenate chunks
    processed_data = pd.concat(processed_data)

    processed_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

input_file = 'input.csv'
output_file = 'input_U.csv'
process_csv(input_file, output_file, limit=900)