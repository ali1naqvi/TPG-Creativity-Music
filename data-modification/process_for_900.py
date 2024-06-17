import pandas as pd

def process_csv(input_file, output_file, limit=900):
    # Read the CSV file
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

    # Concatenate all chunks
    processed_data = pd.concat(processed_data)

    # Save the resulting data to a new CSV file
    processed_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# Example usage
input_file = 'input.csv'
output_file = 'input_U.csv'
process_csv(input_file, output_file, limit=900)