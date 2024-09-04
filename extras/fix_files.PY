#TO LISTEN TO FILES, WE MUST COMBINE THE ORIGINAL DURATION AND OFFSET WITH THE PREDICTED PITCH.
import pandas as pd

# Load the two CSV files
csv_files = ["tpg_15_test_t800.csv", "input.csv"]
df_simulation = pd.read_csv(csv_files[0])
df_input = pd.read_csv(csv_files[1])

new_df = pd.DataFrame({
    'offset': df_input['offset'],
    'duration_ppq': df_input['duration_ppq'],
    'pitch': df_simulation['pitch']
})

new_df.to_csv('sim31_playable.csv', index=False)

print(new_df)
