import pretty_midi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

inputname = "input.csv"
midi_file = "test_files/fur_elise.mid"

def data_extraction(midi_data):
    df = pd.DataFrame()
    for instrument in midi_data.instruments:
        all_notes = instrument.notes
        # Sort notes by their start time
        all_notes.sort(key=lambda note: note.start)

        features = []
        for note in all_notes:
            offset = note.start
            pitch = [note.pitch]  # Enclose pitch in a list
            duration_ticks = midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start)
            duration_ppq = duration_ticks / midi_data.resolution
            features.append({
                'offset': offset,
                'duration_ppq': duration_ppq,
                'pitch': pitch,  # Store pitch as a list
            })

        # Convert the features list for the current instrument to a DataFrame
        instrument_df = pd.DataFrame(features)
        # Append the current instrument's DataFrame to the full DataFrame
        df = pd.concat([df, instrument_df], ignore_index=True)

    df.to_csv(inputname, index=False)
    print("CSV file saved.")

def plot_pitches(input_csv):
    df = pd.read_csv(input_csv)
    df['pitch'] = df['pitch'].apply(lambda x: eval(x)[0] if isinstance(x, str) else x[0])

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Now that 'pitch' contains numeric data, we can plot it
    df.reset_index().plot(kind='line', x='offset', y='duration_ppq', ax=ax, legend=False, linewidth=0.8)

    # Set title and labels with increased font size
    plt.title("Pitch Duration Representation of 'Fur Elise'", fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Pitch Duration (seconds)', fontsize=14)
    
    # Set larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

    
if __name__ == '__main__':
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    data_extraction(midi_data)
    #plot_pitches(inputname)
