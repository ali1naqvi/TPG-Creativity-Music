import pretty_midi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#plot individual graphs or extract from a midi
inputname = "simulation_31.csv"
midi_file = "test_files/bach.mid"

def data_extraction(midi_data):
    df = pd.DataFrame()
    for instrument in midi_data.instruments:
        all_notes = instrument.notes
        # Sort notes by their start time
        all_notes.sort(key=lambda note: note.start)

        features = []
        for note in all_notes:
            offset = note.start
            pitch = [note.pitch] 
            duration_ticks = midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start)
            duration_ppq = duration_ticks / midi_data.resolution
            features.append({
                'offset': offset,
                'duration_ppq': duration_ppq,
                'pitch': pitch, 
            })

        instrument_df = pd.DataFrame(features)
        df = pd.concat([df, instrument_df], ignore_index=True)

    df.to_csv(inputname, index=False)
    print("CSV file saved.")

def plot_pitches(input_csv):
    df = pd.read_csv(input_csv)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    df.reset_index().plot(kind='line', y='duration_ppq', ax=ax, legend=False, linewidth=0.8)

    plt.title("Pitch Duration Representation of 'Fur Elise'", fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Pitch Duration (seconds)', fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()
 
    
if __name__ == '__main__':
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    #data_extraction(midi_data)
    plot_pitches(inputname)
