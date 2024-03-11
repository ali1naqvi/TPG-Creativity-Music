import pretty_midi
import pandas as pd 

inputname = "input.csv"
midi_file = "test_files/bach.mid"
import matplotlib.pyplot as plt
import numpy as np

def get_rest_after_n_seconds(midi_file, seconds):
    # Load the MIDI file
    original_midi = pretty_midi.PrettyMIDI(midi_file)
    
    # Create a new PrettyMIDI object for the rest of the piece after n seconds
    new_midi = pretty_midi.PrettyMIDI()

    for instrument in original_midi.instruments:
        # Create a new instrument object
        new_instrument = pretty_midi.Instrument(program=instrument.program)

        for note in instrument.notes:
            # Check if the note starts after n seconds
            if note.start > seconds:
                new_instrument.notes.append(note)
        
        new_midi.instruments.append(new_instrument)
    
    return new_midi

def get_first_n_seconds(midi_file, seconds):
    # Load the MIDI file
    original_midi = pretty_midi.PrettyMIDI(midi_file)
    
    # Create a new PrettyMIDI object
    new_midi = pretty_midi.PrettyMIDI()

    # Go through each instrument
    for instrument in original_midi.instruments:
        # Create a new instrument object
        new_instrument = pretty_midi.Instrument(program=instrument.program)

        for note in instrument.notes:
            # Check if the note is within the first n seconds
            if note.start < seconds:
                if note.end > seconds:
                    note.end = seconds
                new_instrument.notes.append(note)
        
        # Add the instrument to the new MIDI object
        new_midi.instruments.append(new_instrument)
    # Return the new MIDI object
    return new_midi

def get_max_pitch_length(df):
    # Find the maximum length of any pitch list in the DataFrame
    return max(df['pitch'].apply(len))

def pad_pitches(pitch_list, max_length, pad_value=-1):
    # Pad the pitch list to the maximum length found
    return pitch_list + [pad_value] * (max_length - len(pitch_list))


def plot_pitches(input_csv):
    df = pd.read_csv(input_csv)

    df.replace(-1, np.nan, inplace=True)
    # Create a new figure and axis for the plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Extract the column names for pitch columns only
    pitch_columns = [col for col in df.columns if col.startswith('pitch_')]

    # Plot each pitch column
    for col in pitch_columns:
        df[col].plot(ax=ax, label=col)

    # Set plot title and labels
    plt.title('Pitches over Time')
    plt.xlabel('Time (offset)')
    plt.ylabel('Pitch')

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()

def data_extraction(midi_data):
    df = pd.DataFrame()
    for i in range(0, len(midi_data.instruments)):
        all_notes = midi_data.instruments[i].notes

            # Sort notes by their start time
        all_notes.sort(key=lambda note: note.start)

            # Initialize the list to store features and variable to store the end time of the last note
        features = []
                # Iterate over sorted notes to extract features
        for note in all_notes:
                # Calculate offset as the difference between the start of this note and the end of the last
                # Offset for the first note will be its start time since there's no previous note
            offset = note.start 
                # Note pitch
            pitch = note.pitch
                
                # Note duration in ticks
            duration_ticks = midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start)

                # Convert the duration to 'parts per quarter note' (ppq)
            duration_ppq = duration_ticks / midi_data.resolution

                # Store the features in the list
            features.append({
                'offset': offset,
                'pitch': pitch,
                'duration_ppq': duration_ppq,
            })
                    # Convert the features list for the current instrument to a DataFrame
        instrument_df = pd.DataFrame(features)
        
        # Append the current instrument's DataFrame to the full DataFrame
        df = df._append(instrument_df, ignore_index=True)

    df = df.sort_values(by=['offset'])

    df = df[['offset', 'pitch', 'duration_ppq']]
    df = df.groupby(['offset', 'duration_ppq'])['pitch'].apply(list).reset_index()
    
    max_pitch_length = get_max_pitch_length(df)
    df['pitch'] = df['pitch'].apply(lambda x: pad_pitches(x, max_pitch_length))

    # Display the new DataFrame with combined pitches.
    df.to_csv(inputname, index=False)

    # New section to separate pitches into their own columns
    # We need to get the maximum number of pitches in any given offset
    max_pitches = df['pitch'].apply(lambda x: len(x)).max()

    # Create new columns for each pitch
    for i in range(max_pitches):
        df[f'pitch_{i+1}'] = df['pitch'].apply(lambda x: x[i] if i < len(x) else -1)
    
    # Drop the original 'pitch' column as we now have separate columns
    df.drop('pitch', axis=1, inplace=True)
    
    # Save the DataFrame to CSV
    df.to_csv(inputname, index=False)
    
    print("CSV file saved.")

# Call plot_pitches function after data_extraction
if __name__ == '__main__':
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    data_extraction(midi_data)
    plot_pitches(inputname)