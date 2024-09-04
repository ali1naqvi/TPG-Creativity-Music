import pretty_midi
import pandas as pd 

#pretty stable
inputname = "test_files/csv_files/fur_elise.csv"
midi_file = "test_files/midi_files/fur_elise.mid"

def get_rest_after_n_seconds(midi_file, seconds):
    original_midi = pretty_midi.PrettyMIDI(midi_file)
    
    new_midi = pretty_midi.PrettyMIDI()

    for instrument in original_midi.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program)

        for note in instrument.notes:
            if note.start > seconds:
                new_instrument.notes.append(note)
        
        new_midi.instruments.append(new_instrument)
    
    return new_midi

def get_first_n_seconds(midi_file, seconds):
    original_midi = pretty_midi.PrettyMIDI(midi_file)
    
    new_midi = pretty_midi.PrettyMIDI()

    for instrument in original_midi.instruments:
        # Create a new instrument object
        new_instrument = pretty_midi.Instrument(program=instrument.program)

        for note in instrument.notes:
            if note.start < seconds:
                if note.end > seconds:
                    note.end = seconds
                new_instrument.notes.append(note)
        new_midi.instruments.append(new_instrument)
    return new_midi

def get_max_pitch_length(df):
    return max(df['pitch'].apply(len))

def pad_pitches(pitch_list, max_length, pad_value=-1):
    return pitch_list + [pad_value] * (max_length - len(pitch_list))


def data_extraction(midi_data):

    df = pd.DataFrame()
    for i in range(0, len(midi_data.instruments)):
        all_notes = midi_data.instruments[i].notes


        all_notes.sort(key=lambda note: note.start)

        features = []
        for note in all_notes:
                # Calculate offset as the difference between the start of this note and the end of the last
                # Offset for the first note will be its start time since there's no previous note
            offset = note.start 
            pitch = note.pitch
                
            duration_ticks = midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start)

                # Convert the duration to 'parts per quarter note' (ppq)
            duration_ppq = duration_ticks / midi_data.resolution

            features.append({
                'offset': offset,
                'pitch': pitch,
                'duration_ppq': duration_ppq,
            })
        instrument_df = pd.DataFrame(features)
        
        df = df._append(instrument_df, ignore_index=True)

    df = df[['offset', 'pitch', 'duration_ppq']]
    df = df.groupby(['offset', 'duration_ppq'])['pitch'].apply(list).reset_index()
    
    max_pitch_length = get_max_pitch_length(df)
    df['pitch'] = df['pitch'].apply(lambda x: pad_pitches(x, max_pitch_length))

    df.to_csv(inputname, index=False)

    print("CSV file saved.")


if __name__ == '__main__':

    midi_file = pretty_midi.PrettyMIDI(midi_file)
    data_extraction(midi_file)
