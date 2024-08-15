import pandas as pd
import pretty_midi

def create_midi_from_csv(note_data, output_file, instrument_program=0):
    # Create PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()
    # Create Instrument instance
    instrument = pretty_midi.Instrument(program=instrument_program)

    # Create notes from data
    for note_entry in note_data:
        # Retrieve note information
        start_time = float(note_entry['offset'])  # offset: absolute start time
        pitch = int(note_entry['pitch'])  # parse the string representation
        duration_ppq = float(note_entry['duration_ppq'])

        
        if pitch != -1: # -1 represents absence of note. Will not occur, however kept for multivariable purposes
            note = pretty_midi.Note(
                velocity=100, #velocity is ignored for this program (keep at 100)
                pitch=pitch,
                start=start_time,
                end=start_time + duration_ppq  # Note's end time: start time plus its duration
            )

                # Add the note to the instrument
            instrument.notes.append(note)

    # Add the instrument to PrettyMIDI object
    pm.instruments.append(instrument)
    pm.write(output_file) # Write to output file

if __name__ == '__main__':
    df_notes = pd.read_csv('sim31_playable.csv') #EDIT FOR CSV NAME
    note_data = df_notes.to_dict('records')
    create_midi_from_csv(note_data, 'file_alife2024_prime.mid') #EDIT FOR MIDI FILE NAME