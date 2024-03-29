import pandas as pd
import pretty_midi
import ast  # Import Abstract Syntax Trees to safely parse strings into lists

def create_midi_from_csv(note_data, output_file, instrument_program=0):
    # Create a PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()
    # Create an Instrument instance with the specified program
    instrument = pretty_midi.Instrument(program=instrument_program)

    # Create notes from the data
    for note_entry in note_data:
        # Retrieve note information
        start_time = float(note_entry['offset'])  # Now 'offset' is the absolute start time
        pitch = int(note_entry['pitch'])  # Safely parse the string representation of a list
        duration_ppq = float(note_entry['duration_ppq'])

        # Iterate over pitches and create a Note object for each valid pitch
        if pitch != -1:  # Ignore -1, as it represents the absence of a note
            note = pretty_midi.Note(
                velocity=100,  # Default velocity
                pitch=pitch,
                start=start_time,
                end=start_time + duration_ppq  # Note's end time is its start time plus its duration
            )

                # Add the note to the instrument
            instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    pm.instruments.append(instrument)

    # Write out the MIDI data to the output file
    pm.write(output_file)

if __name__ == '__main__':
    df_notes = pd.read_csv('simulation_18.csv')

    # Convert the DataFrame to a list of dictionaries
    note_data = df_notes.to_dict('records')

    # Now you can pass this list to the function
    create_midi_from_csv(note_data, 'output_predictions.mid')