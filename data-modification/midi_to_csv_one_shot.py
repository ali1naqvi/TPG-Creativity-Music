import csv
import numpy as np
import pretty_midi



#midi -> csv
def encoding(midi_filename, csv_filename):
    # Load the MIDI file
    midi = pretty_midi.PrettyMIDI(midi_filename)
    
    # Assumptions: We're dealing with a single instrument (e.g., piano)
    instrument = midi.instruments[0]

    # Extract note start and end times
    start_times = sorted([note.start for note in instrument.notes])
    end_times = sorted([note.end for note in instrument.notes])

    # Calculate the smallest time difference (dynamic time step)
    all_times = sorted(set(start_times + end_times))
    time_differences = np.diff(all_times)
    time_step = min(time_differences) if len(time_differences) > 0 else 0.25

    # Find the total time in the MIDI file to calculate the number of steps
    total_time = midi.get_end_time()
    num_steps = int(np.ceil(total_time / time_step))

    # Find unique pitches used in the song
    used_pitches = sorted({note.pitch for note in instrument.notes})
    num_pitches = len(used_pitches)
    
    # Initialize the roll with zeros (no active notes)
    roll = np.zeros((num_steps, num_pitches), dtype=int)  # Use dtype=int to ensure integer type

    # Create a pitch to index map for easy lookup
    pitch_to_index = {pitch: idx for idx, pitch in enumerate(used_pitches)}

    # Iterate over each note in the instrument
    for note in instrument.notes:
        start_step = int(np.floor(note.start / time_step))
        end_step = int(np.ceil(note.end / time_step))
        
        # Set the note on the roll matrix
        for step in range(start_step, end_step):
            roll[step, pitch_to_index[note.pitch]] = 1

    # Insert pauses if there are any rows that are all zeros
    final_roll = []
    for i in range(num_steps):
        final_roll.append([time_step] + list(roll[i]))

    # Prepare the header
    header = ["time_difference"] + [f"pitch_{pitch}" for pitch in used_pitches]

    # Write the roll to a CSV file with the header
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(final_roll)
    
    print(f"CSV data saved to {csv_filename}")
#csv -> midi
def decoding(csv_filename, midi_filename):
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header

        # Initialize list to store time differences and note data
        roll = []
        time_differences = []

        for row in reader:
            time_differences.append(float(row[0]))  # Extract time difference from the first column
            roll.append([int(value) for value in row[1:]])  # Append the rest of the row as note data
        roll = np.array(roll)

    # Identify the unique pitches used in the song (from the header)
    used_pitches = [int(col.split('_')[1]) for col in header[1:]]

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Using program 0 (Acoustic Grand Piano)

    # Initialize tracking for active notes and time
    active_notes = {}
    current_time = 0.0

    # Iterate through the note data and add notes to the instrument
    for i in range(len(roll)):
        time_step = time_differences[i]  # Get the time difference for the current step

        for pitch_idx, pitch in enumerate(used_pitches):
            if roll[i, pitch_idx] == 1:  # This is a note
                if pitch in active_notes:
                    # Extend the note's end time if it's already active
                    active_notes[pitch].end = current_time + time_step
                else:
                    # Start a new note
                    note = pretty_midi.Note(
                        velocity=100,  # Fixed velocity for simplicity
                        pitch=pitch,
                        start=current_time,
                        end=current_time + time_step
                    )
                    active_notes[pitch] = note
                    instrument.notes.append(note)
            else:
                if pitch in active_notes:
                    # End the note if it was active but not in this step
                    del active_notes[pitch]

        # Advance the current time by the time step
        current_time += time_step

    midi.instruments.append(instrument)
    midi.write(midi_filename)
    print(f"MIDI data saved to {midi_filename}")

if __name__ == '__main__':
    #midi_file = pretty_midi.PrettyMIDI(midi_file)
    inputname = "test_files/csv_files/reallyeasy_one_shot.csv"
    midi_file = "test_files/midi_files/reallyeasy.mid"
    
    encoding(midi_file, inputname)
    #decoding(inputname, 'test_files/midi_files/bach_one_shot_converted_back.mid')
    
    