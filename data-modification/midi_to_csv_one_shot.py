import csv
import numpy as np
import pretty_midi

#midi -> csv
def encoding(midi_filename, csv_filename):
    # Load the MIDI file
    midi = pretty_midi.PrettyMIDI(midi_filename)
    
    instrument = midi.instruments[0]

    start_times = sorted([note.start for note in instrument.notes])
    end_times = sorted([note.end for note in instrument.notes])

    # Calculate the smallest time difference (dynamic time step)
    all_times = sorted(set(start_times + end_times))
    time_differences = np.diff(all_times)
    time_step = min(time_differences) if len(time_differences) > 0 else 0.25

    total_time = midi.get_end_time()
    num_steps = int(np.ceil(total_time / time_step))

    # Find unique pitches used in the song
    used_pitches = sorted({note.pitch for note in instrument.notes})
    num_pitches = len(used_pitches)
    
    roll = np.zeros((num_steps, num_pitches), dtype=int) 

    pitch_to_index = {pitch: idx for idx, pitch in enumerate(used_pitches)}

    for note in instrument.notes:
        start_step = int(np.floor(note.start / time_step))
        end_step = int(np.ceil(note.end / time_step))
        
        for step in range(start_step, end_step):
            roll[step, pitch_to_index[note.pitch]] = 1

    final_roll = []
    for i in range(num_steps):
        final_roll.append([time_step] + list(roll[i]))

    header = ["time_difference"] + [f"pitch_{pitch}" for pitch in used_pitches]

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(final_roll)
    
    print(f"CSV data saved to {csv_filename}")
    
#csv -> midi
def decoding(csv_filename, midi_filename):
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  

        roll = []
        time_differences = []

        for row in reader:
            time_differences.append(float(row[0])) 
            roll.append([int(value) for value in row[1:]]) 
        roll = np.array(roll)

    used_pitches = [int(col.split('_')[1]) for col in header[1:]]

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  

    active_notes = {}
    current_time = 0.0

    for i in range(len(roll)):
        time_step = time_differences[i]

        for pitch_idx, pitch in enumerate(used_pitches):
            if roll[i, pitch_idx] == 1:
                if pitch in active_notes:
                    active_notes[pitch].end = current_time + time_step
                else:
                    # Start a new note
                    note = pretty_midi.Note(
                        velocity=100, 
                        pitch=pitch,
                        start=current_time,
                        end=current_time + time_step
                    )
                    active_notes[pitch] = note
                    instrument.notes.append(note)
            else:
                if pitch in active_notes:
                    del active_notes[pitch]

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
    
    