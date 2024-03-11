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
