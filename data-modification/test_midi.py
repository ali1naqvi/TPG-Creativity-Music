import pygame
midi_filename = 'output_predictions.mid'

def pygame_in():
    pygame.mixer.init()
    # Load and play the MIDI file
    try:
        pygame.mixer.music.load(midi_filename)
        pygame.mixer.music.play()
        # Wait for the music to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # You can use a Clock to control the checking interval
    except pygame.error as e:
        print(f"Cannot play file {midi_filename}: {e}")
    finally:
        pygame.mixer.quit()


if __name__ == '__main__':
    pygame_in()