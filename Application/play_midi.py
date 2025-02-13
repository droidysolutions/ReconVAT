
#Run this Python script to play the .mid file: python play_midi.py

import pygame
# Initialize pygame mixer
pygame.init()
pygame.mixer.init()

# Load and play MIDI file
midi_path = "Application/Output/ReconVAT-Lemon_groundtruth.mid"
pygame.mixer.music.load(midi_path)
pygame.mixer.music.play()

print("Playing MIDI file... Press CTRL+C to stop.")

# Keep the program running while music plays
while pygame.mixer.music.get_busy():
    continue
