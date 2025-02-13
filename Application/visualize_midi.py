#Run this script to plot the MIDI notes over time: python visualize_midi.py


import pretty_midi
import matplotlib.pyplot as plt

# Load MIDI file
midi_path = "Application/Output/ReconVAT-Lemon_groundtruth.mid"
midi_data = pretty_midi.PrettyMIDI(midi_path)

# Convert MIDI to a piano roll
piano_roll = midi_data.instruments[0].get_piano_roll(fs=100)

# Plot the piano roll
plt.figure(figsize=(12, 6))
plt.imshow(piano_roll, aspect='auto', cmap='coolwarm', origin='lower')
plt.title("Piano Roll Visualization")
plt.xlabel("Time Frames")
plt.ylabel("MIDI Note Number")
plt.colorbar(label="Velocity")
plt.show()
