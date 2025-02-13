#If you want to convert the .mid file to a .wav or .mp3 file: python convert_midi.py


from midi2audio import FluidSynth

# Define input and output file paths
midi_path = "Application/Output/ReconVAT-Lemon_groundtruth.mid"
output_wav = "Application/Output/output.wav"

# Convert MIDI to WAV
fs = FluidSynth()
fs.midi_to_audio(midi_path, output_wav)

print(f"Converted {midi_path} to {output_wav}")
