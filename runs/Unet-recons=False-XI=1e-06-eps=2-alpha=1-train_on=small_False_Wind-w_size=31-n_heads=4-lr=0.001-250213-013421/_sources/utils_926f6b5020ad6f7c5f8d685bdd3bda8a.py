import sys
from functools import reduce

import torch
from PIL import Image
from torch.nn.modules.module import _addindent

import os
import librosa
import soundfile as sf
import xml.etree.ElementTree as ET
import pandas as pd
import glob

def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params) #[92m is green color, [0m is black color
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    onsets = (1 - (onsets.t() > onset_threshold).to(torch.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)
    
class Normalization():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""
    def __init__(self, mode='framewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                output = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                output[torch.isnan(output)]=0 # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                return (x-x_min)/(x_max-x_min)
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def transform(self, x):
        return self.normalize(x)


def convert_wav_to_flac(root_path):
    """Convert all .wav files in dataset structure to .flac (16kHz, mono)."""
    print(f"Checking WAV files in: {root_path}")  # Debugging

    for dataset in ['dataset1','dataset2', 'dataset3']:
        dataset_path = os.path.join(root_path, dataset)
        audio_folders = glob.glob(os.path.join(dataset_path, "**/audio"), recursive=True)

        if not audio_folders:
            print(f"No audio folders found in {dataset_path}")

        for audio_folder in audio_folders:
            print(f"Processing folder: {audio_folder}")

            wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))

            if not wav_files:
                print(f"No WAV files found in {audio_folder}")
                continue  # Skip if no WAV files

            for wav_file in wav_files:
                flac_file = wav_file.replace(".wav", ".flac")

                # Check if conversion is needed
                if os.path.exists(flac_file):
                    print(f"✅ Skipping {wav_file} (already converted)")
                    continue

                print(f"🎵 Converting: {wav_file} → {flac_file}")

                try:
                    # Read WAV
                    audio, sr = sf.read(wav_file)

                    # Convert to 16kHz mono if needed
                    if sr != 16000 or len(audio.shape) > 1:
                        print(f"🔄 Resampling {wav_file} to 16kHz mono")
                        sf.write(flac_file, audio.mean(axis=1) if len(audio.shape) > 1 else audio, 16000, format="FLAC")
                    else:
                        sf.write(flac_file, audio, sr, format="FLAC")

                    print(f"✅ Saved: {flac_file}")

                except Exception as e:
                    print(f"❌ Error converting {wav_file}: {e}")



def convert_wav_to_flac2(root_path):
    """ Convert all .wav files inside 'audio' folders recursively to .flac format (16kHz, Mono). """
    for dataset in ['dataset1', 'dataset2', 'dataset3']:
        dataset_path = os.path.join(root_path, dataset)

        #Finds audio/ inside all subdirectories
        # Find all 'audio' subdirectories inside dataset1, dataset2, dataset3
        audio_folders = [f for f in glob.glob(os.path.join(dataset_path, "**/audio"), recursive=True)]

        for audio_folder in audio_folders:
            print(f"Processing folder: {audio_folder}")
            for wav_file in glob.glob(os.path.join(audio_folder, "*.wav")):
                flac_file = wav_file.replace(".wav", ".flac")

                # Convert to 16kHz Mono
                y, sr = librosa.load(wav_file, sr=16000, mono=True)
                sf.write(flac_file, y, sr)

                print(f"Converted: {wav_file} → {flac_file}")

# Run conversion for all datasets
# convert_wav_to_flac('./IDMT-SMT-GUITAR_V2')


def convert_xml_to_tsv(root_path):
    """ Convert all .xml annotation files inside 'annotation' folders recursively to .tsv format. """
    print(f"Checking XML files in: {root_path}")  # Debugging
    for dataset in ['dataset1', 'dataset2', 'dataset3']:
        dataset_path = os.path.join(root_path, dataset)

        # Find all 'annotation' subdirectories inside dataset1, dataset2, dataset3
        annotation_folders = [f for f in glob.glob(os.path.join(dataset_path, "**/annotation"), recursive=True)]

        if not annotation_folders:
            print(f"No annotation folders found in {dataset_path}")

        for annotation_folder in annotation_folders:
            print(f"Processing folder: {annotation_folder}")

            xml_files = glob.glob(os.path.join(annotation_folder, "*.xml"))
            if not xml_files:
                    print(f"No XML files found in {annotation_folder}")
                    continue  # Skip if no XML files
            for xml_file in xml_files:
                tsv_file = xml_file.replace(".xml", ".tsv")
                print(f"Converting: {xml_file} → {tsv_file}")
                tree = ET.parse(xml_file)
                root = tree.getroot()

                events = []
                for event in root.findall(".//event"):
                    onset = float(event.find("onsetSec").text)
                    offset = float(event.find("offsetSec").text)
                    pitch = int(event.find("pitch").text)
                    velocity = 127  # Default velocity

                    events.append([onset, offset, pitch, velocity])

                df = pd.DataFrame(events, columns=["onset_time", "offset_time", "pitch", "velocity"])
                df.to_csv(tsv_file, sep="\t", index=False)

                print(f"Converted and Saved: {xml_file} → {tsv_file}")
                #return

# Run conversion for all datasets
# convert_xml_to_tsv('./IDMT-SMT-GUITAR_V2')


def delete_wav_xml_files(root_path):
    """Delete .wav and .xml files after converting to .flac and .tsv."""
    print(f"🗑 Searching for .wav and .xml files in: {root_path}")

    for dataset in ['dataset1', 'dataset2', 'dataset3']:
        dataset_path = os.path.join(root_path, dataset)
        
        # Find all annotation & audio folders
        annotation_folders = glob.glob(os.path.join(dataset_path, "**/annotation"), recursive=True)
        audio_folders = glob.glob(os.path.join(dataset_path, "**/audio"), recursive=True)

        # Delete XML files in annotation folders
        for annotation_folder in annotation_folders:
            xml_files = glob.glob(os.path.join(annotation_folder, "*.xml"))
            for xml_file in xml_files:
                try:
                    print(f"🗑 Deleting: {xml_file}")
                    os.remove(xml_file)
                except Exception as e:
                    print(f"❌ Error deleting {xml_file}: {e}")

        # Delete WAV files in audio folders
        for audio_folder in audio_folders:
            wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
            for wav_file in wav_files:
                flac_file = wav_file.replace(".wav", ".flac")
                
                # Ensure the corresponding .flac file exists before deleting .wav
                if os.path.exists(flac_file):
                    try:
                        print(f"🗑 Deleting: {wav_file}")
                        os.remove(wav_file)
                    except Exception as e:
                        print(f"❌ Error deleting {wav_file}: {e}")
                else:
                    print(f"⚠️ Skipping {wav_file}, no corresponding .flac found!")

#delete_wav_xml_files('./IDMT-SMT-GUITAR_V2')
