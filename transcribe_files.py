import pickle
import os
import numpy as np
from model import *
from librosa import midi_to_hz
import librosa

from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver

ex = Experiment('transcription')

def transcribe2midi(data, model, model_type, onset_threshold=0.5, frame_threshold=0.5, save_path=None, reconstruction=True, onset=True, pseudo_onset=False, rule='rule2', VAT=False): 
    for i in data:
        audio = i['audio']  # Extract audio from the input dictionary

        # Convert stereo to mono if necessary
        if len(audio.shape) > 1 and audio.shape[-1] == 2:
            print("# Converting stereo audio to mono")
            audio = torch.mean(audio, dim=-1)  # Average both channels


        # Ensure minimum required length (only pad if needed)
        min_required_length = model.spectrogram.n_fft // 2  # Minimum length required for STFT
        print(f"# Minimum length required for STFT: {min_required_length}")
        print(f"# audio.shape[-1]: {audio.shape[-1]}")
        print(f"# audio.shape: {audio.shape}")

        if audio.shape[-1] < min_required_length:
            print(f"# input audio is smaller: {audio.shape[-1] < min_required_length}")
            pad_length = min_required_length - audio.shape[-1]
            audio = F.pad(audio.unsqueeze(0), (0, pad_length), mode='constant').squeeze(0)  # Use constant padding
        else:
            print(f"# input audio is not smaller")
        print(f"# Input audio shape[batch_size, samples(2 channels (stereo) & 1 mono)] and requird mono: {audio.shape}")  # Debugging: Print input audio shape
        print(f"# Model n_fft: {model.spectrogram.n_fft}")
        
        pred = model.transcribe(i)
#         print(f"pred['onset2'] = {pred['onset2'].shape}")
#         print(f"pred['frame2'] = {pred['frame2'].shape}")            
        for key, value in pred.items():
            if key in ['frame','onset', 'frame2', 'onset2']:
                value.squeeze_(0).relu_() # remove batch dim and remove make sure no negative values
            p_est, i_est = extract_notes_wo_velocity(pred['onset'], pred['frame'], onset_threshold, frame_threshold, rule=rule)
        # print(f"p_ref = {p_ref}\n p_est = {p_est}")
        
        t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        # Converting time steps to seconds and midi number to frequency
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        midi_path = os.path.join(save_path, model_type+'-'+os.path.basename(i['path'])[:-4] + 'mid')
        print(f'midi_path = {midi_path}')
        save_midi(midi_path, p_est, i_est, [127]*len(p_est))
            
            
log=True
mode='imagewise'
spec='Mel'    
root = 'Application'
input_path = os.path.join(root, 'Input')
output_path = os.path.join(root, 'Output')

@ex.config
def config():
    device='cuda:0'
    model_type='ReconVAT'
    # instrument='string'

@ex.automain
def main(device, model_type):
    # Load audios from the Input files
    application_dataset = Application_Dataset(input_path,device=device)
    
    # Choose models
    if model_type=='ReconVAT':
        model = UNet((2,2),(2,2), log=log, reconstruction=True, mode=mode, spec=spec, device=device)
        weight_path = 'Weight/String_MusicNet/Unet_R_VAT-XI=1e-06-eps=1.3-String_MusicNet-lr=0.001/weight.pt'
    elif model_type=='baseline_Multi_Inst':
        model = Semantic_Segmentation(torch.empty(1,1,640,N_BINS), 1, device=device)
        weight_path = 'Weight/String_MusicNet/baseline_Multi_Inst/weight.pt'  
    
    # Load weights
    print(f'Loading model weight')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)    
    print(f'Loading done')


    audio_path = os.path.join(input_path, 'Lemon_groundtruth.flac')
    y, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio length: {len(y)} samples, Duration: {len(y)/sr:.2f} seconds")


    print(f'Transcribing Music')
    transcribe2midi(tqdm(application_dataset), model, model_type, reconstruction=False,
                save_path=output_path)            

             