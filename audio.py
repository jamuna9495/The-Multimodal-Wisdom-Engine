import librosa
import numpy as np

def detect_emotional_peaks(audio_path, top_n=5):
    # Load the audio (using 22khz sample rate for efficiency)
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Calculate Root Mean Square (RMS) Energy
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # Smooth the signal to find sustained speech, not clicks
    window_size = 20 
    smoothed_rms = np.convolve(rms, np.ones(window_size)/window_size, mode='same')
    
    # Return the top energy timestamps
    peak_indices = np.argsort(smoothed_rms)[-top_n:]
    return sorted([times[i] for i in peak_indices])
