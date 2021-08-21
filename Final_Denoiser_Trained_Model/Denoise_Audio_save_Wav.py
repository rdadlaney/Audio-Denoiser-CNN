##############################################################################
# The code will take a noisy audio and denoise it and save as a wav file
###############################################################################
import pyaudio
import numpy as np
import keras
import librosa
import librosa.display
import os
import glob
import scipy
import pyaudio
import numpy as np
import time
import soundfile as sf
from scipy.signal import butter, lfilter

windowLength=255
fftLength=255
hop_length=63
frame_length=8064
debug_flag = False
filtering_flag = True

#model = keras.models.load_model('models/Unet_Background_DB_norm_22k_128x64_10_epochs.h5')
model = keras.models.load_model('New_FFT_Vox_Urban+background_128x128_60_epochs.h5')

# noisy_file = 'test_audio+dog_bark.wav'
# noisy_file_name = 'noisy/' + noisy_file
# clean_file_name = 'predicted/predicted_final_new_fft_urban_' + noisy_file
noisy_file = 'Noisy_Audio_Sample_Roshan.wav'
noisy_file_name = '../Final_Output/' + noisy_file
clean_file_name = '../Final_Output/Denoised_Audio_Sample_Roshan.wav'

CHUNK = 8064
RATE = 22050
samp_interv=CHUNK

def convert_to_stft(data):
    # data_stft = librosa.stft(data, n_fft=fftLength, win_length=windowLength, hop_length=overlap, window=window, center=True)
    data_stft = librosa.stft(data, n_fft=fftLength, hop_length=hop_length)
    data_stft_mag, data_stft_phase =librosa.magphase(data_stft)
    if debug_flag:
        print("STFT shape:")
        print(data_stft_mag.shape)
    data_stft_mag_db = librosa.amplitude_to_db(data_stft_mag, ref=np.max)
    data_stft_mag_db_scaled = (data_stft_mag_db+80)/80
    data_stft_mag_db_scaled = np.reshape(data_stft_mag_db_scaled,(1,data_stft_mag_db_scaled.shape[0],data_stft_mag_db_scaled.shape[1],1))
    return data_stft_mag_db_scaled,data_stft_mag,data_stft_phase

def convert_to_time_domain(predicted_clean,data_stft_phase,data_stft_mag):
   
    predicted_mag_db_unscaled = (predicted_clean*80)-80
    predicted_mag = librosa.db_to_amplitude(predicted_mag_db_unscaled, ref=np.max(data_stft_mag))
    # predicted_sub = data_stft_mag - predicted_mag 
    predicted_stft = predicted_mag * data_stft_phase
    predicted_final = librosa.istft(predicted_stft ,hop_length=hop_length, length=frame_length)
    if debug_flag:
        print("Predicted final shape: ")
        print(predicted_final.shape)
    return(predicted_final)


def run_denoiser(noisy_sample):
        
    data_stft_mag_db_scaled,data_stft_mag,data_stft_phase = convert_to_stft(noisy_sample)

    predicted_clean = model.predict(data_stft_mag_db_scaled)
    # predicted_clean = data_stft_mag_db_scaled

    if debug_flag:
        print("Predicted: ")
        print(predicted_clean.shape)
    predicted_clean = np.reshape(predicted_clean, (predicted_clean.shape[1], predicted_clean.shape[2]))

    output_clean = convert_to_time_domain(predicted_clean,data_stft_phase,data_stft_mag)

    if filtering_flag:
        if np.max(output_clean)<0.01:
            lo,hi=300,1000
            if np.max(output_clean)<0.005:
                lo,hi=1000,1500
            b,a=butter(N=6, Wn=[2*lo/RATE, 2*hi/RATE], btype='band')
            x = lfilter(b,a,output_clean)
            output_clean=np.float32(x)

        lo,hi=50,2000
        b,a=butter(N=6, Wn=[2*lo/RATE, 2*hi/RATE], btype='band')
        x = lfilter(b,a,output_clean)
        output_clean=np.float32(x)
    
    return output_clean

if __name__ == "__main__":

    print("\n\n\n")

    noisy_sample_test_split=[]
    clean_audio_array=[]

    noisy_sample_test, noise_sample_sr = librosa.load(noisy_file_name,RATE)
    for j in range(samp_interv, len(noisy_sample_test), samp_interv):
        k=j-samp_interv
        noisy_sample_test_split.append(noisy_sample_test[k:j])

    noisy_sample_test_split = np.array(noisy_sample_test_split)
    print(noisy_sample_test_split.shape)

    for i in range(len(noisy_sample_test_split)):        
        clean_audio = run_denoiser(noisy_sample_test_split[i])
        clean_audio_array.append(clean_audio)

    clean_audio_array = np.array(clean_audio_array)
    clean_wav = np.reshape(clean_audio_array, (clean_audio_array.shape[0]*clean_audio_array.shape[1]))
    print("Output = " + str(clean_wav.shape))
    sf.write(clean_file_name,clean_wav,RATE)
