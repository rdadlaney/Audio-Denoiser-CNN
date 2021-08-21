##############################################################################
# This can live denoise Audio using Virtual Mic
# Install virtual mic using VB AUdio Cable
# Soundfile library will use the real mic of the device as its input
# It will output the denoised output to the virtual mic
# When using any video conferencing software, use this virtual mic as input
# Denoised audio will be sen to the video conference
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
import keyboard
from scipy.signal import butter, lfilter

debug_flag=False

windowLength=255
fftLength=255
hop_length=63
frame_length=8064


#model = keras.models.load_model('Unet_activation_relu_128x64_10_epochs.h5')
model = keras.models.load_model('New_FFT_Vox_Urban+background_128x128_60_epochs.h5')


CHUNK = 8064
RATE = 22050
# CHUNK = 2048
# RATE = 8000


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=7)
player = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK, output_device_index=12)
# player = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK, output_device_index=9 )

def convert_to_stft(data):
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
    predicted_stft = predicted_mag * data_stft_phase
    predicted_final = librosa.istft(predicted_stft ,hop_length=hop_length, length=frame_length)
    if debug_flag:
        print("Predicted final shape: ")
        print(predicted_final.shape)
    return(predicted_final)


def run_denoiser(actual_data):
    
    data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)    
    actual_data=data    
    if debug_flag:
        print("Data:")
        print(actual_data)

    data_stft_mag_db_scaled,data_stft_mag,data_stft_phase = convert_to_stft(actual_data)
    # Prediction
    before_time=int(time.time()*1000.0)
    predicted_clean=model.predict(data_stft_mag_db_scaled) 
    after_time=int(time.time()*1000.0)
    if debug_flag:
        print("Predicted: ")
        print(predicted_clean.shape)
    predicted_clean = np.reshape(predicted_clean, (predicted_clean.shape[1], predicted_clean.shape[2]))

    output_clean = convert_to_time_domain(predicted_clean,data_stft_phase,data_stft_mag)

    
    #######Optional: Supress background hum

    # for i in range(0,len(output_clean)):
    #     if abs(output_clean[i]) < 0.001:
    #         output_clean[i] = output_clean[i]/10
    #         output_clean[i] = 0
    if np.max(output_clean)<0.01:
        lo,hi=300,1000
        if np.max(output_clean)<0.005:
            lo,hi=1000,1500
        b,a=butter(N=6, Wn=[2*lo/RATE, 2*hi/RATE], btype='band')
        x = lfilter(b,a,output_clean)
        output_clean=np.float32(x)
    
    # cutoff=2000
    # b,a=butter(N=6, Wn=[2*cutoff/RATE], btype='low')
    # x = lfilter(b,a,output_clean)
    # output_clean=np.float32(x)

    lo,hi=50,2000
    b,a=butter(N=6, Wn=[2*lo/RATE, 2*hi/RATE], btype='band')
    x = lfilter(b,a,output_clean)
    output_clean=np.float32(x)

    if debug_flag:
        print("Output:")
        print(output_clean)
    if keyboard.is_pressed('f'):
        lo,hi=200,3000
        b,a=butter(N=6, Wn=[2*lo/RATE, 2*hi/RATE], btype='band')
        x = lfilter(b,a,output_clean)
        output_clean=np.float32(x)
        
    aft_time=int(time.time()*1000.0)
    if keyboard.is_pressed('u'):
        player.write(data,len(data))
    else:
        player.write(output_clean,len(output_clean))
    return actual_data

if __name__ == "__main__":

    print("\n\n\n")
    print("Audio Devices: ")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        a=p.get_device_info_by_host_api_device_index(0, i)
        print(a["name"] + " - " + str(a["index"]))
    print("Default input is: " + str(p.get_default_input_device_info()["name"]))
    actual_data = np.empty(shape=[0],dtype=np.float32)
    while True:        
        actual_data = run_denoiser(actual_data)
