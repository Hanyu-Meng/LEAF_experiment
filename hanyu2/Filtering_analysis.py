import numpy as np
import tensorflow as tf
import os
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from network import impulse_responses

model = tf.keras.models.load_model('Hanyu_model/Speech_commands/LEAF')
# 1. Gabor 1D filtering frequency response
gabor_1D = model.layers[0].layers[0]
layer1_weights = gabor_1D.weights[0]
mu = layer1_weights[:, 0]
sigma = layer1_weights[:, 1]
fs = 16000
fc = (mu/pi)*fs/2
fc = np.array(fc)
bw = (fs/2)/sigma
bw = np.array(bw)
fc_df = pd.DataFrame(fc)
fc_df.to_csv('filtering_data/command_center_frequency.csv')
bw = np.array(sigma)
bw_df = pd.DataFrame(bw)
bw_df.to_csv('filtering_data/command_bw.csv')
# 2. Gaussian Lowpass Frequency Response
layer2 = model.layers[0].layers[2]
layer2_weight = layer2.weights[0]
sigma_n = np.array(layer2_weight)
sigma_n = np.reshape(sigma_n,[40])
sigma_df = pd.DataFrame(sigma_n)
sigma_df.to_csv('filtering_data/command_gaussian_sigma.csv')
W = 401
filter_size = W
gaussian = impulse_responses.gaussian_lowpass(layer2_weight,filter_size)
plt.figure()
n_filter = 40

freq = np.linspace(0,fs/2,num=round(W/2))
magnitude_dict = np.zeros([201,40])
for i in range(n_filter):
    amplitude_gaussian = abs(tf.signal.rfft(gaussian[0,:,i,0]))
    amplitude_gaussian = np.array(amplitude_gaussian)
    magnitude_dict[:,i] = amplitude_gaussian
    # plt.plot(freq,amplitude_gaussian[0:len(freq)])
df = pd.DataFrame(magnitude_dict)
df.to_csv("filtering_data/command_layer2_magnitude_response.csv", index = False, sep=',')

# plt.xlabel("Frequency (Hz)")
# plt.xlim(0,200)
# plt.ylabel('Amplitude')
# plt.title('Amplitude Response of Gaussian lowpass filters')
# plt.savefig("./output/filtering_analysis/out.png")