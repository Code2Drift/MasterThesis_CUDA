
import pandas as pd
import numpy as np
from numpy import fft
from scipy.signal import butter, filtfilt



def butter_LPF_Viz(data: pd.DataFrame, cutoff, fs, order):

    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Create a new DataFrame for filtered data
    filtered_df = pd.DataFrame()

    for column in data.columns:
        column_data = data[column].values.reshape(-1)
        filtered_data = filtfilt(b, a, column_data)
        filtered_df[column] = filtered_data

    return filtered_df


def filt_FFT(data, n_predict):
    amplitude_list = []
    phase_list = []
    n = data.size

    # number of harmonics in model
    n_harm = 8
    t = np.arange(0, n)
    p = np.polyfit(t, data, 1)

    # find linear trend in x, detrended x in the frequency domain
    data_notrend = data - p[0] * t
    data_freqdom = fft.fft(data_notrend)

    # frequencies
    f = fft.fftfreq(n)
    indexes = list(range(n))

    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(data_freqdom[i]))
    indexes.reverse()
    j = 1
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)

    for i in indexes[:1 + n_harm * 2]:

        # Getting amplitude and phase
        amplitude = np.absolute(data_freqdom[i]) / n
        phase = np.angle(data_freqdom[i])
        a = amplitude * np.cos(2 * np.pi * f[i] * t + phase)

        if j % 4 == 0:
            amplitude_list.append(amplitude)
            phase_list.append(phase)

        restored_sig += a
        j += 1

    # Getting the maximum amplitude and corresponding phase
    max_amplitude = min(amplitude_list)
    max_index = amplitude_list.index(max_amplitude)
    max_phase = phase_list[max_index]

    return (restored_sig + p[0] * t, max_amplitude)