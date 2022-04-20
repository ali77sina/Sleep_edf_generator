# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:40:31 2022

@author: lina3953
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter, sosfilt,iirfilter
from scipy.signal import butter, lfilter, sosfiltfilt, filtfilt

def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output ='sos')
    return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=6):
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

class GEN:
    def __init__(self, ind):
        self.folder = r'C:\Users\lina3953\Desktop\sleep_edf_dataset\sleep-edf-database-expanded-1.0.0\sleep-cassette'
        self.all_files = os.listdir(self.folder)
        self.name_rec = self.all_files[ind]
        self.name_hyp = self.all_files[ind+1]
        self.file_rec = os.path.join(self.folder, self.name_rec)
        self.file_hyp = os.path.join(self.folder, self.name_hyp)
    
    def load_hypnogram(self):
        anots = mne.read_annotations(self.file_hyp)
        hyp_stage = anots.description
        hyp_dur = anots.duration
        hyp_onset = anots.onset
        return hyp_stage, hyp_dur, hyp_onset
    
    def return_int_label(self):
        hyp_stage, hyp_dur, hyp_onset = self.load_hypnogram()
        l = hyp_onset[-1]
        l = int(l)*100  #sfreq = 100hz
        int_label = np.zeros(l)
        k = 0
        fs = 100
        for i,j in zip(hyp_onset, hyp_dur):
            if hyp_stage[k] == 'Sleep stage 4':
                stage = 0
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
            if hyp_stage[k] == 'Sleep stage 3':
                stage = 1
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
            if hyp_stage[k] == 'Sleep stage 2':
                stage = 2
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
            if hyp_stage[k] == 'Sleep stage 1':
                stage = 3
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
            if hyp_stage[k] == 'Sleep stage R':
                stage = 4
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
            if hyp_stage[k] == 'Sleep stage W':
                stage = 5
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
            else:
                stage = 6
                int_label[int(i*fs): int((i+j)*fs)] = stage
                k += 1
                continue
        return int_label, hyp_stage, hyp_dur, hyp_onset
    
    def load_data(self):
        psg_raw = mne.io.read_raw_edf(self.file_rec, verbose = 0)
        data = psg_raw.get_data()
        chans = psg_raw.ch_names
        info = psg_raw.info
        fs = info['sfreq']
        time = np.linspace(0, len(data[0])/fs, len(data[0]))
        return data
    
    def creat_epochs(self, num_of_epochs = 128, epoch_length = 504):
        data = self.load_data()
        int_label, hyp_stage, hyp_dur, hyp_onset = self.return_int_label()
        mu = len(data[0]) / 2
        sigma = 1
        rand_ind = np.random.normal(mu, sigma, num_of_epochs)
        epochs = np.zeros(shape = (num_of_epochs, 2, epoch_length))
        eeg_ind = [0,1]
        for num,i in enumerate(rand_ind):
            epochs[num] = data[eeg_ind, int(i):int(i+epoch_length)]
        return epochs
    
    def creat_labeled_epochs_single_channel(self, num_of_epochs = 128, epoch_length = 504, filt = True):
        hyp_stage, hyp_dur, hyp_onset = self.load_hypnogram()
        data = self.load_data()
        # int_label = self.return_int_label()
        epochs = np.zeros(shape = (num_of_epochs, epoch_length))
        epoch_labels = np.zeros(num_of_epochs)
        allowed_inds = np.arange(1, len(hyp_stage)-4, 1)
        for i in range(num_of_epochs):
            ind = np.random.random()*len(allowed_inds)
            ind = int(ind)
            ind_from_list = allowed_inds[ind]
            label = hyp_stage[ind_from_list]
            onset = hyp_onset[ind_from_list]
            if label == 'Sleep stage 4':
                stage = 0
            elif label == 'Sleep stage 3':
                stage = 1
            elif label == 'Sleep stage 2':
                stage = 2
            elif label == 'Sleep stage 1':
                stage = 3
            elif label == 'Sleep stage R':
                stage = 4
            elif label == 'Sleep stage W':
                stage = 5
            else:
                stage = 6
            epoch_labels[i] = stage
            epochs[i] = data[0, int(onset*100):int(onset*100 + epoch_length)]
            epochs[i] = butter_bandpass_filter_sos(epochs[i], 0.5, 40, fs = 100)
        return epochs, epoch_labels
    
    def creat_many_single_channel(self):
        pass
    
    
