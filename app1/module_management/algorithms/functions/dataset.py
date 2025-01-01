import collections
import os
import random

import numpy as np
import pandas as pd
import torch
# from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from app1.module_management.algorithms.functions.augment import WavAugment


def load_audio(filename, second=2):
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    # #调整音乐长度在90s内，不然显存会溢出
    # lim_len=100
    # if audio_length // sample_rate > lim_len:
    #     audio_length = lim_len * sample_rate
    # if second <= 0 and audio_length // sample_rate < lim_len:
    #     return waveform.astype(np.float64).copy()
    # start = np.int64(0)
    # waveform = waveform[start:start+audio_length].astype(np.float64)

    if second <= 0:
        return waveform.astype(np.float64).copy()
    length = np.int64(sample_rate * second)
    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        start = np.int64(random.random() * (audio_length - length))
        waveform = waveform[start:start + length].astype(np.float64)
    return waveform.copy()


class TrainDataset(Dataset):
    def __init__(self, train_csv_path, second=3, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.labels, self.paths = shuffle(self.labels, self.paths)
        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        print("Train Dataset load {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_1 = load_audio(self.paths[index], self.second)
        if self.aug:
            waveform_1 = self.wav_aug(waveform_1)
        if not self.pairs:
            return torch.FloatTensor(waveform_1), self.labels[index]

        else:
            waveform_2 = load_audio(self.paths[index], self.second)
            if self.aug:
                waveform_2 = self.wav_aug(waveform_2)
            return torch.FloatTensor(waveform_1), torch.FloatTensor(waveform_2), self.labels[index]

    def __len__(self):
        return len(self.paths)


class SemiDataset(Dataset):
    def __init__(self, label_csv_path, unlabel_csv_path, second=2, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(label_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values

        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        df = pd.read_csv(unlabel_csv_path)
        self.u_paths = df["utt_paths"].values
        self.u_paths_length = len(self.u_paths)

        if label_csv_path != unlabel_csv_path:
            self.labels, self.paths = shuffle(self.labels, self.paths)
            self.u_paths = shuffle(self.u_paths)

        # self.labels = self.labels[:self.u_paths_length]
        # self.paths = self.paths[:self.u_paths_length]
        print("Semi Dataset load {} speakers".format(len(set(self.labels))))
        print("Semi Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_l = load_audio(self.paths[index], self.second)

        idx = np.random.randint(0, self.u_paths_length)
        waveform_u_1 = load_audio(self.u_paths[idx], self.second)
        if self.aug:
            waveform_u_1 = self.wav_aug(waveform_u_1)

        if not self.pairs:
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1)

        else:
            waveform_u_2 = load_audio(self.u_paths[idx], self.second)
            if self.aug:
                waveform_u_2 = self.wav_aug(waveform_u_2)
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(
                waveform_u_1), torch.FloatTensor(waveform_u_2)

    def __len__(self):
        return len(self.paths)


class EvaluationDataset(Dataset):
    def __init__(self, paths, second=-1, **kwargs):
        self.paths = paths
        self.second = second
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        waveform = load_audio(self.paths[index], self.second)
        return torch.FloatTensor(waveform), self.paths[index]

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    dataset = TrainDataset(train_csv_path="data/train.csv", second=3)
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False
    )
    for x, label in loader:
        pass

