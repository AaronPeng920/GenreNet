import os
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as transforms
import lightning.pytorch as pl
import random
import torch.nn.functional as F
import torchaudio.functional as audio_F
from utils import split_window
import math

class AudioDataset(Dataset):
    def __init__(self, data_config, mode='train'):
        super().__init__()
        assert mode in ['train', 'val']
        if mode == 'train':
            self.data_dir = data_config['train_path']
        elif mode =='val':
            self.data_dir = data_config['val_path']
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.audio_files = self._load_audio_files()
        self.data_config = data_config

    def _load_audio_files(self):
        audio_files = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for file in os.listdir(class_dir):
                audio_files.append((os.path.join(class_dir, file), class_name))
        random.shuffle(audio_files)
        return audio_files

    def _audio_enhancement(self, waveform):
        # add noise
        noise = torch.randn_like(waveform) * 0.02
        waveform = waveform + noise

        return waveform
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file, class_name = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.data_config['sr'])
        # stereo to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0)
        waveform = waveform.squeeze()
        target_length = int(self.data_config['sr'] * self.data_config['duration'])
        waveform, n = split_window(waveform, int(target_length), int(0.5 * target_length))

        return waveform, torch.tensor([self.class_to_idx[class_name]] * n)

class MelSpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(self.data_config, mode='train')
        self.val_dataset = AudioDataset(self.data_config, mode='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.data_config['train_batchsize'], 
                          num_workers=10, collate_fn=self.collate_fn, pin_memory=True, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.data_config['val_batchsize'], 
                          num_workers=10, collate_fn=self.collate_fn, pin_memory=True, shuffle=False)
        

    def collate_fn(self, batch):
        waveforms, labels = zip(*batch)
        mel_transform = transforms.MelSpectrogram(
            sample_rate = self.data_config['sr'],
            n_fft = self.data_config['n_fft'],
            n_mels = self.data_config['n_mels']
        )
        mel_db = transforms.AmplitudeToDB()
        
        mel_spectrograms = []
        for waveform in waveforms:      # [n, seq_len]
            mel_spectrogram = mel_transform(waveform)   # [n, mel, f_len]
            mel_spectrogram = mel_db(mel_spectrogram)
            mel_spectrograms.append(mel_spectrogram)
        batch_mel_spectrograms = torch.concat(mel_spectrograms, dim=0)
        batch_labels = torch.concat(labels, dim=0)
        batch_waveforms = torch.concat(waveforms, dim=0)
        return batch_waveforms, batch_mel_spectrograms, batch_labels
