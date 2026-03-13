import torch
from torch.utils.data import Dataset
import numpy as np

class DESEDDataset(Dataset):
    """
    Standard PyTorch Dataset for DESED.
    Currently supports a 'dummy_mode' for rapid pipeline verification and CI testing
    without needing the 23GB audio download.
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.dummy_mode = cfg['dataset']['dummy_mode']
        
        if self.dummy_mode:
            # Generate deterministic synthetic data for pipeline testing
            self.num_samples = cfg['dataset']['num_samples']
            self.seq_len = cfg['dataset']['seq_len']
            self.mel_bins = cfg['dataset']['mel_bins']
            self.num_classes = cfg['model']['num_classes']
        else:
            # TODO (Final Project): Load actual DESED TSV metadata and .npy files here
            # self.meta_df = pd.read_csv(...)
            pass

    def __len__(self):
        if self.dummy_mode:
            return self.num_samples
        # return len(self.meta_df)

    def __getitem__(self, idx):
        if self.dummy_mode:
            # Simulated Log-Mel Spectrogram: (Time, Frequency)
            features = torch.randn(self.seq_len, self.mel_bins, dtype=torch.float32)
            # Simulated weak label (Multi-class binary vector)
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            label[np.random.randint(0, self.num_classes)] = 1.0 
            return features, label
        
        # TODO (Final Project): return np.load(file_path), actual_label
        pass
