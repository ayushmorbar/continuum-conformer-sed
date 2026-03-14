import torch
from torch.utils.data import Dataset
import numpy as np

class DESEDDataset(Dataset):
    """
    Standard PyTorch Dataset for DESED.
    Supports a 'dummy_mode' for rapid pipeline verification and CI testing
    without needing the 23GB audio download.

    Label shape:
      - dummy_mode: [T, num_classes] frame-level strong labels (SED format)
      - real mode (TODO): same shape, loaded from DESED .npy strong-label arrays
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.dummy_mode = cfg['dataset']['dummy_mode']
        
        if self.dummy_mode:
            self.num_samples = cfg['dataset']['num_samples']
            self.seq_len     = cfg['dataset']['seq_len']
            self.mel_bins    = cfg['dataset']['mel_bins']
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
            # Simulated Log-Mel Spectrogram: [Time, Frequency]
            features = torch.randn(self.seq_len, self.mel_bins, dtype=torch.float32)

            # Frame-level strong labels: [Time, num_classes]
            # Each class is active for a random contiguous segment,
            # simulating a realistic onset/offset event pattern.
            label = torch.zeros(self.seq_len, self.num_classes, dtype=torch.float32)
            cls_idx = np.random.randint(0, self.num_classes)
            onset   = np.random.randint(0, self.seq_len // 2)
            offset  = np.random.randint(onset + 1, self.seq_len)
            label[onset:offset, cls_idx] = 1.0

            return features, label
        
        # TODO (Final Project): return np.load(file_path), strong_label_array
        pass
