import torch
from torch.utils.data import Dataset
import numpy as np


class NewsData(Dataset):
    def __init__(self, features, labels):
        """

        Args:
            features: filename for features
            labels: filename for labels
        """
        self.features = np.load(features, allow_pickle=True)
        self.labels = np.load(labels, allow_pickle=True)
        print(self.labels.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = torch.LongTensor(self.features[item])
        label = torch.Tensor([self.labels[item]])
        sample = {'feature': feature, 'label': label}
        return sample
