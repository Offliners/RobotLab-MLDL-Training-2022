from PIL import torch
from torch.utils.data import Dataset, Subset


class PseudoDataset(Dataset):
    def __init__(self, unlabeled_set, indices, pseudo_labels):
        self.data = Subset(unlabeled_set, indices)
        self.target = torch.LongTensor(pseudo_labels)[indices]

    def __getitem__(self, index):
        # if index < 0:
        #     index += len(self)
        # if index >= len(self):
        #     raise IndexError("index %d is out of bounds for axis 0 with size %d"%(index, len(self)))
            
        x = self.data[index][0]
        y = self.target[index].item()
        return x, y

    def __len__(self):
        
        return len(self.data)