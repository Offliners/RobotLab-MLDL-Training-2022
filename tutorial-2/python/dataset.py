import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id]


class TTADataset(Dataset):
    def __init__(self, path, train_tfm, test_tfm, repeat_num=10, files=None):
        super(TTADataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        
        self.repeat_num = repeat_num
        self.train_transform = train_tfm
        self.test_transform = test_tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)

        ims = [self.test_transform(im)]
        ims += [self.train_transform(im) for _ in range(self.repeat_num)]
        label = -1

        return torch.stack(ims), label
