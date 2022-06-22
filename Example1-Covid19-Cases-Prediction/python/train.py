import os
from cfg import cfg
from utils import same_seed
from model import Neural_Net

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

same_seed(cfg['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    os.makedirs(cfg['save_path'], exist_ok=True)

    criterion = nn.MSELoss(reduction='mean')

    model = Neural_Net().to(device)
    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])

    writer = SummaryWriter(cfg['tensorboard'])



if __name__ == "__main__":
    main()