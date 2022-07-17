import os
from opt import parse
import torch
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)