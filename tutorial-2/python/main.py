import os
from PIL import Image
from opt import parse
from utils import same_seed, trainer

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def run(args):
    same_seed(args.seed)

    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.val_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batchsize, shuffle=False)
    


if __name__ == "__main__":
    args = parse()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    if not os.path.isdir('./data'):
        print('Dataset not found!')
        exit(1)

    run(args)