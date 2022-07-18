import os
from opt import parse
from utils import same_seed, trainer
from dataset import SimDataset
from model import ResNetUNet

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def run(args):
    same_seed(args.seed)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    train_set = SimDataset(args.train_num, transform=tfm)
    val_set = SimDataset(args.val_num, transform=tfm)
    test_dataset = SimDataset(args.test_num, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_worker)
    valid_loader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=True, num_workers=args.num_worker)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=0)

    model = ResNetUNet(6).to(device)
    print(summary(model, (3, 224, 224)))

    for layer in model.base_layers:
        for param in layer.parameters():
            param.requires_grad = False

    trainer(args, train_loader, valid_loader, model, device)


if __name__ == "__main__":
    args = parse()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    run(args)