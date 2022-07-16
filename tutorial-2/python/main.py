import os
from PIL import Image
from opt import parse
from utils import same_seed, trainer, predict
from models import modelEnsemble
import ttach as tta
from randaugment import ImageNetPolicy

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

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    train_set = DatasetFolder(args.train_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder(args.valid_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder(args.unlabeled_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder(args.test_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.val_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batchsize, shuffle=False)

    ensemble_model = modelEnsemble('vgg16', 'resnet18', 'densenet121').to(device)
    for param in ensemble_model.parameters():
        param.requires_grad = False

    for param in ensemble_model.classifier.parameters():
        param.requires_grad = True 

    print(summary(ensemble_model, (3, 224, 224)))

    trainer(args, train_set, unlabeled_set, train_loader, valid_loader, ensemble_model, device)

    ensemble_model.load_state_dict(torch.load(args.save_model_path))
    tta_model = tta.ClassificationTTAWrapper(ensemble_model, tta.aliases.five_crop_transform(224, 224))
    predict(args, test_loader, tta_model, device)


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