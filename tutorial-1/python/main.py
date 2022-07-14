import os
import pandas as pd
from dataset import COVID19Dataset
from opt import parse

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from model import Neural_Net
from utils import same_seed, calc_feature_correlation, train_valid_split, select_feature, trainer, predict, save_pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


def run(args):
    same_seed(args.seed)

    train_data, test_data = pd.read_csv(args.train_path), pd.read_csv(args.test_path).values

    if args.select_all:
        feat = list(range(train_data.values.shape[1] - 1))
    else:
        feat = calc_feature_correlation(train_data, args.correlation_threshold)

    print(f'Selected Features : {feat}')

    train_data = train_data.values
    train_data, valid_data = train_valid_split(train_data, args.split_ratio, args.seed)
    x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, feat)

    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)
    test_dataset = COVID19Dataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batchsize, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, pin_memory=True)

    model = Neural_Net(input_dim=x_train.shape[1]).to(device)
    print(summary(model, x_train.shape))

    trainer(args, train_loader, valid_loader, test_loader, model, device)

    model = Neural_Net(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(args.save_model_path))
    preds = predict(test_loader, model, device) 
    save_pred(preds, args.save_csv_path)


if __name__ == "__main__":
    args = parse()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    run(args)