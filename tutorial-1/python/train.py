import os
from tqdm import tqdm
from cfg import cfg
import pandas as pd
from utils import same_seed, select_feature, train_valid_split, predict, save_pred
from dataset import COVID19Dataset
from model import Neural_Net

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

same_seed(cfg['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def main():
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs(cfg['save_csv_path'], exist_ok=True)
    os.makedirs(cfg['tensorboard'], exist_ok=True)

    train_data, test_data = pd.read_csv(cfg['train_path']).values, pd.read_csv(cfg['test_path']).values
    train_data, valid_data = train_valid_split(train_data, cfg['split_ratio'], cfg['seed'])
    x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, cfg['select_features'], cfg['select_all'])

    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)
    test_dataset = COVID19Dataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batchsize'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['valid_batchsize'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_batchsize'], shuffle=False, pin_memory=True)

    criterion = nn.MSELoss(reduction='mean')

    model = Neural_Net(input_dim=x_train.shape[1]).to(device)
    print(summary(model, x_train.shape))

    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])

    writer = SummaryWriter(cfg['tensorboard'])

    epochs = cfg['epoch']
    best_loss = 9999. 
    step = 0 
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device) 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            train_losses.append(loss.detach().item())
            
            pbar.set_description(f'Epoch [{epoch+1}/{epochs}]')
            pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(train_losses) / len(train_losses)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        valid_losses = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            valid_losses.append(loss.item())
            
        mean_valid_loss = sum(valid_losses) / len(valid_losses)
        print(f'Epoch [{epoch+1}/{epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), cfg['save_model_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= cfg['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            print(f'Best Loss: {best_loss}')
            break
    
    model = Neural_Net(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(cfg['save_model_path']))
    preds = predict(test_loader, model, device) 
    save_pred(preds, os.path.join(cfg['save_csv_path'], 'pred.csv'))


if __name__ == "__main__":
    main()