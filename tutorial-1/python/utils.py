import csv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calc_feature_correlation(train_data, threshold):
    cor_matrix = train_data.corr().abs()
    cor_matrix = cor_matrix['tested_positive.4'].drop('tested_positive.4')
    useful_feature = [train_data.columns.get_loc(index) for index, value in cor_matrix.items() if value > threshold]

    if not useful_feature:
        print('No features')

    return useful_feature

def select_feature(train_data, valid_data, test_data, features):
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data
        
    return raw_x_train[:,features], raw_x_valid[:,features], raw_x_test[:,features], y_train, y_valid


def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    
    return preds


def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


def trainer(args, train_loader, valid_loader, model, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), args.lr, (0.9, 0.98), 1e-8, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True)

    writer = SummaryWriter(args.tensorboard)

    epochs = args.epoch
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
        writer.add_scalars('Loss', {'train_loss': mean_train_loss, 'val_loss': mean_valid_loss}, step)

        scheduler.step(mean_valid_loss)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), args.save_model_path)
            print('Saving model with val loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= args.early_stop:
            print('\nModel is not improving, so we halt the training session.')
            print(f'Best Loss: {best_loss}')
            break