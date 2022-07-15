import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id]


def get_pseudo_labels(dataset, model, batch_size, device, threshold=0.9):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    idx = []
    labels = []
    for i, batch in tqdm(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)

        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * batch_size + j)
                labels.append(int(torch.argmax(x)))

    model.train()
    print ("\nNew pseudo label data: {:5d}\n".format(len(idx)))
    dataset = PseudoDataset(Subset(dataset, idx), labels)

    return dataset


def trainer(args, train_set, unlabeled_set, train_loader, valid_loader, model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, args['optimizer'])(model.parameters(), args['lr'], (0.9, 0.98), 1e-8, args['weight_decay'])

    n_epochs = args.epoch
    best_acc = 0.0
    for epoch in range(n_epochs):
        if args.do_semi and best_acc > args.start_pseudo_threshold:
            pseudo_set = get_pseudo_labels(unlabeled_set, model, args.train_batchsize, device)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_work, pin_memory=True)

        model.train()

        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            imgs, labels = batch

            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), args.save_model_path)
            print('saving model with acc {:.5f}'.format(best_acc))

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")