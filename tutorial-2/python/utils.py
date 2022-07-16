import numpy as np
import torch
from tqdm import tqdm
from dataset import PseudoDataset
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_pseudo_labels(dataset, model, batch_size, device, threshold=0.9):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    i = 0
    idx = []
    labels = []
    for batch in tqdm(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)
        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * batch_size + j)
                labels.append(int(torch.argmax(x)))
        
        i += 1

    model.train()
    print ("Pseudo Labeling: {:5d}\n".format(len(idx)))
    dataset = PseudoDataset(Subset(dataset, idx), labels)

    return dataset


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def trainer(args, train_set, unlabeled_set, train_loader, valid_loader, model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    writer = SummaryWriter(args.tensorboard)

    n_epochs = args.epoch
    best_acc = 0.0
    for epoch in range(n_epochs):
        print(f'Epoch [{epoch + 1}/{n_epochs}]')
        if args.do_semi and best_acc > args.start_pseudo_threshold:
            pseudo_set = get_pseudo_labels(unlabeled_set, model, args.train_batchsize, device)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True, drop_last=True)

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
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), args.save_model_path)
            print('Saving model with valid acc {:.5f}'.format(best_acc))

        print()

        writer.add_scalars('Accuracy', {'train_acc': train_acc, 'val_acc': valid_acc}, epoch)
        writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': valid_loss}, epoch)
    
    writer.close()


def predict(args, test_loader, model, device):
    model.eval()
    predictions = []

    for batch in tqdm(test_loader):
        imgs, _ = batch

        with torch.no_grad():
            logits = model(imgs.to(device))
            
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    with open(args.save_csv_path, "w") as f:
        f.write("Id,Category\n")

        for i, pred in  enumerate(predictions):
            f.write(f"{i},{pred}\n")