import numpy as np
import time
import os
import cv2
import torch
from tqdm import tqdm
from dataset import PseudoDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
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
    print(f'Pseudo Labeling: {len(idx)}')
    dataset = PseudoDataset(Subset(dataset, idx), labels)

    return dataset


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 

    soft_loss = F.kl_div(F.log_softmax(outputs / T , dim=1) , F.softmax(teacher_outputs / T, dim=1), reduction='batchmean') * alpha * (T ** 2)
    return hard_loss + soft_loss


def trainer(args, train_loader, valid_loader, teacher_model, student_model, device, student_index):
    optimizer = getattr(torch.optim, args.optimizer)(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.period, verbose=True)

    writer = SummaryWriter(args.tensorboard)

    n_epochs = args.epoch
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(n_epochs):
        print(f'Student {student_index} Training Epoch [{epoch + 1}/{n_epochs}]')

        student_model.train()

        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            imgs, labels = batch

            logits = student_model(imgs.to(device))

            with torch.no_grad():
                soft_labels = teacher_model(imgs.to(device))

            loss = loss_fn_kd(logits, labels.to(device), soft_labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        student_model.eval()
        valid_loss = []
        valid_accs = []
        for batch in tqdm(valid_loader):
            imgs, labels = batch

            with torch.no_grad():
                logits = student_model(imgs.to(device))
                soft_labels = teacher_model(imgs.to(device))

            loss = loss_fn_kd(logits, labels.to(device), soft_labels)
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        scheduler.step()

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(student_model.state_dict(), os.path.join(args.save_student_model_dir, f'model_{student_index}.pth'))
            print('Saving model with valid acc {:.5f}'.format(best_acc))

        print()

        writer.add_scalars('Accuracy', {f'{student_index}_train_acc': train_acc, f'{student_index}_val_acc': valid_acc}, epoch)
        writer.add_scalars('Loss', {f'{student_index}_train_loss': train_loss, f'{student_index}_val_loss': valid_loss}, epoch)
    
    end_time = time.time()
    writer.close()
    print(f'Time usgae : {format_time(end_time - start_time)}')


def tta_predict(test_loader, model, device, alpha, output_path):
    model.eval()
    predictions = []

    for batch in tqdm(test_loader):
        img_list, _ = batch
        test_pred = []

        with torch.no_grad():
            for imgs in img_list:
                test_imgs = imgs[0].unsqueeze(0)
                origin_logit = model(test_imgs.to(device)).squeeze(0)
                
                tta_logit = model(imgs[1:].to(device))
                tta_logit = torch.mean(tta_logit, 0)

                logit = (alpha * origin_logit) + ((1 - alpha) * tta_logit)
                test_pred.append(logit)
            
        test_preds = torch.stack(test_pred)
        test_label = np.argmax(test_preds.cpu().data.numpy(), axis=1)
        predictions += test_label.tolist()

    with open(output_path, "w") as f:
        f.write("Id,Category\n")

        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")

    print('Done!')


def predict(test_loader, model, device, output_path):
    model.eval()
    predictions = []

    for batch in tqdm(test_loader):
        imgs, _ = batch

        with torch.no_grad():
            logits = model(imgs.to(device))
            
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    with open(output_path, "w") as f:
        f.write("Id,Category\n")

        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
    
    print('Done!')


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def generate_video(image_path, output_video, video_name):
    images = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_video, video_name), fourcc, 3, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_path, image)))

    video.release()


def generate_predict_video(test_video_path, pred_video_path, model, device):
    cap = cv2.VideoCapture(test_video_path) 
    vw = None
    frame = -1

    food11_label = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles or Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable or Fruit']

    model.eval()
    while True:
        frame += 1
        ret, img = cap.read()
        if not ret: break

        if img.shape[0] != 720:
            img = cv2.resize(img, (720, 720))

        img_proc = img.copy()
        img_proc = cv2.resize(img_proc, (224, 224))
        
        test_img = transforms.ToTensor()(img_proc).to(device)
        test_img = torch.unsqueeze(test_img, 0)

        preds = model(test_img)
        
        guess = preds.argmax(dim=-1).cpu().numpy()[0]
        perc = preds.detach().cpu().numpy()[0]
        perc = np.exp(perc) / sum(np.exp(perc))
        perc = [round(e * 100) for e in perc]

        pad_color = 0
        img = np.pad(img, ((0,0), (0,1280-720), (0,0)), mode='constant', constant_values=(pad_color))  
        
        line_type = cv2.LINE_AA
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1        
        thickness = 2
        x, y = 740, 60
        color = (255, 255, 255)
        
        text = 'Neural Network Output:'
        cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
        
        text = 'Input:'
        cv2.putText(img, text=text, org=(30, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)   

        y = 100
        for i, p in enumerate(perc):
            if i == guess:
                color = (255, 218, 158)
            else:
                color = (100, 100, 100)

            text = '{:>18} {:>3}%'.format(food11_label[i], p)
            cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
            y += 60

        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_width_height = img.shape[1], img.shape[0]
            vw = cv2.VideoWriter(pred_video_path, fourcc, 3, vid_width_height)
        
        vw.write(img)
        vw.write(img)
            
    cap.release()
    if vw is not None:
        vw.release()