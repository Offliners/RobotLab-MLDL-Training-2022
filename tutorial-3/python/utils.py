import itertools
from collections import defaultdict
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_side_by_side(img_arrays, save_path):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    ncol = len(img_arrays)
    img_array = np.array(flatten_list)
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))
    plt.text(x=0.5, y=0.94, s="Input image - Ground-truth - Predicted mask", fontsize=18, ha="center", transform=f.transFigure)
    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])
    
    f.savefig(save_path)


def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def trainer(args, train_loader, valid_loader, model, device):
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    dataloaders = {
        'train': train_loader,
        'val': valid_loader
    }
    
    best_loss = np.inf

    writer = SummaryWriter(args.tensorboard)

    start_time = time.time()
    n_epochs = args.epoch
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            outputs = []
            for k in metrics.keys():
                outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
                writer.add_scalars(f'{phase}_loss', {f'{k}' : metrics[k] / epoch_samples}, epoch)
            
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                scheduler.step()

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print('Saving model with valid loss {:.5f}'.format(best_loss))
                torch.save(model.state_dict(), args.save_model_path)

    end_time = time.time()
    print(f'Time usgae : {format_time(end_time - start_time)}')


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def generate_video(image_path, output_video, video_name):
    images = [img for img in os.listdir(image_path) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_video, video_name), fourcc, 2, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_path, image)))

    video.release()


def tester(args, test_loader, model, device):
    count = 0
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model(inputs)
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()

        input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

        target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
        pred_rgb = [masks_to_colorimg(x) for x in pred]

        plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb], os.path.join(args.output, f'result_{count}.png'))
        count += 1


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