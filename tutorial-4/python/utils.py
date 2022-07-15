import os
import cv2
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import Generator, Discriminator

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def trainer(args, dataset, writer, device):
    z_dim = args.z_dim
    z_sample = Variable(torch.randn(100, z_dim)).to(device)

    G = Generator(in_dim=z_dim).to(device)
    D = Discriminator(3).to(device)

    G.train()
    D.train()

    criterion = nn.BCELoss()
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    dataloader = DataLoader(dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=2)

    steps = 0
    start_time = time.time()
    for e, epoch in enumerate(range(args.epoch)):
        progress_bar = tqdm(dataloader)
        for _, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.to(device)

            bs = imgs.size(0)

            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())
            
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            if steps % args.n_critic == 0:
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)

                f_logit = D(f_imgs)
                loss_G = criterion(f_logit, r_label)

                G.zero_grad()
                loss_G.backward()

                opt_G.step()

            steps += 1

            progress_bar.set_postfix({
                'Loss_D': round(loss_D.item(), 4),
                'Loss_G': round(loss_G.item(), 4),
                'Epoch': e + 1,
                'Step': steps,
            })

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(args.outdir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
 
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        writer.add_image('Generated images', grid_img, e)

        G.train()

        torch.save(G.state_dict(), args.save_g_model_path)
        torch.save(D.state_dict(), args.save_d_model_path)
    
    end_time = time.time()
    writer.close()
    print(f'Time usgae : {format_time(end_time - start_time)}')


def generate_video(image_path, output_video, video_name):
    images = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_video, video_name), fourcc, 5, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_path, image)))

    video.release()

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