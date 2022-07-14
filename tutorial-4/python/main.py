import os
from opt import parse
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from utils import same_seed, trainer, generate_video
from dataset import get_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def run(args):
    same_seed(args.seed)

    dataset = get_dataset(args.train_path)

    writer = SummaryWriter(args.tensorboard)
    images = [(dataset[i] + 1) / 2 for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)

    writer.add_image('Show some images', grid_img)

    trainer(args, dataset, writer, device)
    generate_video(args.outdir, args.outvideo, args.video_name)


if __name__ == "__main__":
    args = parse()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.outvideo, exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    run(args)