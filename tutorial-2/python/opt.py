import argparse

def parse():
    parser = argparse.ArgumentParser(description="Robotlab MLDL Training Tutorial 2 - Food Classification")
    parser.add_argument('--seed', type=int, default=3366, help='Set random seed')
    parser.add_argument('--epoch', type=int, default=100, help='Set training epochs')
    parser.add_argument('--n_critic', type=int, default=1, help='Set number of critics')
    parser.add_argument('--z_dim', type=int, default=100, help='Set the dimension of z space')
    parser.add_argument('--train_batchsize', type=int, default=64, help='Set training batchsize')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Set optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Set learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Set weight decay')
    parser.add_argument('--train_path', type=str, default='./data', help='Path of training data')
    parser.add_argument('--outdir', type=str, default='./checkpoints/images', help='Path of generated images')
    parser.add_argument('--outvideo', type=str, default='./checkpoints/video', help='Path of vidoe of generated images')
    parser.add_argument('--video_name', type=str, default='output.mp4', help='Name of output video')
    parser.add_argument('--save_g_model_path', type=str, default='./checkpoints/model/G.pth', help='Path of generator model')
    parser.add_argument('--save_d_model_path', type=str, default='./checkpoints/model/D.pth', help='Path of discriminator model')
    parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboard', help='Path of tensorboard')

    return parser.parse_args()