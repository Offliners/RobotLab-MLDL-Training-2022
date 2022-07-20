import os
from opt import parse
from utils import same_seed, TrainCallback
import torch
from env import create_env
from model import creat_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def run(args):
    env = create_env(args)

    model = creat_model(args, env)
    callback = TrainCallback(args, env, model)
    model.learn(total_timesteps=args.total_timestep, callback=callback)

    print('Training Done!')


if __name__ == "__main__":
    args = parse()

    same_seed(args.seed)

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    run(args)