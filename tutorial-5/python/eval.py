import os
from opt import parse
from env import create_env
from model import creat_model
from utils import tester

def run(args):
    env = create_env(args)

    model = creat_model(args, env)
    model_path = os.path.join(args.save_model_dir, 'mario_world_{}_{}.pth'.format(args.world, args.stage))

    if os.path.isfile(model_path):
        model = model.load(model_path)
        tester(args, env, model)
    else:
        print('Model not found!')

if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_video, exist_ok=True)

    run(args)