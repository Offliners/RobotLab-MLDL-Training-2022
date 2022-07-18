import os
from opt import parse
from utils import same_seed, GlobalAdam, trainer, tester
import torch
from environment import create_train_env
from model import ActorCritic
import torch.multiprocessing as _mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def run(opt_args):
    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(opt_args.world, opt_args.stage, opt_args.action_type)
    global_model = ActorCritic(num_states, num_actions).to(device)
    
    global_model.share_memory()

    optimizer = GlobalAdam(global_model.parameters(), lr=opt_args.lr)
    processes = []
    for index in range(opt_args.num_processes):
        if index == 0:
            process = mp.Process(target=trainer, args=(index, opt_args, global_model, optimizer, device, True))
        else:
            process = mp.Process(target=trainer, args=(index, opt_args, global_model, optimizer, device))
        
        process.start()
        processes.append(process)

    process = mp.Process(target=tester, args=(opt_args.num_processes, opt_args, global_model))
    process.start()
    processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    opt_args = parse()

    same_seed(opt_args.seed)

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs(opt_args.output_video, exist_ok=True)
    os.makedirs(opt_args.tensorboard, exist_ok=True)

    run(opt_args)