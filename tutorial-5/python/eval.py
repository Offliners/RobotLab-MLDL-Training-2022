import os
from opt import parse
import torch
from environment import create_train_env
from model import ActorCritic
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def eval(opt_args):
    torch.manual_seed(opt_args.seed)
    env, num_states, num_actions = create_train_env(opt_args.world, opt_args.stage, opt_args.action_type,
                                                    "{}/mario_world_{}_{}.mp4".format(opt_args.output_video, opt_args.world, opt_args.stage))
    model = ActorCritic(num_states, num_actions)

    if opt_args.use_pretrained:
        opt_args.save_model_dir = './pretrained_model'
        if not os.path.isdir(opt_args.save_model_dir):
            print('Pretrained model not found!')
            exit(1)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_mario_{}_{}".format(opt_args.save_model_dir, opt_args.world, opt_args.stage)))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load("{}/a3c_mario_{}_{}".format(opt_args.save_model_dir, opt_args.world, opt_args.stage),
                                         map_location=lambda storage, loc: storage))
    
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        state = state.to(device)

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        
        if opt_args.render:
            env.render()

        if info['flag_get']:
            print("World {} stage {} completed".format(opt_args.world, opt_args.stage))
            break


if __name__ == "__main__":
    opt_args = parse()
    os.makedirs(opt_args.output_video, exist_ok=True)

    eval(opt_args)