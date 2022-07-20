import os
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from stable_baselines3.common.callbacks import BaseCallback
import torch
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed): 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TrainCallback(BaseCallback):
    def __init__(self, args, env, model, verbose=1):
        super(TrainCallback, self).__init__(verbose)
        self.args = args
        self.env = env
        self.model = model
        self.writer = SummaryWriter(args.tensorboard)

    def _on_step(self):
        n_episodes = self.args.episode
        if self.n_calls % self.args.check_freq == 0:
            model_path = os.path.join(self.args.save_model_dir, 'mario_world_{}_{}.pth'.format(self.args.world, self.args.stage))
            self.model.save(model_path)

            total_reward = [0] * n_episodes
            total_time = [0] * n_episodes
            best_reward = 0
            for i in range(n_episodes):
                state = self.env.reset()
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < self.args.max_timestep_test:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = self.env.step(action)
                    total_time[i] += 1

                total_reward[i] = info[0]['max_x_pos']
                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]

                state = self.env.reset()

            reward_avg = sum(total_reward) / n_episodes
            print('time steps:', self.n_calls, '/', self.args.total_timestep, '\t',
                  'average reward:', reward_avg, '\t',
                  'best reward:', best_reward)

            self.writer.add_scalars('Reward', {'average reward' : reward_avg, 'best reward' : best_reward}, self.n_calls)

        return True


def tester(args, env, model):
    n_episodes = args.episode
    total_reward = [0] * n_episodes
    total_time = [0] * n_episodes
    best_reward = 0
    frames_best = []
    for i in range(n_episodes):
        state = env.reset()
        done = False
        total_reward[i] = 0
        total_time[i] = 0
        frames = []
        while not done and total_time[i] < args.max_timestep_test:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            total_reward[i] += reward[0]
            total_time[i] += 1
            frames.append(copy.deepcopy(env.render(mode='rgb_array')))

        if total_reward[i] > best_reward:
            best_reward = total_reward[i]
            frames_best = copy.deepcopy(frames)

        print('test episode:', i, 'reward:', total_reward[i], 'time:', total_time[i])

    print('average reward:', (sum(total_reward) / n_episodes),
        'average time:', (sum(total_time) / n_episodes),
        'best_reward:', best_reward)

    frames_new = np.array(frames_best)
    matplotlib.rcParams['animation.embed_limit'] = 2**128
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames_new[0])
    plt.axis('off')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    animate = lambda i: patch.set_data(frames_new[i])
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames_new), interval = 50)
    plt.close()

    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'mpeg4'])
    ani.save(os.path.join(args.output_video, f'video_world_{args.world}_{args.stage}.mp4'), writer=FFwriter)