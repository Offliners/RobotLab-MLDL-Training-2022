import random
import numpy as np
from model import ActorCritic
from environment import create_train_env
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import timeit

def same_seed(seed): 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def trainer(index, opt_args, global_model, optimizer, device, save=False):
    torch.manual_seed(opt_args.seed + index)
    if save:
        start_time = timeit.default_timer()

    writer = SummaryWriter(opt_args.tensorboard)
    env, num_states, num_actions = create_train_env(opt_args.world, opt_args.stage, opt_args.action_type)
    local_model = ActorCritic(num_states, num_actions).to(device)

    local_model.train()
    state = torch.from_numpy(env.reset())
    state = state.to(device)

    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            if curr_episode % opt_args.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_mario_{}_{}".format(opt_args.save_model_dir, opt_args.world, opt_args.stage))

            print("Process {}. Episode {}".format(index, curr_episode))
        
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        log_policies = []
        values = []
        rewards = []
        entropies = []
        for _ in range(opt_args.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            state = state.to(device)
                
            if curr_step > opt_args.num_global_steps:
                done = True

            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
                state = state.to(device)

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        R = R.to(device)
        
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        gae = gae.to(device)
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt_args.gamma * opt_args.tau
            gae = gae + reward + opt_args.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt_args.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt_args.beta * entropy_loss
        writer.add_scalar("Train/Process{}_loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt_args.num_global_steps / opt_args.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('Time usage %.2f s ' % (end_time - start_time))
            
            return


def tester(index, opt_args, global_model):
    torch.manual_seed(opt_args.seed + index)
    env, num_states, num_actions = create_train_env(opt_args.world, opt_args.stage, opt_args.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())

    done = True
    curr_step = 0
    actions = deque(maxlen=opt_args.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        
        if opt_args.render:
            env.render()

        actions.append(action)
        
        if curr_step > opt_args.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()

        state = torch.from_numpy(state)