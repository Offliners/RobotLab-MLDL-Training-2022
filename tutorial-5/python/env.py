import cv2
import numpy as np
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if done:
                break
        return obs, reward, done, info


class Downsample(gym.ObservationWrapper):
    def __init__(self, env, ratio):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (oldh//ratio, oldw//ratio, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame


class DeadlockEnv(gym.Wrapper):
    def __init__(self, env, threshold=20):
        super().__init__(env)
        self.last_x_pos = 0
        self.max_x_pos = 0
        self.count = 0
        self.threshold = threshold
        
    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.max_x_pos = 0
        self.count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos, y_pos = info['x_pos'], info['y_pos']
        
        if self.max_x_pos > 3600 and x_pos < 1080:
            x_pos += 4000
        elif x_pos >= 4152:
            x_pos += 1000
        
        if x_pos > self.max_x_pos:
            info['max_x_pos'] = self.max_x_pos = x_pos
        else:
            info['max_x_pos'] = self.max_x_pos
        
        if x_pos <= self.last_x_pos:
            self.count += 1
        else:
            self.count = 0
            
        if self.count >= self.threshold or (
            x_pos < self.last_x_pos - 500 and (self.max_x_pos < 1200 or self.max_x_pos >= 1848)) or (
            self.max_x_pos >= 1200 and 300 <= x_pos < 1200 or (
            2290 < x_pos < 3128 and y_pos < 127) or (
            2450 < x_pos < 3128) or (
            3670 < x_pos < 3849)):
            reward = -15
            done = True
        
        self.last_x_pos = x_pos        
        return state, reward, done, info


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        if done: 
            reward = 1 if info['flag_get'] else -1
        else:
            reward /= 30
        
        return state, reward, done, info


def create_env(args):
    env = gym_super_mario_bros.make(f'SuperMarioBros-{args.world}-{args.stage}-v{args.version}')
    env = JoypadSpace(env, [["right", "B"], ["right", "A", "B"], ["down"]])
    env = SkipFrame(env, skip=args.num_skip_frame)
    env = DeadlockEnv(env, threshold=40)
    env = CustomRewardEnv(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = Downsample(env, args.downsample_rate)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, args.num_stack_frame, channels_order='last')

    return env