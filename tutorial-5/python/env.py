import cv2
import numpy as np
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
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

    if args.action_type == "right":
        actions = RIGHT_ONLY
    elif args.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    elif args.action_type == 'complex':
        actions = COMPLEX_MOVEMENT
    else:
        print('Unknown action type!')

    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=args.num_skip_frame)
    env = CustomRewardEnv(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = Downsample(env, args.downsample_rate)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, args.num_stack_frame, channels_order='last')

    return env