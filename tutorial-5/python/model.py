from stable_baselines3 import PPO

def creat_model(args, env):
    model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=args.tensorboard, learning_rate=args.lr, n_steps=args.step,
            batch_size=args.batchsize, n_epochs=args.epoch, gamma=args.gamma)

    return model
