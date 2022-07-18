import argparse

def parse():
    parser = argparse.ArgumentParser(description="Robotlab MLDL Training Tutorial 5 - Super Mario")
    parser.add_argument('--seed', type=int, default=3366, help='Set random seed')
    parser.add_argument('--world', type=int, default=1, help='Set a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world')
    parser.add_argument('--stage', type=int, default=1, help='Set a number in {1, 2, 3, 4} indicating the stage within a world')
    parser.add_argument('--render', type=bool, default=True, help='Whether to render the environment')
    parser.add_argument('--version', type=int, default=0, help='Set a number in {0, 1, 2, 3} specifying the ROM mode to use')
    parser.add_argument('--action_type', type=str, default='complex', help='Set game difficulty in {right_only, simple, complex}')
    parser.add_argument('--lr', type=float, default=1e-4, help='Set learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Set discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='Set parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='Set entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6, help='Set the number of processes')
    parser.add_argument("--save_interval", type=int, default=500, help='Set number of steps between savings')
    parser.add_argument("--max_actions", type=int, default=200, help='Set maximum repetition steps in test phase')
    parser.add_argument('--save_model_dir', type=str, default='./checkpoints/model', help='Path of saved model directory')
    parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboard', help='Path of tensorboard')
    parser.add_argument('--output_video', type=str, default='./video', help='Path of output vidoe directory')
    parser.add_argument('--use_pretrained', type=bool, default=True, help='Whether to use pretrained model weight')

    return parser.parse_args()