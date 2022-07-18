import argparse

def parse():
    parser = argparse.ArgumentParser(description="Robotlab MLDL Training Tutorial 3 - Synthetic Image Segmentation")
    parser.add_argument('--seed', type=int, default=3366, help='Set random seed')
    parser.add_argument('--epoch', type=int, default=10, help='Set training epochs')
    parser.add_argument('--num_worker', type=int, default=8, help='Set number of worker')
    parser.add_argument('--train_num', type=int, default=4000, help='Set number of training data')
    parser.add_argument('--val_num', type=int, default=400, help='Set number of validation data')
    parser.add_argument('--test_num', type=int, default=30, help='Set number of test data')
    parser.add_argument('--train_batchsize', type=int, default=25, help='Set training batchsize')
    parser.add_argument('--val_batchsize', type=int, default=25, help='Set validation batchsize')
    parser.add_argument('--test_batchsize', type=int, default=3, help='Set test batchsize')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Set optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Set learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Set weight decay')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints/model/model.pth', help='Path of saved model')
    parser.add_argument('--video_name', type=str, default='output.mp4', help='Name of output video')
    parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboard', help='Path of tensorboard')
    parser.add_argument('--output', type=str, default='./output', help='Path of output result')
    parser.add_argument('--video_dir', type=str, default='./video', help='Path of test images vidoe directory')

    return parser.parse_args()