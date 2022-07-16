import argparse

def parse():
    parser = argparse.ArgumentParser(description="Robotlab MLDL Training Tutorial 2 - Food Classification")
    parser.add_argument('--seed', type=int, default=3366, help='Set random seed')
    parser.add_argument('--epoch', type=int, default=50, help='Set training epochs')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use pretrained model')
    parser.add_argument('--do_semi', type=bool, default=True, help='Whether to do semi-supervised learning')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.95, help='Set threshold of pseudo labels')
    parser.add_argument('--start_pseudo_threshold', type=float, default=0.7, help='Set accuracy threshold of using pseudo labels')
    parser.add_argument('--num_worker', type=int, default=8, help='Set number of worker')
    parser.add_argument('--train_batchsize', type=int, default=64, help='Set training batchsize')
    parser.add_argument('--val_batchsize', type=int, default=64, help='Set validation batchsize')
    parser.add_argument('--test_batchsize', type=int, default=64, help='Set test batchsize')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Set optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Set learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Set weight decay')
    parser.add_argument('--gamma', type=float, default=0.97, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--use_tta', type=bool, default=True, help='Whether to do test time augmentation')
    parser.add_argument('--train_dir', type=str, default='./data/training/labeled', help='Path of labeled training data directory')
    parser.add_argument('--unlabeled_dir', type=str, default='./data/training/unlabeled', help='Path of unlabeled training data directory')
    parser.add_argument('--valid_dir', type=str, default='./data/validation', help='Path of validation data directory')
    parser.add_argument('--test_dir', type=str, default='./data/testing', help='Path of test data directory')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints/model/model.pth', help='Path of best model')
    parser.add_argument('--save_csv_path', type=str, default='./output/pred.csv', help='Path of prediction csv')
    parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboard', help='Path of tensorboard')

    return parser.parse_args()