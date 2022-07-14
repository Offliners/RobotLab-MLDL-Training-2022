import argparse

def parse():
    parser = argparse.ArgumentParser(description="Robotlab MLDL Training Tutorial 1 - Covid19 Cases Prediction")
    parser.add_argument('--seed', type=int, default=5201314, help='Set random seed')
    parser.add_argument('--epoch', type=int, default=5000, help='Set training epochs')
    parser.add_argument('--select_all', type=bool, default=False, help='Whether to select all features')
    parser.add_argument('--train_batchsize', type=int, default=512, help='Set training batchsize')
    parser.add_argument('--val_batchsize', type=int, default=512, help='Set validation batchsize')
    parser.add_argument('--test_batchsize', type=int, default=200, help='Set test batchsize')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Set split ratio')
    parser.add_argument('--correlation_threshold', type=float, default=0.8, help='Set correlation coefficient threshold')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Set optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Set learning rate')
    parser.add_argument('--lr_patience', type=int, default=50, help='Set learning rate patience')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Set weight decay')
    parser.add_argument('--train_path', type=str, default='./data/train.csv', help='Path of training data')
    parser.add_argument('--test_path', type=str, default='./data/test.csv', help='Path of test data')
    parser.add_argument('--early_stop', type=int, default=800, help='Set epoch of early stopping')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints/model/model.pth', help='Path of best model')
    parser.add_argument('--save_csv_path', type=str, default='./output/pred.csv', help='Path of prediction csv')
    parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboard', help='Path of tensorboard')

    return parser.parse_args()