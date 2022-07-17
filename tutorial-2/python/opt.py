import argparse

def parse():
    parser = argparse.ArgumentParser(description="Robotlab MLDL Training Tutorial 2 - Food Classification")
    parser.add_argument('--seed', type=int, default=3366, help='Set random seed')
    parser.add_argument('--epoch', type=int, default=120, help='Set training epochs')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.95, help='Set threshold of pseudo labels')
    parser.add_argument('--num_worker', type=int, default=8, help='Set number of worker')
    parser.add_argument('--train_batchsize', type=int, default=64, help='Set training batchsize')
    parser.add_argument('--val_batchsize', type=int, default=64, help='Set validation batchsize')
    parser.add_argument('--test_batchsize', type=int, default=64, help='Set test batchsize')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Set optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Set learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Set weight decay')
    parser.add_argument('--period', type=int, default=10, help='Set maximum number of iterations')
    parser.add_argument('--use_tta', type=bool, default=True, help='Whether to do test time augmentation')
    parser.add_argument('--alpha', type=float, default=0.6, help='Set weight between test_tfm and train_tfm')
    parser.add_argument('--train_dir', type=str, default='./data/training/labeled', help='Path of labeled training data directory')
    parser.add_argument('--unlabeled_dir', type=str, default='./data/training/unlabeled', help='Path of unlabeled training data directory')
    parser.add_argument('--valid_dir', type=str, default='./data/validation', help='Path of validation data directory')
    parser.add_argument('--test_dir', type=str, default='./data/testing/00', help='Path of test data directory')
    parser.add_argument('--teacher_model_path', type=str, default='./data/teacher_model.pth', help='Path of teacher model')
    parser.add_argument('--student_A_name', type=str, default='resnet50', help='Name of student A model')
    parser.add_argument('--student_B_name', type=str, default='resnet34', help='Name of student B model')
    parser.add_argument('--student_C_name', type=str, default='resnet101', help='Name of student C model')
    parser.add_argument('--save_student_model_dir', type=str, default='./checkpoints/model', help='Path of saved student model directory')
    parser.add_argument('--save_csv_dir', type=str, default='./output', help='Path of prediction csv directory')
    parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboard', help='Path of tensorboard')
    parser.add_argument('--video_dir', type=str, default='./video', help='Path of test images vidoe directory')
    parser.add_argument('--test_video_name', type=str, default='test_image.mp4', help='Name of output test video')
    parser.add_argument('--pred_video_name', type=str, default='test_image_pred.mp4', help='Name of output predict video')

    return parser.parse_args()