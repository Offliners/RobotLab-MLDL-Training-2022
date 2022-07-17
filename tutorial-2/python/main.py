import os
from PIL import Image
from opt import parse
from utils import same_seed, trainer, get_pseudo_labels, predict, tta_predict
from models import select_model, modelEnsemble
from dataset import TTADataset
from randaugment import ImageNetPolicy

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def run(args):
    same_seed(args.seed)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    train_set = DatasetFolder(args.train_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder(args.valid_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder(args.unlabeled_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    if args.use_tta:
        test_set = TTADataset(args.test_dir, train_tfm, test_tfm)
    else:
        test_set = DatasetFolder(args.test_dir, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    # Teacher
    teacher_model = select_model('resnet18', False).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_model_path))
    print(summary(teacher_model, (3, 224, 224)))

    test_loader = DataLoader(test_set, batch_size=args.test_batchsize, shuffle=False, num_workers=args.num_worker, pin_memory=True)
    teacher_csv_path = os.path.join(args.save_csv_dir, 'teacher_pred.csv')
    if args.use_tta:
        tta_predict(test_loader, teacher_model, device, args.alpha, teacher_csv_path)
    else:
        predict(test_loader, teacher_model, device, teacher_csv_path)

    # Semi-supervised learning
    print('\nDo semi-supervised learning ...')
    pseudo_set = get_pseudo_labels(unlabeled_set, teacher_model, args.train_batchsize, device, args.pseudo_label_threshold)
    concat_dataset = ConcatDataset([train_set, pseudo_set])
    train_loader = DataLoader(concat_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.val_batchsize, shuffle=True, num_workers=args.num_worker, pin_memory=True)

    # Student A, B, C
    student_index = ['A', 'B', 'C']
    student_model_names = [args.student_A_name, args.student_B_name, args.student_C_name]
    save_paths = []
    for i in range(len(student_index)):
        student_model = select_model(student_model_names[i], False).to(device)
        print(summary(student_model, (3, 224, 224)))

        trainer(args, train_loader, valid_loader, teacher_model, student_model, device, student_index[i])
        save_model_path = os.path.join(args.save_student_model_dir, f'model_{student_index[i]}.pth')
        save_csv_path = os.path.join(args.save_csv_dir, f'student_{student_index[i]}_pred.csv')
        if args.use_tta:
            tta_predict(test_loader, student_model, device, args.alpha, save_csv_path)
        else:
            predict(test_loader, student_model, device, save_csv_path)
        
        save_paths.append(save_model_path)
        del student_model

    ensemble_model = modelEnsemble(student_model_names, save_paths).to(device)
    student_csv_path = os.path.join(args.save_csv_dir, 'student_ensemble_pred.csv')
    if args.use_tta:
        tta_predict(test_loader, ensemble_model, device, args.alpha, student_csv_path)
    else:
        predict(test_loader, ensemble_model, device, student_csv_path)


if __name__ == "__main__":
    args = parse()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./checkpoints/model', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    if not os.path.isdir('./data'):
        print('Dataset not found!')
        exit(1)

    run(args)