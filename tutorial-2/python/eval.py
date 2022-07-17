import os
from opt import parse
from utils import generate_video, generate_predict_video
from models import select_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse()

    print('Generating test video...')
    generate_video(args.test_dir, args.outvideo, args.test_video_name)
    print('Done')

    model = select_model('resnet18', pretrained=False, num_classes=11)
    model.load_state_dict(torch.load(args.teacher_model_path))
    model = model.to(device)

    test_video_path = os.path.join(args.outvideo, args.test_video_name)
    pred_video_path = os.path.join(args.outvideo, args.pred_video_name)

    print('Generating predict video...')
    generate_predict_video(test_video_path, pred_video_path, model, device)
    print('Done')
