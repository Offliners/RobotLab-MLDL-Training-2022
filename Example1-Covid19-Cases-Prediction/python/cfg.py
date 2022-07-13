cfg = {
    'seed' : 5211314,
    'epoch' : 5000,
    'train_batchsize' : 8,
    'valid_batchsize' : 8,
    'test_batchsize' : 8,
    'split_ratio' : 0.3,
    'select_all' : False,
    'select_features' : [38, 39, 40, 41, 53, 54, 55, 56, 57, 69, 70, 71, 72, 73, 85, 86, 87, 88, 89, 101, 102, 103, 104, 105],

    'optimizer': 'AdamW',
    'optim_hparas': {
        'lr' : 1e-5,
        'betas' : (0.9, 0.98),
        'weight_decay' : 1e-6
    },

    'train_path' : './data/train.csv',
    'test_path' : './data/test.csv',

    'early_stop' : 400,
    'save_model_path' : './checkpoints/model.pth',
    'save_csv_path' : './result',
    'tensorboard' : './tensorboard'
}