cfg = {
    'seed' : 0,
    'epoch' : 5000,
    'train_batchsize' : 16,
    'split_ratio' : 0.2,
    'select_all' : True,
    'select_feature' : [],

    'optimizer': 'Adam',
    'optim_hparas': {
        'lr' : 1e-5,
        'weight_decay' : 1e-6
    },

    'early_stop' : 400,
    'save_path' : './checkpoints',
    'tensorboard' : './'
}