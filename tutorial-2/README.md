# Tutorial 2 - Food Classification

## Usage
在本機端開始訓練
```shell 
$ cd python

# 下載訓練資料
$ python download_dataset.py

# 開始訓練
$ python main.py
```

使用Tensorboard來觀看訓練與驗證的Accuracy與Loss
```shell
$ tensorboard --logdir=./checkpoints/tensorboard/
```
![tensorboard]()

產生的資料夾內容
```shell
python/
    checkpoints/
        model/
            model.pth
        tensorboard/
            Accuracy_train_acc/
            Accuracy_val_acc/
            Loss_train_loss/
            Loss_val_loss/
    data/
        testing/
        training/
        validation/
```

產生的`pred.csv`可上傳至[Kaggle](https://www.kaggle.com/competitions/ml2021spring-hw3/)看表現如何

我的分數如下:

Public score : `0.0` (Rank : `/1404`)

Private score : `0.0` (Rank : `/1404`)


## Help
```shell
$ python main.py --help
usage: main.py [-h] [--seed SEED] [--epoch EPOCH] [--model_name MODEL_NAME] [--pretrained PRETRAINED] [--do_semi DO_SEMI] [--pseudo_label_threshold PSEUDO_LABEL_THRESHOLD]
               [--start_pseudo_threshold START_PSEUDO_THRESHOLD] [--num_worker NUM_WORKER] [--train_batchsize TRAIN_BATCHSIZE] [--val_batchsize VAL_BATCHSIZE] [--test_batchsize TEST_BATCHSIZE]
               [--optimizer OPTIMIZER] [--lr LR] [--weight_decay WEIGHT_DECAY] [--train_dir TRAIN_DIR] [--unlabeled_dir UNLABELED_DIR] [--valid_dir VALID_DIR] [--test_dir TEST_DIR]
               [--save_model_path SAVE_MODEL_PATH] [--save_csv_path SAVE_CSV_PATH] [--tensorboard TENSORBOARD]

Robotlab MLDL Training Tutorial 2 - Food Classification

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Set random seed
  --epoch EPOCH         Set training epochs
  --model_name MODEL_NAME
                        Set name of model
  --pretrained PRETRAINED
                        Whether to use pretrained model
  --do_semi DO_SEMI     Whether to do semi-supervised learning
  --pseudo_label_threshold PSEUDO_LABEL_THRESHOLD
                        Set threshold of pseudo labels
  --start_pseudo_threshold START_PSEUDO_THRESHOLD
                        Set accuracy threshold of using pseudo labels
  --num_worker NUM_WORKER
                        Set number of worker
  --train_batchsize TRAIN_BATCHSIZE
                        Set training batchsize
  --val_batchsize VAL_BATCHSIZE
                        Set validation batchsize
  --test_batchsize TEST_BATCHSIZE
                        Set test batchsize
  --optimizer OPTIMIZER
                        Set optimizer
  --lr LR               Set learning rate
  --weight_decay WEIGHT_DECAY
                        Set weight decay
  --train_dir TRAIN_DIR
                        Path of labeled training data directory
  --unlabeled_dir UNLABELED_DIR
                        Path of unlabeled training data directory
  --valid_dir VALID_DIR
                        Path of validation data directory
  --test_dir TEST_DIR   Path of test data directory
  --save_model_path SAVE_MODEL_PATH
                        Path of best model
  --save_csv_path SAVE_CSV_PATH
                        Path of prediction csv
  --tensorboard TENSORBOARD
                        Path of tensorboard
```