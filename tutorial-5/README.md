# Tutorial 5 - Super Mario
|World/Stage|1|2|3|4|
|-|-|-|-|-|
|1|![World 1-1](./img/mario_world_1_1.gif)|![World 1-2](./img/mario_world_1_2.gif)|![World 1-3](./img/mario_world_1_3.gif)|![World 1-4](./img/mario_world_1_4.gif)|
|2|![World 2-1](./img/mario_world_2_1.gif)|![World 2-2](./img/mario_world_2_2.gif)|![World 2-3](./img/mario_world_2_3.gif)|![World 2-4](./img/mario_world_2_4.gif)|
|3|![World 3-1](./img/mario_world_3_1.gif)|![World 3-2](./img/mario_world_3_2.gif)|![World 3-3](./img/mario_world_3_3.gif)|![World 3-4](./img/mario_world_3_4.gif)|
|4|![World 4-1](./img/mario_world_4_1.gif)|![World 4-2](./img/mario_world_4_2.gif)|![World 4-3](./img/mario_world_4_3.gif)|![World 4-4](./img/mario_world_4_4.gif)|
|5|![World 5-1](./img/mario_world_5_1.gif)|![World 5-2](./img/mario_world_5_2.gif)|![World 5-3](./img/mario_world_5_3.gif)|![World 5-4](./img/mario_world_5_4.gif)|
|6|![World 6-1](./img/mario_world_6_1.gif)|![World 6-2](./img/mario_world_6_2.gif)|![World 6-3](./img/mario_world_6_3.gif)|![World 6-4](./img/mario_world_6_4.gif)|
|7|![World 7-1](./img/mario_world_7_1.gif)|![World 7-2](./img/mario_world_7_2.gif)|![World 7-3](./img/mario_world_7_3.gif)|![World 7-4](./img/mario_world_7_4.gif)|
|8|![World 8-1](./img/mario_world_8_1.gif)|![World 8-2](./img/mario_world_8_2.gif)|![World 8-3](./img/mario_world_8_3.gif)|![World 8-4](./img/mario_world_8_4.gif)|

## Usage
在本機端開始訓練
```shell
$ cd python

# 開始訓練(預設是World 1-1)
$ python main.py

# 針對特定World或者Stage進行訓練
$ python main.py --world {index of world (1~8)} --stage {index of stage (1~4)}

# 訓練完成後可以看表現如何(預設是World 1-1)
$ python eval.py

# 針對特定World或者Stage看表現如何
$ python eval.py --world {index of world (1~8)} --stage {index of stage (1~4)}
```

使用Tensorboard來觀看模型的Reward，以及PPO模型的其他參數變化
```shell
$ tensorboard --logdir=./checkpoints/tensorboard/
```

![tensorboard](./img/tutorial-5-tensorboard.png)

產生的資料夾內容
```shell
python/
    checkpoints/
        model/
            mario_world_{index of world}_{index of stage}.pth
        tensorboard/
        video/
            video_world_{index of world}_{index of stage}.mp4
    data/
        0.jpg
        ...
```

## Help
```shell
$ python main.py --help
usage: main.py [-h] [--seed SEED] [--world WORLD] [--stage STAGE] [--render RENDER] [--version VERSION] [--action_type ACTION_TYPE] [--lr LR] [--gamma GAMMA] [--tau TAU] [--beta BETA]
               [--num_local_steps NUM_LOCAL_STEPS] [--num_global_steps NUM_GLOBAL_STEPS] [--num_processes NUM_PROCESSES] [--save_interval SAVE_INTERVAL] [--max_actions MAX_ACTIONS]
               [--save_model_dir SAVE_MODEL_DIR] [--tensorboard TENSORBOARD] [--output_video OUTPUT_VIDEO] [--use_pretrained USE_PRETRAINED]

Robotlab MLDL Training Tutorial 5 - Super Mario

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Set random seed
  --world WORLD         Set a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world
  --stage STAGE         Set a number in {1, 2, 3, 4} indicating the stage within a world
  --render RENDER       Whether to render the environment
  --version VERSION     Set a number in {0, 1, 2, 3} specifying the ROM mode to use
  --action_type ACTION_TYPE
                        Set game difficulty in {right_only, simple, complex}
  --lr LR               Set learning rate
  --gamma GAMMA         Set discount factor for rewards
  --tau TAU             Set parameter for GAE
  --beta BETA           Set entropy coefficient
  --num_local_steps NUM_LOCAL_STEPS
  --num_global_steps NUM_GLOBAL_STEPS
  --num_processes NUM_PROCESSES
                        Set the number of processes
  --save_interval SAVE_INTERVAL
                        Set number of steps between savings
  --max_actions MAX_ACTIONS
                        Set maximum repetition steps in test phase
  --save_model_dir SAVE_MODEL_DIR
                        Path of saved model directory
  --tensorboard TENSORBOARD
                        Path of tensorboard
  --output_video OUTPUT_VIDEO
                        Path of output vidoe directory
  --use_pretrained USE_PRETRAINED
                        Whether to use pretrained model weight
```