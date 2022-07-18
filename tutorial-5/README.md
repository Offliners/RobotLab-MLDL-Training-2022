# Tutorial 5 - Super Mario
|World/Stage|1|2|3|4|
|-|-|-|-|-|
|1|![World 1-1](./img/mario_world_1_1.gif)|Unsolved<!--![World 1-2]()-->|Unsolved<!--![World 1-3]()-->|![World 1-4](./img/mario_world_1_4.gif)|
|2|Unsolved<!--![World 2-1]()-->|![World 2-2](./img/mario_world_2_2.gif)|![World 2-3](./img/mario_world_2_3.gif)|Unsolved<!--![World 2-4]()-->|
|3|![World 3-1](./img/mario_world_3_1.gif)|Unsolved<!--![World 3-2]()-->|Unsolved<!--![World 3-3]()-->|![World 3-4](./img/mario_world_3_4.gif)|
|4|![World 4-1](./img/mario_world_4_1.gif)|Unsolved<!--![World 4-2]()-->|Unsolved<!--![World 4-3]()-->|Unsolved<!--![World 4-4]()-->|
|5|![World 5-1](./img/mario_world_5_1.gif)|Unsolved<!--![World 5-2]()-->|Unsolved<!--![World 5-3]()-->|Unsolved<!--![World 5-4]()-->|
|6|Unsolved<!--![World 6-1]()-->|Unsolved<!--![World 6-2]()-->|Unsolved<!--![World 6-3]()-->|Unsolved<!--![World 6-4]()-->|
|7|Unsolved<!--![World 7-1]()-->|Unsolved<!--![World 7-2]()-->|![World 7-3](./img/mario_world_7_3.gif)|Unsolved<!--![World 7-4]()-->|
|8|Unsolved<!--![World 8-1]())-->|![World 8-2](./img/mario_world_8_2.gif)|![World 8-3](./img/mario_world_8_3.gif)|Unsolved<!--![World 8-4]()-->|

## Usage
在本機端開始訓練
```shell
$ cd python

# 開始訓練
$ python main.py

# 針對特定World或者Stage進行訓練
$ python main.py --world {index of world (1~8)} --stage {index of stage (1~4)}

# 訓練完成後可以看表現如何
$ python eval.py

# 針對特定World或者Stage看表現如何
$ python eval.py --world {index of world (1~8)} --stage {index of stage (1~4)}
```

使用Tensorboard來觀看各個Process的Loss
```shell
$ tensorboard --logdir=./checkpoints/tensorboard/
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