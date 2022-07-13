![meme](meme.jpg)

# NTUME Robot Lab - ML/DL Training
2022/07/23 released

Report : [pdf](20220723_training.pdf)

## Environment
* Python 3.8.13
* CUDA 11.3
* Pytorch 1.11

## Usage
```shell
# 使用Anaconda建立名為training的虛擬環境
$ conda create --name training python=3.8.13

# 啟動名為training的虛擬環境
$ conda activate training

$ git clone https://github.com/Offliners/RobotLab-MLDL-Training-2022.git
$ cd RobotLab-MLDL-Training-2022

# 安裝此專案所需的所有函式庫
$ pip install -r requirements.txt

# 關閉虛擬環境
$ conda deactivate

# 若之後確定不會使用的話，可以移除虛擬環境
$ conda env remove -n training
```

## Tutorial 1 - Covid19 Cases Prediction

Links : [README]() [Google Colab]() 

## Tutorial 2 - CIFAR 10

Links : [README]() [Google Colab]() 

## Tutorial 3 - Walking Scene Segmentation

Links : [README]() [Google Colab]() 

## Tutorial 4 - Anime Face Generation

Links : [README]() [Google Colab]() 

## Tutorial 5 - 2-Dof Robot

Links : [README]() [Google Colab]() 

## References
* [Pytorch official tutorials](https://pytorch.org/tutorials/)
* [Pytorch lightning official tutorials](https://www.pytorchlightning.ai/tutorials)
* [CS 285 at UC Berkeley - Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)
* [NTU Machine Learning 2021 Spring(EE5184, by Prof. Hung-yi Lee)](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)