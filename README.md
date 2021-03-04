# TSP Solver with Deep RL
This is PyTorch implementation of NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING, Bello et al. 2016
[https://arxiv.org/abs/1611.09940]
  
Pointer Networks is the model architecture proposed by Vinyals et al. 2015
[https://arxiv.org/abs/1506.03134]
  
This model uses attention mechanism to output a permutation of the input index.

![Screen Shot 2021-02-25 at 12 45 34 AM](https://user-images.githubusercontent.com/51239551/109026426-13756500-7703-11eb-9880-6b8be0b47b4e.png)

<br><br>
In this work, we will tackle Traveling Salesman Problem(TSP), which is one of the combinatorial optimization problems known as NP-hard. TSP seeks for the shortest tour for a salesman to visit each city exactly once.

## Training without supervised solution

In the training phase, this TSP solver will optimize 2 different types of Pointer Networks, Actor and Critic model. 

Given a graph of cities where the cities are the nodes, critic model predicts expected tour length, which is generally called state-value. Parameters of critic model are optimized as the estimated tour length catches up with the actual length calculated from the tour(city permutation) predicted by actor model. Actor model updates its policy parameters with the value called advantage which subtracts state-value from the actual tour length.

### Actor-Critic
``` 
Actor:  Defines the agent's behavior, its policy
Critic: Estimates the state-value 
```

<br><br>

## Inference
### Active Search and Sampling
In this paper, two approaches to find the best tour at inference time are proposed, which we refer to as Sampling and Active Search. 

Search strategy called Active Search takes actor model and use policy gradient for updating its parameters to find the shortest tour. Sampling simply just select the shortest tour out of 1 batch.

![Figure_1](https://user-images.githubusercontent.com/51239551/99033373-17e79900-25be-11eb-83c3-c7f4ce50c2be.png)

## Usage

### Training

First generate the pickle file contaning hyperparameter values by running the following command  

(in this example, train mode, batch size 512, 20 city nodes, 13000 steps).

```bash
python config.py -m train -b 512 -t 20 -s 13000
```
`-m train` could be replaced with `-m train_emv`. emv is the abbreviation of 'Exponential Moving Average', which doesn't need critic model. Then, go on training.
```bash
python train.py -p Pkl/train20.pkl
```  
<br><br>

### Inference
If training is done, set the configuration for inference.  
Now, you can see how the training process went from the csv files in the `Csv` dir.  
You may use my pre-trained weight `Pt/train20_1113_12_12_step14999_act.pt` which I've trained for 20 nodes'.
```bash
python config.py -m test -t 20 -s 10 -ap Pt/train20_1113_12_12_step14999_act.pt --islogger --seed 123
```
```bash
python test.py -p Pkl/test20.pkl
```
<br><br>

## Environment
I leave my own environment below. I tested it out on a single GPU.
* OS:
	* Linux(Ubuntu 18.04.5 LTS) 
* GPU:
	* NVIDIA® GeForce® RTX 2080 Ti VENTUS 11GB OC
* CPU:
	* Intel® Xeon® CPU E5640 @ 2.67GHz
* NVIDIA® Driver = 455.45.01
* Docker = 20.10.3
* [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)(for GPU)

### Dependencies
* Python = 3.6.10
* PyTorch = 1.6.0
* numpy
* tqdm (if you need)
* matplotlib (only for plotting)

### Docker(option)
Make sure you've already installed `Docker`
```bash
docker version
```
latest `NVIDIA® Driver`
```bash
nvidia-smi
```
and `nvidia-docker2`(for GPU)
<br>
#### Usage

1. build or pull docker image

build image(this might take some time)
```bash
./docker.sh build
```
pull image from [dockerhub](https://hub.docker.com/repository/docker/docker4rintarooo/tspdrl/tags?page=1&ordering=last_updated)
```bash
docker pull docker4rintarooo/tspdrl:latest
```

2. run container using docker image(-v option is to mount directory)
```bash
./docker.sh run
```
If you don't have a GPU, you can run
```bash
./docker.sh run_cpu
```
<br><br>

## Reference
* https://github.com/higgsfield/np-hard-deep-reinforcement-learning
* https://github.com/zhengsr3/Reinforcement_Learning_Pointer_Networks_TSP_Pytorch
* https://github.com/pemami4911/neural-combinatorial-rl-pytorch
* https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow
* https://github.com/jingw2/neural-combinatorial-optimization-rl
* https://github.com/dave-yxw/rl_tsp
* https://github.com/shirgur/PointerNet
* https://github.com/MichelDeudon/encode-attend-navigate
* https://github.com/qiang-ma/HRL-for-combinatorial-optimization
* https://www.youtube.com/watch?v=mxCVgVrUw50&ab_channel=%D0%9A%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5%D0%BD%D0%B0%D1%83%D0%BA%D0%B8
