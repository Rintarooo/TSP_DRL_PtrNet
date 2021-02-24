# TSP Solver with Deep RL
This is PyTorch implementation of NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING, Bello et al. 2016
[https://arxiv.org/abs/1611.09940]
  
Pointer Networks is the model architecture proposed by Vinyals et al. 2015
[https://arxiv.org/abs/1506.03134]
  
This model uses attention mechanism to output a permutation of the input index.


![Figure_1](https://user-images.githubusercontent.com/51239551/99033373-17e79900-25be-11eb-83c3-c7f4ce50c2be.png)


## Actor-Critic
### Training without supervised solution
In this work, we tackle Traveling Salesman Problem(TSP), which is one of the combinatorial optimization problems known as NP-hard. TSP seeks for the shortest tour for a salesman to visit each city exactly once.

In the training phase, this TSP solver optimizes 2 different types of Pointer Networks, Actor and Critic model. 

Given a graph of cities where the cities are the nodes, critic model predicts expected tour length, which is generally called state-value. Parameters of critic model are optimized as estimated tour length catch up with the actual length calculated from the tour(city permutation) predicted by actor model. Actor model updates its policy parameters with the value called advantage which subtracts state-value from the actual tour length.

``` 
Actor:  Defines the agent's behavior, its policy
Critic: Estimates the state-value 
```
<br><br>

## Active Search and Sampling
### Inference
In this paper, two approaches to find the best tour at inference time are proposed, which we refer to as Sampling and Active Search. 

Search strategy called Active Search takes actor model and use policy gradient for updating its parameters to find the shortest tour. Sampling simply just select the shortest tour out of 1 batch.

![Figure_13png](https://user-images.githubusercontent.com/51239551/82798619-bae31400-9eb3-11ea-9cf4-59f1c0a49a88.png)


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
After completing training, set the configuration for inference.  
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

### Dependencies
* Python = 3.6.10
* PyTorch = 1.6.0
* numpy
* tqdm (if you need)
* matplotlib (only for plotting)

### Docker(option)
1. build or pull docker image

build image(this might take some time)
```bash
./docker.sh build
```
pull image
```bash
docker pull docker4rintarooo/tspdrl:latest
```

2. run container using docker image(-v option is to mount directory)
```bash
./docker.sh run
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
* https://www.youtube.com/watch?v=mxCVgVrUw50&ab_channel=%D0%9A%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5%D0%BD%D0%B0%D1%83%D0%BA%D0%B8
