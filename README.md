# TSP Solver with Deep Reinforcement Learning 
This is PyTorch implementation of NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING(Bello et al. 2016(https://arxiv.org/abs/1611.09940).

Pointer Networks(PN) is introduced by Vinyals et al. 2015(https://arxiv.org/abs/1506.03134). This model uses attention to output a permutation of the input index.


![Screen Shot 2020-05-12 at 12 15 35 AM](https://user-images.githubusercontent.com/51239551/81578424-bf082f80-93e5-11ea-812a-914c9046587a.png)


## Actor-Critic Algorithm to train PN on TSP without supervised solution
In this work, we tackle Traveling Salesman Problem(TSP), which is one of the combinatorial optimization problems known as NP-hard. TSP seeks for the shortest tour for a salesman to visit each city exactly once.

In the training phase, this TSP solver optimizes 2 different types of Pointer Networks which are Actor and Critic model. 

Given a graph of cities where the cities are the nodes, critic model predicts expected tour length, which is generally called state-value. Parameters of critic model are optimized as estimated tour length catch up with the actual length calculated from the tour(city permutation) predicted by actor model. Actor model updates its policy parameters with the value called advantage which subtracts state-value from the actual tour length.

``` 
Actor:  Defines the agent's behavior, its policy
Critic: Estimates the state-value 
```


## Active Search and Sampling
In this paper, two approaches to find the best tour at inference time are proposed, which we refer to as Sampling and Active Search. Search strategy called Active Search takes actor model and use policy gradient for updating its parameters to find the shortest tour. Sampling simply just select the shortest tour out of 1 batch.

![Figure_13png](https://user-images.githubusercontent.com/51239551/82798619-bae31400-9eb3-11ea-9cf4-59f1c0a49a88.png)


## Usage

### Training

First generate the pickle file contaning hyperparameter values by running the following command.

```bash
python config.py -m 'train' -b 512 -ct 20 -s 10000
```
-m 'train' could be replaced with -m 'train_emv'. emv is the abbreviation of Exponential Moving Average which doesn't need critic model. Then, go on training.
```bash
python train.py -p './Pkl/train20.pkl' 
```
or
```bash
python train.py -p './Pkl/train_emv20.pkl' 
```

### Inference
After completing training, set the configuration for inference. Now you can see how the training process went from the csv files in the 'Csv' dir. You may use my pre-trained weight './Pt/0825_15_49_step4990_act.pt' in the 'Pt' dir.
```bash
python config.py -m 'test' -ct 20 -s 10 --islogger -ap './Pt/0825_15_49_step4990_act.pt'
```
```bash
python infer.py -p './Pkl/test20.pkl' 
```

## Environment
I leave my own environment below. I tested it out on single GPU.
* OS:
	* Linux(Ubuntu 18.04.5 LTS) 
* GPU:
	* NVIDIA® GeForce® RTX 2080 Ti VENTUS 11GB OC
* CPU:
	* Intel® Xeon® CPU E5640 @ 2.67GHz

### Dependencies
* Python = 3.6.10
* PyTorch = 1.2.0
* tqdm
* numpy
* matplotlib (only for plotting)
