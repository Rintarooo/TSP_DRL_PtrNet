# TSP Solver with Reinforcement Learning 
This is PyTorch implementation of NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING(Bello et al. 2016)(https://arxiv.org/abs/1611.09940).

Pointer Networks is introduced by Vinyals et al. 2015(https://arxiv.org/abs/1506.03134). This model uses attention to output a index permutation of the input.
![Screen Shot 2020-05-12 at 12 15 35 AM](https://user-images.githubusercontent.com/51239551/81578424-bf082f80-93e5-11ea-812a-914c9046587a.png)

## Actor-Critic algorithm to train the Pointer Networks on TSP without supervised solution
In this work, we tackle Traveling Salesman Problem(TSP), which is one of the combinatorial optimization problems known as NP-hard. TSP seeks for the shortest tour for a salesman to visit each city exactly once.

In the training phase, this TSP solver optimizes 2 different types of Pointer Networks which are Actor and Critic model. Given a graph of cities where the cities are the nodes, critic model predicts expected tour length, which is generally called state value. Parameters of critic model are optimized as estimated tour length catch up with actual length calculated from the tour(city permutation) predicted by actor model. Actor model updates its policy parameters with advantage value which subtracts state value from actual tour length.

## Active Search
In this paper, two approaches to find the best tour at inference time are proposed, which we refer to as Sampling and Active Search. Search strategy called Active Search takes actor model and use policy gradient for updating its parameters to find the best tour.
![Figure_1](https://user-images.githubusercontent.com/51239551/81577550-8451c780-93e4-11ea-8ab3-3b412caf1fcb.png)
