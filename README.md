## Actor-Critic algorithm to train the Pointer Networks on TSP without supervised solution
This is implementation of NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING(Bello et al, 2016)(https://arxiv.org/abs/1611.09940).

## The code is still in progress

In this work, we tackle Traveling Salesman Problem(TSP), which is one of the combinatorial optimization problems known as NP-hard. TSP seeks for the shortest tour for a salesman to visit each city exactly once.

In the training phase, this TSP solver optimizes 2 different types of Pointer Networks which are Actor and Critic. Given a graph of cities where the cities are the nodes, critic model predicts expected tour length, which is generally called state value. Parameter of critic model is optimized as estimated tour length catch up with actual length sampled from the tour(city permutation) predicted by actor model. Actor model updates its policy parameter with advantage value which subtracts state value from actual tour length.

In the test phase, search strategy called Active Search takes actor model and use policy gradient to search for the best tour.
