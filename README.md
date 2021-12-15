# Parallel Floyd's All-Pairs Shortest Paths

Consider a weighted graph G, which consists of a set of nodes V and a set of edges E. An edge from node i to node j in E has a weight c_(i,j). Floyd's algorithm determines the cost d_(i,j) of the shortest path between each pair of nodes (i,j) in V. The cost of a path is the sum of the weights of the edges in the path. Let d^k_(i,j) be the minimum cost of a path from node i to node j, then the functional equation for this problem is

![This is an image](https://github.com/nelsonfilipecosta/Parallel-Floyds-All-Pairs-Shortest-Paths/blob/main/Figures/path_cost.png)

In general, the solution is a matrix D^n = [d^n_(i,j)]. Floyd's algorithm solves the above eqution bottom-up in the order of increasing values of k. A generic parallel formulation of Floyd's algorithm assigns the task of computing matrix D^k for each value of k to a set of processes. Let p be the number of processes available. Matrix D^k is partitioned into p parts, and each part is assigned to a process. Each process computes the D^k values of its partition. To accomplish this, a process must access the corresponding segments of the kth row and column of matrix D^(k-1) as shown in the figure below.

![This is an image](https://github.com/nelsonfilipecosta/Parallel-Floyds-All-Pairs-Shortest-Paths/blob/main/Figures/communication_mapping.png)

The theoretical parallel runtime of the algorithm is

![This is an image](https://github.com/nelsonfilipecosta/Parallel-Floyds-All-Pairs-Shortest-Paths/blob/main/Figures/parallel_runtime.png)

The implementation of this algorithm was based on chapters 10 and 12 from the textboox

> Kumar, V., Grama, A., Gupta, A. and Karypis, G. Introduction to parallel computing (Vol. 110). Redwood City, CA: Benjamin/Cummings, 1994.

## Code

Use the Makefile to compile the code and run the following command

```
mpirun -np x lines_cols
```

to find the shortest paths using the parallel Floyd's all-pairs shortest paths algorithm with x processors, or the following command

```
mpirun -np x pipeline
```

to find the shortest paths using the pipeline version of the parallel algorithm with x processors.
