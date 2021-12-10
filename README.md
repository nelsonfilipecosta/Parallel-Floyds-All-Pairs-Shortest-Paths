# Parallel Floyd's All-Pairs Shortest Paths



The implementation of this algorithm was based on chapter 10 and 12 from the textboox

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
