#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <iostream>

#define INF 99999
#define N   36     // vertex number

using namespace std;

void printDistanceMatrix(int **dist, int n){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			if (dist[i][j] == INF)
				printf("INF  ");
			else
				printf("%3d  ", dist[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]){

	// create graph
	int graph[N][N] = {
		{0  ,1  ,INF,INF,4  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,0  ,2  ,INF,3  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,0  ,3  ,2  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,0  ,4  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,0  ,5  ,2  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,20 ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{6  ,INF,INF,INF,INF,0  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,0  ,1  ,INF,INF,INF,INF,6  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,0  ,2  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,0  ,3  ,2  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,4  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,5  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,6  ,INF,3  ,INF,INF,0  ,1  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,1  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,2  ,INF,3  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,3  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,4  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,5  ,INF,INF,INF,INF,INF,10 ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,6  ,INF,INF,INF,INF,0  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,INF,2  ,INF,INF,6  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,5  ,0  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,4  ,0  ,INF,2  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,3  ,0  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,14 },
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,2  ,0  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,2  ,1  ,0  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,1  ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{25 ,INF,INF,10 ,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,2  ,INF,INF,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,1  ,2  ,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,3  ,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,4  ,3  ,INF,INF,0  ,INF,INF,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,0  ,INF,INF,INF,INF,6  ,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,2  ,5  ,0  ,INF,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,4  ,0  ,INF,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,2  ,3  ,0  ,INF,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,2  ,0  ,INF,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,3  ,INF,1  ,0  ,INF},
		{INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,1  ,0  }};

	// create distance matrix
	int **dist = new int*[N];
	for (int i = 0; i < N; i++){
		dist[i] = new int[N];
		for (int j = 0; j < N; j++)
			dist[i][j] = graph[i][j];
	}

	double start_time, elapsed_time;
	int nProc, rank, rank2d;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	start_time = MPI_Wtime();

	int q = sqrt(nProc); // side of the processor matrix
	int dims[2] = {q, q},
		periods[2] = {1,1},
		comm_coords[2];

	// create cartesian communicator
	MPI_Comm COMM_2D, COMM_ROW, COMM_COL;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &COMM_2D);
	MPI_Comm_rank(COMM_2D, &rank2d);
	MPI_Cart_coords(COMM_2D, rank2d, 2, comm_coords);

	// create row and column subgroup communicators
	int keep_dims[2] = {0,1};
	MPI_Cart_sub(COMM_2D, keep_dims, &COMM_ROW);
	keep_dims[0] = 1;
	keep_dims[1] = 0;
	MPI_Cart_sub(COMM_2D, keep_dims, &COMM_COL);

	int rank_col, rank_row;
	MPI_Comm_rank(COMM_COL, &rank_col);
	MPI_Comm_rank(COMM_ROW, &rank_row);

	// sqrt of number of elements in each processor's matrix (side of the square)
	int t = N / q;
	// get indexes for the square that this process is responsible
	int i_min = comm_coords[0] * t,
		i_max = comm_coords[0] * t + t,
		j_min = comm_coords[1] * t,
		j_max = comm_coords[1] * t + t;

	MPI_Request req; // store the async send request

	// main loop for K
	for (int k = 0; k < N; k++){
		int p = k/t; // rank of the processor from which we need to get data from

		// receive data from processes in the column
		if (k > 0 && p != rank_col) // if it's not the first iteration we need data from the previous iteration
			// we need a column from another process on this process column
			for (int o = 0; o < t; o++) // to get a column from this process we just get the entire process matrix,
				MPI_Recv(dist[p*t+o]+j_min, t, MPI_INT, p, k-1, COMM_COL, MPI_STATUS_IGNORE);

		for (int i = i_min; i < i_max; i++){

			// receive line data from processes in the row
			if (k > 0 && p != rank_row) // we need a line from another process on this process row
				MPI_Recv(dist[i]+(p*t), t, MPI_INT, p, k-1, COMM_ROW, MPI_STATUS_IGNORE);

			// solve process line
			for (int j = j_min; j < j_max; j++)
				dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);

			// send line to other processes
			for (int l = 0; k < N-1 && l < q; l++){ // we don't need to run this for the last iteration of K
				if (l != rank_row) // send this line to other processes in the row
					MPI_Isend(dist[i] + j_min, t, MPI_INT, l, k, COMM_ROW, &req);
				if (l != rank_col) // send this line to other processes in the column
					MPI_Isend(dist[i] + j_min, t, MPI_INT, l, k, COMM_COL, &req);
			}
		}
	}

	// array to gather results from other processes (holds one line of each process matrix)
	int gather_arr[nProc*t];

	// call gather for each line of the processor matrix and copy it to the right place in the final matrix
	for (int i = 0; i < t; i++){
		MPI_Gather(dist[i_min+i]+(t * comm_coords[1]), t, MPI_INT, gather_arr, t, MPI_INT, 0, MPI_COMM_WORLD);
		if (rank == 0){
			// copy elements from received array into matrix
			for (int j = 0; j < nProc*t; j++){
				dist[i+(j/N*t)][j%N] = gather_arr[j];
			}
		}
	}

	elapsed_time = MPI_Wtime () - start_time;

	MPI_Finalize();

	if (rank == 0){
		printDistanceMatrix(dist, N);
		printf("\n");
        printf("Time taken: %fs\n", elapsed_time);
	}

	// free memory
	for (int i = 0; i < N; i++) delete dist[i];
	delete[] dist;

	return 0;
}