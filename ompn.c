#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>

typedef double TYPE;
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

// Method signatures
TYPE** randomSquareMatrix(int dimension);
TYPE** zeroSquareMatrix(int dimension);

// Test cases
void MultiplyTest(int dimension);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

int main(int argc, char* argv[]){

	for(int dimension=200; dimension<=2000; dimension+=200){
		MultiplyTest(dimension);
	}

	return 0;
}

TYPE** randomSquareMatrix(int dimension){
	/*
		Generate 2 dimensional random TYPE matrix.
	*/

	TYPE** matrix = malloc(dimension * sizeof(TYPE*));

	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}
	
	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
		}
	}

	return matrix;
}

TYPE** zeroSquareMatrix(int dimension){
	/*
		Generate 2 dimensional zero TYPE matrix.
	*/

	TYPE** matrix = malloc(dimension * sizeof(TYPE*));

	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = 0;
		}
	}

	return matrix;
}

double BlockMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int size){
	/*
		Parallel multiply given input matrices and return resultant matrix
	*/
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	float tmp;
	int block_size = 16;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	{
		for (jj = 0; jj < size; jj += block_size)
		{
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0.0f;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += matrixA[i][k] * matrixB[k][j];
						}
						matrixC[i][j] += tmp;
					}
				}
			}
		}
	}
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

double BlockMultiplyPragma(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int size){
	/*
		Parallel multiply given input matrices and return resultant matrix
	*/
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	float tmp;
	int chunk = 1;
	int block_size = 16;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
#pragma omp parallel shared(matrixA, matrixB, matrixC, size, chunk) private(i, j, k, jj, kk, tmp)
	{
		#pragma omp for schedule (static, chunk)
		for (jj = 0; jj < size; jj += block_size)
		{
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0.0f;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += matrixA[i][k] * matrixB[k][j];
						}
						matrixC[i][j] += tmp;
					}
				}
			}
		}
	}
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}
void MultiplyTest(int dimension){

	// Console write
	printf("\n           ----------------------------------\n");
	printf("                 Dimension : %d x %d\n", dimension,dimension);
	printf("           ----------------------------------\n");

	double* opmLatency = malloc(2 * sizeof(double));
	TYPE** matrixA = randomSquareMatrix(dimension);
	TYPE** matrixB = randomSquareMatrix(dimension);
	
	// Iterate and measure performance
		TYPE** matrixResult = zeroSquareMatrix(dimension);
		printf("\n     Processing matrix multiplication without pragma ......\n");
		opmLatency[1] = BlockMultiply(matrixA, matrixB, matrixResult, dimension);
		printf("     Matrix Multiplication without Pragma latency:\t%f\n", opmLatency[1]);
		printf("\n     Processing matrix multiplication with pragma......\n");
		opmLatency[0] = BlockMultiplyPragma(matrixA, matrixB, matrixResult, dimension);
		printf("     Matrix Multiplication with Pragma latency:   \t%f\n", opmLatency[0]);
		free(matrixResult);
    
	// Releasing memory
	free(opmLatency);
	free(matrixA);
	free(matrixB);
}