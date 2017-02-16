#include <stdio.h>
#include "debug.h"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

#define MAX_TIME_STEP 30
#define INPUT_DIM 128
#define BATCH_SIZE 64

/* Macro for index calculations */
#define INDX( time_step, batch_index, col ) ( (time_step + BATCH_SIZE * (batch_index + INPUT_DIM * col) )

__global__ void fpool(double* h, double* z, double* f) {
	for(int t = 1; t < MAX_TIME_STEP; t++) {
		// detemine this thread's index in the batch and input dims
		const int mybatch = blockDim.x * blockIdx.x + threadIdx.x;
		const int mycol = blockDim.y * blockIdx.y + threadIdx.y;

		if(mybatch < BATCH_SIZE && mycol < INPUT_DIM) {
			h[INDX(t, mybatch, mycol)] = f[INDX(t, mybatch, mycol)]	* h[INDX(t-1, mybatch, mycol)] +
									(1 - f[INDX(t, mybatch, mycol)]) * z[INDX(t, mybatch, mycol)];
		}
	}
	return;
}

int main(int args, char* argv[])
{
	int dev;
	cudaDeviceProp deviceProp;
	checkCUDA( cudaGetDevice( &dev ) );
	checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
	printf("Using GPU %d: %s\n", dev, deviceProp.name );

	// hidden state
	double *h;
	cudaMallocManaged(&h, BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM * sizeof(double));
	// convolution outputs
	double *z;
	double *f;
	cudaMallocManaged(&z, BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM * sizeof(double));
	cudaMallocManaged(&f, BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM * sizeof(double));

	// initialize conv outputs
	for(int i=0; i<BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM; i++) {
		z[i] = double(rand()) / (double(RAND_MAX) + 1.0);
		f[i] = double(rand()) / (double(RAND_MAX) + 1.0);
	}

	// set pooling initial state to zero
	memset(h, 0, MAX_TIME_STEP);
	
	dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
	dim3 blocks( size / THREADS_PER_BLOCK_X + 1, 
			   size / THREADS_PER_BLOCK_Y + 1, 1 );

	float elapsedTime;

	checkCUDA( cudaEventRecord( start, 0 ) );

	fpool<<< blocks, threads >>> (h, z, f)

	checkCUDA( cudaEventRecord( stop, 0 ) );
	checkCUDA( cudaEventSynchronize( stop ) );
	checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

	cudaDeviceSynchronize();
	for(int i = 0; i<MAX_TIME_STEP; i++) {
		printf("%d: A+B = %f\n", i, z[i]);
	}
	cudaFree(h);
	return 0;
}
