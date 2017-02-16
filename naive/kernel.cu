#include <stdio.h>
#include "../debug.h"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

#define MAX_TIME_STEP 30
#define INPUT_DIM 128
#define BATCH_SIZE 64

/* Macro for index calculations */
#define INDX( time_step, batch_index, col ) ( (time_step + BATCH_SIZE * (batch_index + INPUT_DIM * col)) )

__global__ void fpool_GPU(double* h,  const double* z, const double* f) {
	for(int t = 1; t < MAX_TIME_STEP; t++) {
		// detemine this thread's index in the batch and input dims
		const int mybatch = blockDim.x * blockIdx.x + threadIdx.x;
		const int mycol = blockDim.y * blockIdx.y + threadIdx.y;

		if(mybatch < BATCH_SIZE && mycol < INPUT_DIM) {
			h[INDX(t, mybatch, mycol)] = f[INDX(t, mybatch, mycol)]	* h[INDX(t-1, mybatch, mycol)] + (1 - f[INDX(t, mybatch, mycol)]) * z[INDX(t, mybatch, mycol)];
		}
	}
	return;
}

void fpool_CPU(double* h,  const double* z, const double* f) {
	for(int t = 1; t < MAX_TIME_STEP; t++) {
		for(int row = 1; row < BATCH_SIZE; row++) {
			for(int col = 1; col < BATCH_SIZE; col++) {
				h[INDX(t, row, col)] = f[INDX(t, row, col)]	* h[INDX(t-1, row, col)] + (1 - f[INDX(t, row, col)]) * z[INDX(t, row, col)];
			}
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
	dim3 blocks( BATCH_SIZE / THREADS_PER_BLOCK_X + 1, 
			   INPUT_DIM / THREADS_PER_BLOCK_Y + 1, 1 );

	/*float elapsedTime;*/
	/*cudaEvent_t start, stop;*/
	/*checkCUDA( cudaEventCreate( &start ) );*/
	/*checkCUDA( cudaEventCreate( &stop ) );*/
	/*checkCUDA( cudaEventRecord( start, 0 ) );*/

	/*fpool_GPU<<< blocks, threads >>> (h, z, f);*/
	/*checkKERNEL();*/
	fpool_CPU(h, z, f);

	/*checkCUDA( cudaEventRecord( stop, 0 ) );*/
	/*checkCUDA( cudaEventSynchronize( stop ) );*/
	/*checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );*/

	cudaDeviceSynchronize();


	for(int i=0; i<BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM; i++) {
		printf("%f\n", h[i]);
	}

	cudaFree(h);
	cudaFree(z);
	cudaFree(f);

	return 0;
}
