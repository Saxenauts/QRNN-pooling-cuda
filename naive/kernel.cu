#include <stdio.h>
#include "../debug.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define MAX_TIME_STEP 50
#define INPUT_DIM 1024
#define BATCH_SIZE 128

/* Macro for index calculations */
#define INDX( time_step, batch_index, col ) ( (time_step + MAX_TIME_STEP * (batch_index + BATCH_SIZE * col)) )

__global__ void fpool_GPU(float* h,  const float* z, const float* f) {

	for(int t = 1; t < MAX_TIME_STEP; t++) {
		// detemine this thread's index in the batch and input dims
		const int mybatch = blockDim.x * blockIdx.x + threadIdx.x;
		const int mycol = blockDim.y * blockIdx.y + threadIdx.y;

		int index = INDX(t, mybatch, mycol);
		int prev_index = INDX(t-1, mybatch, mycol);

		if(mybatch < BATCH_SIZE && mycol < INPUT_DIM) {
			h[index] = f[index]	* h[prev_index] + (1 - f[index]) * z[index];
		}
	}
	return;
}

void fpool_CPU(float* h,  const float* z, const float* f) {
	for(int t = 1; t < MAX_TIME_STEP; t++) {
		for(int row = 0; row < BATCH_SIZE; row++) {
			for(int col = 0; col < INPUT_DIM; col++) {
				int index = INDX(t, row, col);
				int prev_index = INDX(t-1, row, col);
				h[index] = f[index]	* h[prev_index] + (1 - f[index]) * z[index];
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
	float *h;
	cudaMallocManaged(&h, BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM * sizeof(float));
	// convolution outputs
	float *z;
	float *f;
	cudaMallocManaged(&z, BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM * sizeof(float));
	cudaMallocManaged(&f, BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM * sizeof(float));

	srand(37);
	// initialize conv outputs
	for(int i=0; i<BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM; i++) {
		z[i] = float(rand()) / (float(RAND_MAX) + 1.0);
		f[i] = float(rand()) / (float(RAND_MAX) + 1.0);
	}

	/* Naive GPU Test */

	// set pooling initial state to zero
	memset(h, 0, MAX_TIME_STEP * BATCH_SIZE * INPUT_DIM * sizeof(float));
	
	dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
	dim3 blocks( BATCH_SIZE / THREADS_PER_BLOCK_X + 1, 
			   INPUT_DIM / THREADS_PER_BLOCK_Y + 1, 1 );

	float elapsedTime;
	cudaEvent_t start, stop;
	checkCUDA( cudaEventCreate( &start ) );
	checkCUDA( cudaEventCreate( &stop ) );
	checkCUDA( cudaEventRecord( start, 0 ) );

	fpool_GPU<<< blocks, threads >>> (h, z, f);
	cudaDeviceSynchronize();
	checkKERNEL();

	checkCUDA( cudaEventRecord( stop, 0 ) );
	checkCUDA( cudaEventSynchronize( stop ) );
	checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );


	float gpu_sum = 0;
	for(int i=0; i<BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM; i++) {
		gpu_sum += h[i];
	}

	/* CPU Test */
	memset(h, 0, MAX_TIME_STEP * BATCH_SIZE * INPUT_DIM * sizeof(float));

	checkCUDA( cudaEventRecord( start, 0 ) );

	fpool_CPU(h, z, f);

	checkCUDA( cudaEventRecord( stop, 0 ) );
	checkCUDA( cudaEventSynchronize( stop ) );
	checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );

	float cpu_sum = 0;
	for(int i=0; i<BATCH_SIZE * MAX_TIME_STEP * INPUT_DIM; i++) {
		cpu_sum += h[i];
	}

	float error = gpu_sum - cpu_sum;
	printf("cpu_sum %f\n", cpu_sum);
	printf("error is %f\n", error);
	if(error > 10)printf("FAIL\n");
	else printf("PASS\n");

	cudaFree(h);
	cudaFree(z);
	cudaFree(f);

	return 0;
}
