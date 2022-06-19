// Course:           High Performance Computing
// A.Y:              2021/22
// Lecturer:         Francesco Moscato           fmoscato@unisa.it

// Team:
// Alessio Pepe          0622701463      a.pepe108@studenti.unisa.it
// Teresa Tortorella     0622701507      t.tortorella3@studenti.unisa.it
// Paolo Mansi           0622701542      p.mansi5@studenti.unisa.it

// Copyright (C) 2021 - All Rights Reserved

// This file is part of Counting_Sort.

// Counting_Sort is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Counting_Sort is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Counting_Sort.  If not, see <http://www.gnu.org/licenses/>.

/**
 * @file    counting_sort.c
 * @author  Alessio Pepe         (a.pepe108@studenti.unisa.it)
 * @author  Paolo Mansi          (p.mansi5@studenti.unisa.it)
 * @author  Teresa Tortorella    (t.tortorella3@studenti.unisa.it)
 * @version 1.0.0
 * @date 2022-01-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>

/**
 * @brief Use start_time with an non used id to start measure time in that point of the code.
 * 
 */
#define STARTTIME(id)                             \
   struct timeval start_time_##id, end_time_##id; \
   gettimeofday(&start_time_##id, NULL);

/**
 * @brief Use end_tipe with a previous used id to stop measure time in that point of the code.
 *        The value of time will be saved in x.
 * 
 */
#define ENDTIME(id, x)                 \
   gettimeofday(&end_time_##id, NULL); \
   x = ((end_time_##id.tv_sec  - start_time_##id.tv_sec) * 1000000u +  end_time_##id.tv_usec - start_time_##id.tv_usec) / 1.e6;


#define FIXED_ARRAY

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


texture <int, 1> pmfTextRef;


void cudaGetError()
{
    cudaError_t err = cudaGetLastError();  
    if (err != cudaSuccess) 
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err)); 
    }
}


#define cudaCheck(status, err)                              \
    if (status != cudaSuccess) {                            \
        fprintf(stderr, "CUDA check failed: %s\n", err);    \
        cudaGetError();                                     \
        exit(1);                                            \
    }


#define cudaStartTime(id)               \
    cudaEvent_t start##id, stop##id;     \
    cudaEventCreate(&start##id);        \
    cudaEventCreate(&stop##id);         \
    cudaEventRecord(start##id);         \


#define cudaStopTime(id)                \
    cudaEventRecord(stop##id);          \
    cudaEventSynchronize(stop##id);     \


#define cudaElapsedTime(id, x)                              \
    cudaEventElapsedTime(&x, start##id, stop##id);          \
    cudaEventDestroy(start##id);                            \
    cudaEventDestroy(stop##id);                             \


void init_rand_vector(int *A, int A_len, int min_value, int max_value)
{ 
    #ifdef FIXED_ARRAY
    srand(1256765);
    #endif
 
    for (unsigned int i = 0; i < A_len; i++)
    {
        A[i] = min_value + (rand() % (max_value - min_value + 1));
    }
}

void printV(int *array, int len)
{
    for (unsigned int i = 0; i < len; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}


__global__ void max_min(int *d_max_A, int *d_min_A, int *d_data, int d_data_len)
{
    extern __shared__ int arr[];
    int *s_min = arr;
    int *s_max = arr + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread loads one element from global to shared mem
    s_min[tid] = d_data[i];
    s_max[tid] = d_data[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
    {  
        if ((tid < s) && ((i + s) < d_data_len))
        {
            s_min[tid] = MIN(s_min[tid], s_min[tid + s]);
            s_max[tid] = MAX(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        d_min_A[blockIdx.x] = s_min[0];
        d_max_A[blockIdx.x] = s_max[0];
    }
}

__global__ void max_min_red(int *d_max_A, int *d_min_A, int d_len)
{
    extern __shared__ int arr[];
    int *s_min = arr;
    int *s_max = arr + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread loads one element from global to shared mem
    s_min[tid] = d_min_A[i];
    s_max[tid] = d_max_A[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
    {  
        if ((tid < s) && ((i + s) < d_len))
        {
            s_min[tid] = MIN(s_min[tid], s_min[tid + s]);
            s_max[tid] = MAX(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        d_min_A[blockIdx.x] = s_min[0];
        d_max_A[blockIdx.x] = s_max[0];
    } 
}


__global__ void pmf_count(int *d_data, int d_data_len, int *d_data_max, int *d_data_min, int *d_pmf_data)
{
    // init a shared pmf array for each block
    extern __shared__ int s_pmf[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // set initial value to 0
    int range = *d_data_max - *d_data_min + 1;

    for (int offset = 0; offset < range; offset += blockDim.x)
    {
        if ((tid + offset) < range)
        {
            s_pmf[tid + offset] = 0;
        }
    }

    // wait until all threads have completed the initialization process
    __syncthreads();

    // increment the local pdf array
    if (i < d_data_len)
    {
        atomicAdd(&s_pmf[d_data[i] - *d_data_min], 1);
    }
    
    // wait until all threads have completed the counting process
    __syncthreads();

    // merge the various pdf array
    for (int offset = 0; offset < range; offset += blockDim.x)
    {
        if ((tid + offset) < range)
        {
            atomicAdd(&d_pmf_data[tid + offset], s_pmf[tid + offset]);
        }
    }
}


__global__ void scan(int *d_pmf, int d_len)
{
    extern __shared__ int scan_a[];

    int i, j, tid;

    tid = threadIdx.x;
    j = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    // Copy array in block
    if (j < d_len)
    {
        scan_a[tid] = d_pmf[j];
    }
    
    if ((j + blockDim.x) < d_len)
    {
        scan_a[tid + blockDim.x] = d_pmf[j + blockDim.x];
    }

    __syncthreads();

    // Scan 
    for (int stride = 1; stride <= blockDim.x; stride <<= 1)
    {
        i = (threadIdx.x + 1) * stride * 2 - 1;
        if (i < 2 * blockDim.x)
        {
            scan_a[i] += scan_a[i - stride];
        }
        __syncthreads();
    }

    // Post scan
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        i = (threadIdx.x + 1) * stride * 2 - 1;
        if ((i + stride) < 2 * blockDim.x)
        {
            scan_a[i + stride] += scan_a[i];
        }
        __syncthreads();
    }

    // Copy partially cdf in the global memory
    if (j < d_len)
    {
        d_pmf[j] = scan_a[tid];
    }

    if ((j + blockDim.x) < d_len)
    {
        d_pmf[j + blockDim.x] = scan_a[tid + blockDim.x];
    }
}

__global__ void scan_red(int *d_pmf, int d_len, int stride)
{
    // First block was already complete
    if (blockIdx.x % 2 == 0)
    {
        return;
    }

    // Copy last element of the previous block on all block element.
    int i = blockIdx.x * (stride * blockDim.x) + threadIdx.x;
    int prec_sum = d_pmf[blockIdx.x * (stride * blockDim.x) - 1];

    for (int j = 0; j < stride; j++)
    {
        if ((i + j * blockDim.x) < d_len)
        {
            d_pmf[i + j * blockDim.x] += prec_sum;
        }
    }
}

__global__ void populate(int *d_data, int *d_data_min, int *d_cdf, int d_cdf_len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < d_cdf_len)
    {
        int start = i != 0 ? d_cdf[i-1] : 0;
        for (int j = 0; j < d_cdf[i] - start; j++)
        {
            d_data[start + j] = *d_data_min + i;
        }
    }
}

__global__ void populate_text(int *d_data, int *d_data_min, int d_cdf_len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < d_cdf_len)
    {
        int start = i != 0 ? tex1Dfetch(pmfTextRef, i-1) : 0;
        for (int j = 0; j < tex1Dfetch(pmfTextRef, i) - start; j++)
        {
            d_data[start + j] = *d_data_min + i;
        }
    }
}


/**
 * This GPU kernel takes an array of states, and an array of ints, and puts a random int into each 
 */
__global__ void randoms(int* numbers, int len, int min, int max, int seed) 
{   
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // curand works like rand - except that it takes a state as a parameter
    //curandState state;
    if (i < len)
    {
        //curand_init(seed, i, 0, &state);
        if (i % 3 == 0)
        {
            numbers[i] = min + (i * seed / blockDim.x) % (max - min + 1);
        }
        else if (i % 3 == 1)
        {
            numbers[i] = min + (i + seed / blockDim.x) % (max - min + 1);
        }
        else
        {
            numbers[i] = min + (seed / blockDim.x - i) % (max - min + 1);
        }
    }  
}

void cuda_init_rand_vector(int gridSize, int blockSize, int *h_A, int h_len, int min, int max)
{
    /* allocate an array of unsigned ints on the CPU and GPU */
    int *d_A1;
    cudaMalloc((void**) &d_A1, h_len * sizeof(int));

    /* invoke the kernel to get some random numbers */
    randoms <<<gridSize, blockSize>>> (d_A1, h_len, min, max, 2342);

    /* copy the random numbers back */
    cudaMemcpy(h_A, d_A1, h_len * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A1);
}


// main
int main(int argc, char** argv) {

    // ------------------------------------------------------
    // Read parameter from argv
    if (argc < 8)
    {
        printf("USAGE: %s len min max blockMinMAx blockPmf blockScan blockPopulate\n", argv[0]);
        exit(1);
    }
    int min = atoi(argv[2]);  // Just for generation 
    int max = atoi(argv[3]);  // Just for generation
    int h_len = atoi(argv[1]);
    // Bench parameters
    int blockMinMax = atoi(argv[4]);
    int blockPmf = atoi(argv[5]);
    int blockScan = atoi(argv[6]);
    int blockPopulate = atoi(argv[7]);

    // --------------- Random Array Generation --------------
    int *h_A;
    h_A = (int *) malloc(h_len * sizeof(int));
    //init_rand_vector(h_A, h_len, min, max); // sequential init
    cuda_init_rand_vector((int) ceilf((float) h_len / (float) blockMinMax), blockMinMax, h_A, h_len, min, max);
    cudaGetError();
    //printV(h_A, h_len);  // Debug print

    double t_algo;
    STARTTIME(0);

    cudaStartTime(0);

    // -------------- Define block and grid size ------------
    // The block size was the maximum (1024). Grid size was
    // selected dinamically to cover all tha array.
    dim3 blockSizeMinMax(blockMinMax);
    dim3 gridSizeMinMax((int) ceilf((float) h_len / (float) blockSizeMinMax.x));

    // ------------ Allocate space on cuda ------------------
    // - Array
    // - Array of local max and min
    // - global max and min
    int *d_A, *d_min_A, *d_max_A, *d_min, *d_max;

    cudaCheck( cudaMalloc((void **)&d_A, h_len * sizeof(int)), "Allocation d_A" );
    cudaCheck( cudaMalloc((void **)&d_min_A, gridSizeMinMax.x * sizeof(int)), "Allocation d_min_A" );
    cudaCheck( cudaMalloc((void **)&d_max_A, gridSizeMinMax.x * sizeof(int)), "Allocation d_max_A" );
    cudaCheck( cudaMalloc((void **)&d_min, sizeof(int)), "Allocation d_min" );
    cudaCheck( cudaMalloc((void **)&d_max, sizeof(int)), "Allocation d_max" );

    // ------------- Copy array to gpu ----------------------
    cudaCheck( cudaMemcpy(d_A, h_A, h_len * sizeof(int), cudaMemcpyHostToDevice), "memcpy h_A to d_A");

    // ------------- Max&Min Kernels ------------------------
    cudaStartTime(1);

    max_min <<<gridSizeMinMax, blockSizeMinMax, 2 * blockSizeMinMax.x * sizeof(int) >>> (d_max_A, d_min_A, d_A, h_len); // Now we have an array of gridSize.x local minimum
    cudaGetError();

    int old_grid_size;
    dim3 redGridSize(gridSizeMinMax.x);
    do 
    {
        old_grid_size = redGridSize.x;
        redGridSize.x = (int) ceilf((float) redGridSize.x / (float) gridSizeMinMax.x);
        // printf("Running with %d, %d, %d\n", redGridSize.x, blockSizeMinMax.x, old_grid_size);
        max_min_red <<<redGridSize, blockSizeMinMax, 2 * blockSizeMinMax.x * sizeof(int)>>> (d_max_A, d_min_A, old_grid_size);
        cudaGetError();
    }
    while (redGridSize.x != 1);

    cudaStopTime(1);

    // ----------- DEBUG: Print max and min ---------------------
    /*int *h_max_A, *h_min_A;

    h_max_A = (int *) malloc( sizeof(int));
    h_min_A = (int *) malloc( sizeof(int));

    cudaCheck(cudaMemcpy((void *) h_min_A, (const void *)d_min_A, sizeof(int), cudaMemcpyDeviceToHost), "memcpy h_min_A");
    cudaCheck(cudaMemcpy((void *) h_max_A, (const void *)d_max_A, sizeof(int), cudaMemcpyDeviceToHost), "memcpy h_max_A");

    printf("Min: ");
    printV(h_min_A, 1); //gridSize.x);
    printf("\nMax: ");
    printV(h_max_A, 1); //gridSize.x);
    printf("\n");

    free(h_max_A);
    free(h_min_A);
    // ----------------------------------------------------------*/

    // -------------- Compute PMF --------------------------------
    int h_max, h_min;
    cudaCheck( cudaMemcpy((void *) &h_max, (const void *)d_max_A, sizeof(int), cudaMemcpyDeviceToHost), "memcpy d_max_A[0] to h_max");
    cudaCheck( cudaMemcpy((void *) &h_min, (const void *)d_min_A, sizeof(int), cudaMemcpyDeviceToHost), "memcpy d_min_A[0] to h_min");
    
    int range_size = (h_max - h_min + 1);

    int *d_pmf;
    cudaCheck( cudaMalloc((void **) &d_pmf, range_size * sizeof(int)), "Allocate d_pmf" );
    cudaCheck( cudaMemset((void *) d_pmf, 0, range_size * sizeof(int)), "memcpy d_pmf" );  

    dim3 blockSizePmf(blockPmf);
    dim3 gridSizePmf((int) ceilf((float) h_len / (float) blockSizePmf.x));

    cudaStartTime(2);
    pmf_count <<< gridSizePmf, blockSizePmf, range_size * sizeof(int) >>> (d_A, h_len, d_max_A, d_min_A, d_pmf);
    cudaGetError();
    cudaStopTime(2);

    // ---------------- CDF calculate ------------------------
    dim3 cdfBlockDim(blockScan);
    dim3 cdfGridDim((int) ceilf((float) range_size / (float) cdfBlockDim.x * 2.f));   

    cudaStartTime(3);
    scan <<< cdfGridDim, cdfBlockDim, 2 * cdfBlockDim.x * sizeof(int) >>> (d_pmf, range_size);
    cudaGetError();

    int new_gridDim = cdfGridDim.x;
    int stride = 2;
    while (new_gridDim != 1)
    {
        cdfGridDim.x = new_gridDim % 2 == 0 ? new_gridDim : new_gridDim - 1;

        scan_red <<< cdfGridDim, cdfBlockDim >>> (d_pmf, range_size, stride);
        cudaGetError();

        new_gridDim = (int) ceilf((float) new_gridDim / 2.f);
        stride *= 2;
    }
    cudaStopTime(3);

    // Debug print 
    /*int *h_pmf;
    h_pmf = (int *) malloc(range_size * sizeof(int));
    cudaCheck(cudaMemcpy(h_pmf, d_pmf, range_size * sizeof(int), cudaMemcpyDeviceToHost), "memcpy d_pmf to h_pmf");
    int k = 0;
    while (k < range_size)
    {
        printV(h_pmf+k, 1024);
        k+= 1024;
        if (k%2048 == 0) printf("\n");
    } 
    free(h_pmf);    // */

    // --------------- Populate array ------------------------
    // To populate we use d_pdf in texture memory
    cudaChannelFormatDesc pmfChRef = cudaCreateChannelDesc <int> ();
    cudaCheck( cudaBindTexture(0, pmfTextRef, d_pmf, pmfChRef), "bindTexture d_pmf" );

    dim3 populateBlockSize(blockPopulate);
    dim3 populateGridSize((int) ceilf((float) range_size / (float) populateBlockSize.x));

    cudaStartTime(4);
    populate_text <<<populateGridSize, populateBlockSize>>> (d_A, d_min_A, /*d_pmf,*/ range_size);
    cudaGetError();
    cudaStopTime(4);

    cudaCheck( cudaUnbindTexture(pmfTextRef), "unbind texture d_pmf" );

    // --------------- Copy array to CPU ---------------------
    cudaCheck( cudaMemcpy(h_A, d_A, h_len * sizeof(int), cudaMemcpyDeviceToHost), "memcpy d_A to h_A");   

    // ------------------- Free --------------------------
    cudaCheck( cudaFree(d_A), "Free d_A" );
    cudaCheck( cudaFree(d_min_A), "Free d_min_A" );
    cudaCheck( cudaFree(d_max_A), "Free d_max_A" );
    cudaCheck( cudaFree(d_min), "Free d_min" );
    cudaCheck( cudaFree(d_max), "Free d_max" );
    cudaCheck( cudaFree(d_pmf), "Free d_pmf_A" );  

    cudaStopTime(0);
    
    ENDTIME(0, t_algo);

    // ------------------- Test working properly ------------
    int flag = 1;
    for (unsigned int i = 1; flag && i < h_len; i++)
    {
        if (h_A[i-1] > h_A[i])
        {
            printf("0");
        }
    }

    float t0, t1, t2, t3, t4;
    cudaElapsedTime(0, t0);
    cudaElapsedTime(1, t1);
    cudaElapsedTime(2, t2);
    cudaElapsedTime(3, t3);
    cudaElapsedTime(4, t4);

    printf("%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f\n", h_len, range_size, blockMinMax, blockPmf, blockScan, blockPopulate, flag, t_algo, t0, t1, t2, t3, t4);

    free(h_A);

    return 0;
}






























