#include <cstdlib>
#include <memory>
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <time.h>
#include <fstream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <typeinfo>
#include <unistd.h>


#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#ifndef __MNIST_H__
#define __MNIST_H__

/*
 * MNIST loader by Nuri Park - https://github.com/projectgalateia/mnist
 */

#ifdef USE_MNIST_LOADER /* Fundamental macro to make the code active */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Make mnist_load function static.
 * Define when the header is included multiple time.
 */
#ifdef MNIST_STATIC
#define _STATIC static
#else
#define _STATIC 
#endif

/*
 * Make mnist loader to load image data as double type.
 * It divides unsigned char values by 255.0, so the results ranges from 0.0 to 1.0
 */
#ifdef MNIST_DOUBLE
#define MNIST_DATA_TYPE double
#else
#define MNIST_DATA_TYPE unsigned char
#endif

typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28]; /* 28x28 data for the image */
	unsigned int label; /* label : 0 to 9 */
} mnist_data;

/*
 * If it's header inclusion, make only function prototype visible.
 */
#ifdef MNIST_HDR_ONLY

_STATIC int mnist_load(
	const char *image_filename,
	const char *label_filename,
	mnist_data **data,
	unsigned int *count);

#else



/*
 * Load a unsigned int from raw data.
 * MSB first.
 */
static unsigned int mnist_bin_to_int(char *v)
{
	int i;
	unsigned int ret = 0;

	for (i = 0; i < 4; ++i) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}

	return ret;
}

/*
 * MNIST dataset loader.
 *
 * Returns 0 if successed.
 * Check comments for the return codes.
 */
_STATIC int mnist_load(
	const char *image_filename,
	const char *label_filename,
	mnist_data **data,
	unsigned int *count)
{
	int return_code = 0;
	int i;
	char tmp[4];

	unsigned int image_cnt, label_cnt;
	unsigned int image_dim[2];

	FILE *ifp = fopen(image_filename, "rb");
	FILE *lfp = fopen(label_filename, "rb");

	if (!ifp || !lfp) {
		return_code = -1; /* No such files */
		goto cleanup;
	}

	fread(tmp, 1, 4, ifp);
	if (mnist_bin_to_int(tmp) != 2051) {
		return_code = -2; /* Not a valid image file */
		goto cleanup;
	}

	fread(tmp, 1, 4, lfp);
	if (mnist_bin_to_int(tmp) != 2049) {
		return_code = -3; /* Not a valid label file */
		goto cleanup;
	}

	fread(tmp, 1, 4, ifp);
	image_cnt = mnist_bin_to_int(tmp);

	fread(tmp, 1, 4, lfp);
	label_cnt = mnist_bin_to_int(tmp);

	if (image_cnt != label_cnt) {
		return_code = -4; /* Element counts of 2 files mismatch */
		goto cleanup;
	}

	for (i = 0; i < 2; ++i) {
		fread(tmp, 1, 4, ifp);
		image_dim[i] = mnist_bin_to_int(tmp);
	}

	if (image_dim[0] != 28 || image_dim[1] != 28) {
		return_code = -2; /* Not a valid image file */
		goto cleanup;
	}

	*count = image_cnt;
	*data = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);

	for (i = 0; i < image_cnt; ++i) {
		int j;
		unsigned char read_data[28 * 28];
		mnist_data *d = &(*data)[i];

		fread(read_data, 1, 28*28, ifp);

#ifdef MNIST_DOUBLE
		for (j = 0; j < 28*28; ++j) {
			d->data[j/28][j%28] = read_data[j] / 255.0;
		}
#else
		memcpy(d->data, read_data, 28*28);
#endif

		fread(tmp, 1, 1, lfp);
		d->label = tmp[0];
	}

cleanup:
	if (ifp) fclose(ifp);
	if (lfp) fclose(lfp);

	return return_code;
}

#endif /* MNIST_HDR_ONLY */

#ifdef __cplusplus
}
#endif

#endif /* USE_MNIST_LOADER */
#endif /* __MNIST_H__ */




const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;



// define Layer class
class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
	//void save_param();
};



// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]);
__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]);
__global__ void fp_bias_s1(float preact[6][6][6], float bias[1]);
__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
__global__ void fp_bias_f(float preact[10], float bias[10]);

// Back propagation kernels
__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]);
__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);


// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

__global__ void custom_snrm2(float* x, int n, float* result)
{
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float sum = 0;
    while (tid < n) {
        sum += x[tid] * x[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = sum;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(result, sqrt(cache[0]));
    }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 5*5*6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 5);
		const int i2 = ((idx /= 5	) % 5);
		const int i3 = ((idx /= 5	) % 6);
		const int i4 = ((idx /= 6	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
	}
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		preact[i1][i2][i3] += bias[i1];
	}
}


__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		preact[i1][i2][i3] += bias[0];
	}
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 5);
		const int i3 = ((idx /= 5	) % 5);
		const int i4 = ((idx /= 5	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}

static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static double test();
static double forward_pass(double data[28][28]);
static double back_pass();
void save_params(Layer* layer,const char* filename);
static void load_params(Layer* layer, char* filename);
static void learn_with_save(std::string dir);
static void test_with_load(std::string dir);

int main(int argc, char* argv[])
{

	std::string dir = argv[1];
	std::string train_images_path = dir + "/../../data/FashionMNIST/raw/train-images-idx3-ubyte";
	std::string train_labels_path = dir + "/../../data/FashionMNIST/raw/train-labels-idx1-ubyte";
	std::string test_images_path = dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte";
	std::string test_labels_path = dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte";
	int ret;
	if (ret = mnist_load(test_images_path.c_str(), test_labels_path.c_str(), &test_set, &test_cnt)) { 
			fflush(stderr);
			fprintf(stderr, "An error occured: %d\n", ret);
			}
	if (ret = mnist_load(train_images_path.c_str(), train_labels_path.c_str(), &train_set, &train_cnt)) {
			fflush(stderr);
			fprintf(stderr, "An error occured: %d\n", ret);
			}
	test_with_load(dir); // 貌似没读上参数，但是我在本地先跑train后跑test是可以的
	// learn();
	// auto start = std::chrono::high_resolution_clock::now();
	// double acc = test();
	// auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;
	// fprintf(stdout, "%.4f:%.4f\n", diff.count(), acc);
	return 0;
}

static void test_with_load(std::string dir){
	std::string c1_param = dir + "/c1.0033";
	std::string s1_param = dir + "/s1.0033";
	std::string f_param = dir + "/f.0033";
	load_params(&l_c1, const_cast<char*>(c1_param.c_str()));
	load_params(&l_s1, const_cast<char*>(s1_param.c_str()));
	load_params(&l_f, const_cast<char*>(f_param.c_str()));
	// fprintf(stdout, "load params from %s, %s, %s\n", c1_param.c_str(), s1_param.c_str(), f_param.c_str());
	auto start = std::chrono::high_resolution_clock::now();
	double acc = test();
	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
	fprintf(stdout, "%.4f:%.4f\n", diff.count(), acc);
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

	clock_t start, end;
	start = clock();

	l_input.setOutput((float *)input);
	
	fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
	fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
	apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

	fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
	fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
	apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();

	bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
	bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

	bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
	bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
	bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

	bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
	bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
	bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);


	apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{

	float err;
	int iter = 1;
	
	double time_taken = 0.0;

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
            float *tmp_err;
            float tmp_err_host;
            cudaMalloc(&tmp_err, sizeof(float));
            cudaMemset(tmp_err, 0, sizeof(float));

			time_taken += forward_pass(train_set[i].data);
			
			
			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
            custom_snrm2<<<1, 10>>>(l_f.d_preact, 10, tmp_err); 
            cudaMemcpy(&tmp_err_host, tmp_err, sizeof(float), cudaMemcpyDeviceToHost);
			err += tmp_err_host;

			time_taken += back_pass();

		}
		


	}

}

// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];
	
	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static double test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	return 1.0-(double(error) / double(test_cnt));
}

void save_params(Layer* layer, const char* filename){
	float bias_save[layer->N];
	float weight_save[layer->N][layer->M];
	cudaMemcpy(bias_save, layer->bias, layer->N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy2D(weight_save, layer->M * sizeof(float), layer->weight, layer->M * sizeof(float), layer->M * sizeof(float), layer->N, cudaMemcpyDeviceToHost);
	// fprintf(stdout, "file name: %s\n", filename);
	std::ofstream file(filename, std::ios::app);
		if (!file.is_open()) {
			fflush(stderr);
			fprintf(stderr, "无法打开文件\n");
			return;
		}

	for (auto bia : bias_save){
		file << bia << " ";
	}
	
	for (int i=0; i<layer->N; ++i) {
		for (int j=0; j<layer->M; ++j) {
			file << weight_save[i][j] << " ";
		}
	}
		
	file.close();
	return;
}

static void load_params(Layer* layer, char* filename){
	float bias_save[layer->N];
	float weight_save[layer->N][layer->M];

	fprintf(stdout, "file name: %s\n", filename);
	// open file
	std::ifstream file(filename);
		if (!file.is_open()) {
			fflush(stderr);
			fprintf(stderr, "无法打开文件\n");
			return;
		}

	// read bias
	for (int i=0; i < layer->N; ++i) {
		file >> bias_save[i];
	}

	// read weight
	for (int i=0; i < layer->N; ++i){
		for (int j=0; j<layer->M; ++j){
			file >> weight_save[i][j];
		}
	}

	cudaMemcpy(layer->bias, bias_save, layer->N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy2D(layer->weight, layer->M * sizeof(float), weight_save, layer->M * sizeof(float), layer->M * sizeof(float), layer->N, cudaMemcpyHostToDevice);

	
}