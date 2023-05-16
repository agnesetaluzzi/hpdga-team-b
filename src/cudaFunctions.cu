#include "../include/module.h"
#include "../include/cudaFunctions.h"

__global__ void gpu_matmul_forward(float *a_gpu, float *b_gpu, float *c_gpu, int *m, int *n, int *p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    c_gpu[i * (*p) + k] = 0;

    for (int j = 0; j < (*n); j++)
        c_gpu[i * (*p) + k] += a_gpu[i * (*n) + j] * b_gpu[j * (*p) + k];
}

__global__ void gpu_matmul_backward1(float *a_grad, float *a_data, float *b_data, float *b_grad, float *c_grad, int *m, int *n, int *p)
{
	int i = blockIdx.x;
    int j = threadIdx.x;
	
	a_grad[i * (*n) + j] = 0;	
	
	float tmp = 0;
    for (int k = 0; k < *p; k++){
        tmp += c_grad[i * (*p) + k] * b_data[j * (*p) + k];
    }
    a_grad[i * (*n) + j] = tmp;
}

__global__ void gpu_matmul_backward2(float *a_grad, float *a_data, float *b_data, float *b_grad, float *c_grad, int *m, int *n, int *p)
{
	int j = blockIdx.x;
    int k = threadIdx.x;
    
	b_grad[j * (*p) + k] = 0;
	
    for (int i = 0; i < (*m); i++){
		b_grad[j * (*p) + k] += c_grad[i * (*p) + k] * a_data[i * (*n) + j];
	}
}

__global__ void gpu_sparse_matmul_forward(float *a_data, float *b_data, float *c_data, int *sp_indptr, int *sp_indices, int *p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

	c_data[i * (*p) + k] = 0;

    for (int jj = sp_indptr[i]; jj < sp_indptr[i + 1]; jj++){
        int j = sp_indices[jj];
        c_data[i * (*p) + k] += a_data[jj] * b_data[j * (*p) + k];
    }
}

__global__ void gpu_sparse_matmul_backward(float *a_data, float *b_grad, float *c_grad, int *sp_indptr, int *sp_indices, int *p, int sp_indptr_size)
{
    int jj = blockIdx.x;
    int k = threadIdx.x;

	for (int i = 0; i < sp_indptr_size - 1; i++){
		int j = sp_indices[jj];
		if(j >= sp_indptr[i] && j < sp_indptr[i + 1]){
			b_grad[j * (*p) + k] += c_grad[i * (*p) + k] * a_data[jj];
		}
	}
}

// Still not tested
__global__ void gpu_graphsum_forward(float *in_data, float *out_data, float *graph_indptr, float *graph_indices, int *dim)
{
    int src = blockIdx.x;
    int j = threadIdx.x;
	
	out_data[src * (*dim) + j] = 0;
	
	for (int i = graph_indptr[src]; i < graph_indptr[src + 1]; i++){
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf((graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
		out_data[src * (*dim) + j] += coef * in_data[dst * (*dim) + j];
	}
}

//Still not tested
__global__ void gpu_graphsum_backward(float *in_grad, float *out_grad, float *graph_indptr, float *graph_indices, int *dim)
{
    int src = blockIdx.x;
    int j = threadIdx.x;
	
	out_grad[src * (*dim) + j] = 0;
	
	for (int i = graph_indptr[src]; i < graph_indptr[src + 1]; i++){
		int dst = graph_indices[i];
        float coef = 1.0 / sqrtf((graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
		in_grad[src * (*dim) + j] += coef * out_grad[dst * (*dim) + j];
	}
}