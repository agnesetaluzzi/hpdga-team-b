#ifndef KERNELS_H
#define BLOCK_DIM 256
#define TILE_WIDTH 32

__global__ void gpu_zero(float *x, const int dim, const int idx_max);

__global__ void gpu_matmul_forward(float *a_data, float *b_data, float *c_data, const int m, const int n, const int p);
__global__ void gpu_matmul_forward2(float *a, float *b, float *c, int a_rows, int a_columns, int b_rows, int b_columns, int c_rows, int c_columns);
__global__ void gpu_matmul_backward1(float *a_grad, float *b_data, float *c_grad, const int m, const int n, const int p);
__global__ void gpu_matmul_backward2_copy(float *a_grad, float *a_data, float *c_grad, const int m, const int n, const int p, float *values);
__global__ void gpu_matmul_backward2(float *b_grad, const int n, const int p, float *values);
__global__ void gpu_matmul_backward2_sum(float *values, const int dim, const int dim2, const int m, const int n, const int p);
__global__ void gpu_matmul_backward3(float *b_grad, float *a_data, float *c_grad, const int m, const int n, const int p);

__global__ void gpu_sparse_matmul_forward(int *i_index, float *a_data, float *b_data, float *c_data, int *sp_indptr, int *sp_indices, const int p, const int idx_max);
__global__ void gpu_sparse_matmul_backward(float *a_data, float *b_grad, float *c_grad, int *sp_indptr, int *sp_indices, const int p, const int idx_max);

__global__ void gpu_graph_sum_forward(float *in_data, float *out_data, int *graph_indptr, int *graph_indices, const int dim, const int length, const int idx_max);
__global__ void gpu_graph_sum_forward2(int *src_index, float *in_data, float *out_data, int *graph_indptr, int *graph_indices, const int dim, const int idx_max);

__global__ void gpu_graph_sum_backward(float *in_grad, float *out_grad, int *graph_indptr, int *graph_indices, const int dim, const int length, const int idx_max);
__global__ void gpu_graph_sum_backward2(int *src_index, float *in_grad, float *out_grad, int *graph_indptr, int *graph_indices, const int dim, const int idx_max);

__global__ void gpu_cross_entropy_loss_forward1(int *truth, int *count, float *logits_data, float *total_loss, float *logits_grad, const bool training, const int idx_max, const int num_classes);
__global__ void gpu_cross_entropy_loss_forward2(float *logits_grad, const int count, const int idx_max);

__global__ void gpu_relu_forward(float *in_data, bool *mask, const bool training, const int idx_max);
__global__ void gpu_relu_backward(float *in_grad, bool *mask, const int idx_max);

__global__ void gpu_set_original_input(float *in_data, float *original_input_data, const int idx_max);
__global__ void gpu_dropout_forward(float *in_data, bool *mask, const bool isMask, const int threshold, const int scale, const int idx_max, unsigned long long *rand1, unsigned long long *rand2);
__global__ void gpu_dropout_backward(float *in_grad, bool *mask, const int scale, const int idx_max);

#define RAND_H
#endif