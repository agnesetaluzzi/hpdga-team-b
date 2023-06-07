#include "../include/kernels.cuh"

__global__ void gpu_zero(float *x, const int n_cols, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;
    int i = idx / n_cols;
    int j = idx % n_cols;

    x[i * n_cols + j] = 0;
}

__global__ void gpu_matmul_forward(float *a_data, float *b_data, float *c_data, const int m, const int n, const int p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    if(idx >= m * p) return;
    int i = idx / p;
    int k = idx % p;

    __shared__ float local_vars[BLOCK_DIM];

    local_vars[thread_id] = 0;

    for (int j = 0; j < n; j++)
        local_vars[thread_id] += a_data[i * n + j] * b_data[j * p + k];

    c_data[i * p + k] = local_vars[thread_id];
}

__global__ void gpu_matmul_forward2(float *a, float *b, float *c, int a_rows, int a_columns, int b_rows, int b_columns, int c_rows, int c_columns)
{
    __shared__ float shared_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float c_val = 0.0;
    shared_a[threadIdx.y][threadIdx.x] = 0.0;
    shared_b[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((a_columns - 1) / TILE_WIDTH) + 1); ph++)
    {
        if (row < a_rows && (threadIdx.x + (ph * TILE_WIDTH)) < a_columns)
        {
            shared_a[threadIdx.y][threadIdx.x] = a[(row * a_columns) + threadIdx.x + (ph * TILE_WIDTH)];
        }
        else
        {
            shared_a[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (col < b_columns && (threadIdx.y + ph * TILE_WIDTH) < b_rows)
        {
            shared_b[threadIdx.y][threadIdx.x] = b[(threadIdx.y + ph * TILE_WIDTH) * b_columns + col];
        }
        else
        {
            shared_b[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            c_val += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
        }
    }
    if (row < c_rows && col < c_columns)
    {
        c[row * c_columns + col] = c_val;
    }
}

__global__ void gpu_matmul_backward1(float *a_grad, float *b_data, float *c_grad, const int m, const int n, const int p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    if(idx >= m * n) return;
    int i = idx / n;
    int j = idx % n;

    __shared__ float local_vars[BLOCK_DIM];

    local_vars[thread_id] = 0;

    for (int k = 0; k < p; k++)
    {
         local_vars[thread_id] += c_grad[i * p + k] * b_data[j * p + k];
    }

    a_grad[i * n + j] = local_vars[thread_id];
}

__global__ void gpu_matmul_backward2_copy(float *a_grad, float *a_data, float *c_grad, const int m, const int n, const int p, float *values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = blockIdx.y;
    if(idx >= n * p || i >= m) return;
    int j = idx / p;
    int k = idx % p;
	
    values[i * n * n + j * p + k] = c_grad[i * p + k] * a_data[i * n + j];
}

__global__ void gpu_matmul_backward2(float *b_grad, const int n, const int p, float *values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n * p) return;
    int j = idx / p;
    int k = idx % p;

    b_grad[j * p + k] = values[j * p + k];
}

__global__ void gpu_matmul_backward2_sum(float *values, const int dim, const int dim2, const int m, const int n, const int p)
{
    int pos = blockIdx.y;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n * p || pos >= dim2) return;
    int j = idx / (p);
    int k = idx % (p);
    if(dim % 2 == 0 || pos != int(dim / 2))
    {
     	values[pos * n * n + j * p + k] += values[(pos + dim2) * n * n + j * p + k];
    }
}

__global__ void gpu_matmul_backward3(float *b_grad, float *a_data, float *c_grad, const int m, const int n, const int p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    if(idx >= n * p) return;
    int j = idx / p;
    int k = idx % p;

    __shared__ float local_vars[BLOCK_DIM];

    local_vars[thread_id] = 0;

    for (int i = 0; i < m; i++)
    {
         local_vars[thread_id] += c_grad[i * p + k] * a_data[i * n + j];
    }

    b_grad[j * p + k] = local_vars[thread_id];
}

__global__ void gpu_sparse_matmul_forward(int *i_index, float *a_data, float *b_data, float *c_data, int *sp_indptr, int *sp_indices, const int p, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    if(idx >= idx_max) return;
    int ind_i = idx / p;
	int i = i_index[ind_i];
    int k = idx % p;

    __shared__ float local_vars[BLOCK_DIM];

    local_vars[thread_id] = 0;

    for (int jj = sp_indptr[i]; jj < sp_indptr[i + 1]; jj++)
    {
        int j = sp_indices[jj];
        local_vars[thread_id] += a_data[jj] * b_data[j * p + k];
    }

    c_data[i * p + k] = local_vars[thread_id];
}

__global__ void gpu_sparse_matmul_backward(float *a_data, float *b_grad, float *c_grad, int *sp_indptr, int *sp_indices, const int p, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;
    int i = idx / p;
    int k = idx % p;

    for (int jj = sp_indptr[i]; jj < sp_indptr[i + 1]; jj++)
    {
        int j = sp_indices[jj];
        atomicAdd(&b_grad[j * p + k], c_grad[i * p + k] * a_data[jj]);
    }
}

__global__ void gpu_graph_sum_forward(float *in_data, float *out_data, int *graph_indptr, int *graph_indices, const int dim, const int length, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > idx_max) return;
    int src = idx / dim;
    int j = idx % dim;
    int delta_i = blockIdx.y;

    float sum = 0;
	
    for (int i = graph_indptr[src] + delta_i; i < graph_indptr[src + 1]; i += length)
    {
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf((graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
	    sum += coef * in_data[dst * dim + j];
    }
	atomicAdd(&out_data[src * dim + j], sum);
}

__global__ void gpu_graph_sum_forward2(int *src_index, float *in_data, float *out_data, int *graph_indptr, int *graph_indices, const int dim, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > idx_max) return;
    int ind_src = idx / dim;
	int src = src_index[ind_src];
    int j = idx % dim;

    float sum = 0;
	
    for (int i = graph_indptr[src]; i < graph_indptr[src + 1]; i++)
    {
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf((graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
	    sum += coef * in_data[dst * dim + j];
    }
	out_data[src * dim + j] = sum;
}

__global__ void gpu_graph_sum_backward(float *in_grad, float *out_grad, int *graph_indptr, int *graph_indices, const int dim, const int length, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > idx_max) return;
    int src = idx / dim;
    int j = idx % dim;
    int delta_i = blockIdx.y;
	
    float sum = 0;
	
    for (int i = graph_indptr[src] + delta_i; i < graph_indptr[src + 1]; i += length)
    {
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf(
                               (graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
        sum += coef * out_grad[dst * dim + j];
    }
    atomicAdd(&in_grad[src * dim + j], sum);
}

__global__ void gpu_graph_sum_backward2(int *src_index, float *in_grad, float *out_grad, int *graph_indptr, int *graph_indices, const int dim, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > idx_max) return;
    int ind_src = idx / dim;
	int src = src_index[ind_src];
    int j = idx % dim;
	
    float sum = 0;
	
    for (int i = graph_indptr[src]; i < graph_indptr[src + 1]; i++)
    {
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf(
                               (graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
        sum += coef * out_grad[dst * dim + j];
    }
    in_grad[src * dim + j] = sum;
}

__global__ void gpu_cross_entropy_loss_forward1(int *truth, int *count, float *logits_data, float *total_loss, float *logits_grad, const bool training, const int idx_max, const int num_classes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= idx_max || truth[i] < 0) return;

    atomicAdd(count, 1);

    float *logit = &logits_data[i * num_classes];
    float max_logit = -1e30, sum_exp = 0;
    for (int j = 0; j < num_classes; j++)
        max_logit = fmax(max_logit, logit[j]);
    for (int j = 0; j < num_classes; j++)
    {
        logit[j] -= max_logit;
        sum_exp += expf(logit[j]);
    }
    atomicAdd(total_loss, logf(sum_exp) - logit[truth[i]]);

    if (training)
    {
        for (int j = 0; j < num_classes; j++)
        {
            float prob = expf(logit[j]) / sum_exp;
            logits_grad[i * num_classes + j] = prob;
        }
        logits_grad[i * num_classes + truth[i]] -= 1.0;
    }
}

__global__ void gpu_cross_entropy_loss_forward2(float *logits_grad, const int count, const int idx_max)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;

    logits_grad[i] /= count;
}

__global__ void gpu_relu_forward(float *in_data, bool *mask, const bool training, const int idx_max)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;
	
    bool keep = in_data[i] > 0;
    if (training)
        mask[i] = keep;
    if (!keep)
        in_data[i] = 0;	
}

__global__ void gpu_relu_backward(float *in_grad, bool *mask, const int idx_max)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;
	
    if (!mask[i])
        in_grad[i] = 0;
}

__global__ void gpu_set_original_input(float *in_data, float *original_input_data, const int idx_max)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;
	
    in_data[i] = original_input_data[i];
}

__global__ void gpu_dropout_forward(float *in_data, bool *mask, const bool isMask, const int threshold, const int scale, const int idx_max, unsigned long long *rand1, unsigned long long *rand2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;

    unsigned long long t = rand1[i];
    unsigned long long const s = rand2[i];
    rand1[i] = s;
    t ^= t << 23;		// a
    t ^= t >> 17;		// b
    t ^= s ^ (s >> 26);	// c
    rand2[i] = t;
    unsigned int res = (t + s) & 0x7fffffff;
    int rand = (int)res;

    in_data[i] *= (rand >= threshold) ? scale : 0;
    if (isMask)
        mask[i] = (rand >= threshold);
}

__global__ void gpu_dropout_backward(float *in_grad, bool *mask, const int scale, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;

    in_grad[idx] *= mask[idx] ? scale : 0;
}