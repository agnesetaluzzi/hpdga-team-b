#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include <vector>
#define BLOCK_DIM 256

/* error handling for CUDA API functions */
#define CHECK(call)                                                  \
    {                                                                \
        const cudaError_t err = call;                                \
        if (err != cudaSuccess)                                      \
        {                                                            \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                              \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

/* check to kernel call */
#define CHECK_KERNELCALL()                                           \
    {                                                                \
        const cudaError_t err = cudaGetLastError();                  \
        if (err != cudaSuccess)                                      \
        {                                                            \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                              \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

float *input_data, *input_grad, *layer1_var1_data, *layer1_var1_grad, *layer1_var2_data, *layer1_var2_grad, *layer2_var1_data, *layer2_var1_grad, *output_data, *output_grad;
float *b_sum;

// ################################################################################################################

/**
 * Dense matrix multiplication layer.
 */

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) : a(a), b(b), c(c), m(m), n(n), p(p)
{
    CHECK(cudaMalloc(&b_data, b->data.size() * sizeof(float)));
    CHECK(cudaMalloc(&layer2_var1_data, c->data.size() * sizeof(float)));

    CHECK(cudaMalloc(&b_grad, b->grad.size() * sizeof(float)));
    CHECK(cudaMalloc(&layer2_var1_grad, c->grad.size() * sizeof(float)));
	
    CHECK(cudaMalloc(&b_sum, a->data.size() * b->data.size() * sizeof(float)));
}

Matmul::~Matmul()
{
    CHECK(cudaFree(b_data));
    CHECK(cudaFree(b_grad));
}

__global__ void gpu_matmul_forward(float *a_data, float *b_data, float *c_data, const int m, const int n, const int p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= m * p) return;
    int i = idx / p;
    int k = idx % p;

    c_data[i * p + k] = 0;

    for (int j = 0; j < n; j++)
        c_data[i * p + k] += a_data[i * n + j] * b_data[j * p + k];
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);

    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid((m * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, b_data, layer2_var1_data, m, n, p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_FW);
}

__global__ void gpu_matmul_backward1(float *a_grad, float *b_data, float *c_grad, const int m, const int n, const int p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= m * n) return;
    int i = idx / n;
    int j = idx % n;

    a_grad[i * n + j] = 0;

    for (int k = 0; k < p; k++)
    {
        a_grad[i * n + j] += c_grad[i * p + k] * b_data[j * p + k];
    }
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

__global__ void gpu_matmul_backward2_sum(float *values, const int dim, const int dim2, const int m, const int n, const int p){
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

void Matmul::backward()
{
    timer_start(TMR_MATMUL_BW);
    
    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid1((m * n + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock1(BLOCK_DIM, 1, 1);
    gpu_matmul_backward1<<<blocksPerGrid1, threadsPerBlock1>>>(layer1_var2_grad, b_data, layer2_var1_grad, m, n, p);
    CHECK_KERNELCALL();
	
    int multiple32 = m + 32 - 1;
    multiple32 -= (multiple32 % 32);

    dim3 blocksPerGrid0((n * p + BLOCK_DIM - 1) / BLOCK_DIM, multiple32, 1);
    dim3 threadsPerBlock0(BLOCK_DIM, 1, 1);
    gpu_matmul_backward2_copy<<<blocksPerGrid0, threadsPerBlock0>>>(layer1_var2_grad, layer1_var2_data, layer2_var1_grad, m, n, p, b_sum);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    dim3 blocksPerGridSum((n * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlockSum(BLOCK_DIM, 1, 1);

    int dim = m;
    int dim2 = m;

    for (int x = 0; x < ceil(log2(m)); x++)
    {
        dim2 = ceil(dim2/2.0);
        multiple32 = dim2 + 32 - 1;
        multiple32 -= (dim2 % 32);
        blocksPerGridSum.y = multiple32;
        gpu_matmul_backward2_sum<<<blocksPerGridSum, threadsPerBlockSum>>>(b_sum, dim, dim2, m, n, p);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
        dim = dim2;
    }

    dim3 blocksPerGrid2((n * p + BLOCK_DIM - 1), 1, 1);
    dim3 threadsPerBlock2(BLOCK_DIM, 1, 1);
    gpu_matmul_backward2<<<blocksPerGrid2, threadsPerBlock2>>>(b_grad, n, p, b_sum);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&a->grad[0], layer1_var2_grad, sizeof(float) * a->grad.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&b->grad[0], b_grad, sizeof(float) * b->grad.size(), cudaMemcpyDeviceToHost));

    timer_stop(TMR_MATMUL_BW);
}

// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
 */
SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) : a(a), b(b), c(c), sp(sp), m(m), n(n), p(p)
{
    CHECK(cudaMalloc(&b_data, b->data.size() * sizeof(float)));
    CHECK(cudaMalloc(&layer1_var1_data, c->data.size() * sizeof(float)));

    CHECK(cudaMalloc(&b_grad, b->grad.size() * sizeof(float)));
    CHECK(cudaMalloc(&layer1_var1_grad, c->grad.size() * sizeof(float)));

    CHECK(cudaMalloc(&sp_indptr, sp->indptr.size() * sizeof(float)));
    CHECK(cudaMalloc(&sp_indices, sp->indices.size() * sizeof(float)));

    CHECK(cudaMemcpy(sp_indptr, &(sp->indptr[0]), sizeof(int) * sp->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_indices, &(sp->indices[0]), sizeof(int) * sp->indices.size(), cudaMemcpyHostToDevice));
}

SparseMatmul::~SparseMatmul()
{
    CHECK(cudaFree(b_data));
    CHECK(cudaFree(b_grad));
    CHECK(cudaFree(sp_indptr));
    CHECK(cudaFree(sp_indices));
}

__global__ void gpu_sparse_matmul_forward(float *a_data, float *b_data, float *c_data, int *sp_indptr, int *sp_indices, const int p, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;
    int i = idx / p;
    int k = idx % p;

    c_data[i * p + k] = 0;

    for (int jj = sp_indptr[i]; jj < sp_indptr[i + 1]; jj++)
    {
        int j = sp_indices[jj];
        c_data[i * p + k] += a_data[jj] * b_data[j * p + k];
    }
}

void SparseMatmul::forward(bool training)
{
    timer_start(TMR_SPMATMUL_FW);

    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(((sp->indptr.size() - 1) * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_sparse_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(input_data, b_data, layer1_var1_data, sp_indptr, sp_indices, p, (sp->indptr.size() - 1) * p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_FW);
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

void SparseMatmul::backward()
{
    timer_start(TMR_SPMATMUL_BW);

    CHECK(cudaMemset(b_grad, 0, sizeof(float) * b->grad.size()));

    dim3 blocksPerGrid(((sp->indptr.size() - 1) * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_sparse_matmul_backward<<<blocksPerGrid, threadsPerBlock>>>(input_data, b_grad, layer1_var1_grad, sp_indptr, sp_indices, p, sp->indptr.size() * p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&b->grad[0], b_grad, sizeof(float) * b->grad.size(), cudaMemcpyDeviceToHost));

    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

int max_diff;

/**
 * A specialized sparse matrix multiplication for graphs.
 */
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim, bool isFirst) : in(in), out(out), graph(graph), dim(dim), isFirst(isFirst)
{
    if (isFirst)
    {
        CHECK(cudaMalloc(&layer1_var2_data, out->data.size() * sizeof(float)));
        CHECK(cudaMalloc(&layer1_var2_grad, out->grad.size() * sizeof(float)));
    }
    else
    {
        CHECK(cudaMalloc(&output_data, out->data.size() * sizeof(float)));
        CHECK(cudaMalloc(&output_grad, out->grad.size() * sizeof(float)));
    }
    CHECK(cudaMalloc(&graph_indptr, graph->indptr.size() * sizeof(int)));
    CHECK(cudaMalloc(&graph_indices, graph->indices.size() * sizeof(int)));

    CHECK(cudaMemcpy(graph_indptr, &(graph->indptr[0]), sizeof(int) * graph->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(graph_indices, &(graph->indices[0]), sizeof(int) * graph->indices.size(), cudaMemcpyHostToDevice));
	
    max_diff = 0;
    for(int i = 1; i < graph->indptr.size(); i++){
    	max_diff = max(max_diff, graph->indptr[i] - graph->indptr[i - 1]);
    }
}

GraphSum::~GraphSum()
{
    CHECK(cudaFree(graph_indptr));
    CHECK(cudaFree(graph_indices));
}

__global__ void gpu_graph_sum_forward_zero(float *in_data, float *out_data, int *graph_indptr, int *graph_indices, const int dim, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;
    int src = idx / dim;
    int j = idx % dim;

    out_data[src * dim + j] = 0;
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

void GraphSum::forward(bool training)
{
    timer_start(TMR_GRAPHSUM_FW);

    dim3 blocksPerGrid0(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock0(BLOCK_DIM, 1, 1);
    if (isFirst)
        gpu_graph_sum_forward_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer1_var1_data, layer1_var2_data, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
    if (!isFirst)
        gpu_graph_sum_forward_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer2_var1_data, output_data, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    dim3 blocksPerGrid(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, sqrt(max_diff), 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    if (isFirst)
        gpu_graph_sum_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var1_data, layer1_var2_data, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
    else
        gpu_graph_sum_forward<<<blocksPerGrid, threadsPerBlock>>>(layer2_var1_data, output_data, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    if (isFirst)
    {
        CHECK(cudaMemcpy(&out->data[0], layer1_var2_data, sizeof(float) * out->data.size(), cudaMemcpyDeviceToHost));
    }
    else
    {
        CHECK(cudaMemcpy(&out->data[0], output_data, sizeof(float) * out->data.size(), cudaMemcpyDeviceToHost));
        timer_stop(TMR_GRAPHSUM_FW);
    }
}

__global__ void gpu_graph_sum_backward_zero(float *in_grad, float *out_grad, int *graph_indptr, int *graph_indices, const int dim, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;
    int src = idx / dim;
    int j = idx % dim;

    in_grad[src * dim + j] = 0;
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

void GraphSum::backward()
{
    timer_start(TMR_GRAPHSUM_BW);

    dim3 blocksPerGrid0(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock0(BLOCK_DIM, 1, 1);
    if (isFirst)
        gpu_graph_sum_backward_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer1_var1_grad, layer1_var2_grad, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
    if (!isFirst)
        gpu_graph_sum_backward_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer2_var1_grad, output_grad, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);

    dim3 blocksPerGrid(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, sqrt(max_diff), 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    if (isFirst)
        gpu_graph_sum_backward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var1_grad, layer1_var2_grad, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
    else
        gpu_graph_sum_backward<<<blocksPerGrid, threadsPerBlock>>>(layer2_var1_grad, output_grad, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    
    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) 
{
    CHECK(cudaMalloc(&count_gpu, sizeof(int)));
    CHECK(cudaMalloc(&total_loss_gpu, sizeof(float)));

    CHECK(cudaMalloc(&truth_gpu, sizeof(int) * (logits->data.size() / num_classes)));
}

CrossEntropyLoss::~CrossEntropyLoss()
{
    CHECK(cudaFree(input_data));
    CHECK(cudaFree(input_grad));
    CHECK(cudaFree(layer1_var1_data));
    CHECK(cudaFree(layer1_var1_grad));
    CHECK(cudaFree(layer1_var2_data));
    CHECK(cudaFree(layer1_var2_grad));
    CHECK(cudaFree(layer2_var1_data));
    CHECK(cudaFree(layer2_var1_grad));
    CHECK(cudaFree(output_data));
    CHECK(cudaFree(output_grad));
    CHECK(cudaFree(b_sum));
}

__global__ void gpu_cross_entropy_loss_forward1(int *truth, int *count, float *logits_data, float *total_loss, float *logits_grad, const bool training, const int idx_max, const int num_classes){
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

void CrossEntropyLoss::forward(bool training) {
    
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    CHECK(cudaMemset(count_gpu, 0, sizeof(int)));
    CHECK(cudaMemset(total_loss_gpu, 0.0, sizeof(float)));
    
    CHECK(cudaMemcpy(truth_gpu, truth, sizeof(int) * (logits->data.size() / num_classes), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(((logits->data.size() / num_classes) + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_cross_entropy_loss_forward1<<<blocksPerGrid, threadsPerBlock>>>(truth_gpu, count_gpu, output_data, total_loss_gpu, output_grad, training, (logits->data.size() / num_classes), num_classes);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&total_loss, total_loss_gpu, sizeof(float), cudaMemcpyDeviceToHost));
	  CHECK(cudaMemcpy(&count, count_gpu, sizeof(int), cudaMemcpyDeviceToHost));

    *loss = total_loss / count;
    if (training)
    {
        blocksPerGrid.x = (logits->grad.size() + BLOCK_DIM - 1) / BLOCK_DIM;
        gpu_cross_entropy_loss_forward2<<<blocksPerGrid, threadsPerBlock>>>(output_grad, count, logits->grad.size());
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
    }

    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {
}

// ################################################################################################################

/**
 * Rectified Linear Unit activation function.
 * If input is negative it will output 0.
 */
ReLU::ReLU(Variable *in)
{
    this->in = in;
    mask = new bool[in->data.size()];
	
    CHECK(cudaMalloc(&mask_gpu, in->data.size() * sizeof(bool)));
}

ReLU::~ReLU()
{
    delete[] mask;
    CHECK(cudaFree(mask_gpu));
}

__global__ void gpu_relu_forward(float *in_data, bool *mask, const bool training, const int idx_max){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;
	
    bool keep = in_data[i] > 0;
    if (training)
        mask[i] = keep;
    if (!keep)
        in_data[i] = 0;	
}

void ReLU::forward(bool training)
{
    timer_start(TMR_RELU_FW);
	
    dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_relu_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, mask_gpu, training, in->data.size());
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
	
    CHECK(cudaMemcpy(&in->data[0], layer1_var2_data, sizeof(float) * in->data.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mask, mask_gpu, in->data.size() * sizeof(bool), cudaMemcpyDeviceToHost));
	
    timer_stop(TMR_RELU_FW);
}

__global__ void gpu_relu_backward(float *in_grad, bool *mask, const int idx_max){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= idx_max) return;
	
    if (!mask[i])
        in_grad[i] = 0;
}

void ReLU::backward()
{
    timer_start(TMR_RELU_BW);
	
    CHECK(cudaMemcpy(mask_gpu, mask, in->data.size() * sizeof(bool), cudaMemcpyHostToDevice));
	
    dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_relu_backward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_grad, mask_gpu, in->data.size());
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at each step during training time to prevent overfitting.
 * Inputs that are not set to 0 are scaled up by 1/(1-P).
 */
Dropout::Dropout(Variable *in, float p, bool isFirst) : isFirst(isFirst)
{
    this->in = in;
    this->p = p;
    if (!in->grad.empty())
    {
        mask = new int[in->data.size()];
        CHECK(cudaMalloc(&mask_gpu, in->data.size() * sizeof(int)));
    }
    else
    {
        mask = nullptr;
    }

    if (isFirst)
    {
        CHECK(cudaMalloc(&input_data, in->data.size() * sizeof(float)));
        CHECK(cudaMalloc(&input_grad, in->grad.size() * sizeof(float)));
    }
}

Dropout::~Dropout()
{
    if (mask)
    {
        delete[] mask;
        CHECK(cudaFree(mask_gpu));
    }
}

void Dropout::forward(bool training)
{
    if (!training)
    {
        if (isFirst)
            CHECK(cudaMemcpy(input_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
        return;
    }
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++)
    {
        bool keep = (int)RAND() >= threshold;
        in->data[i] *= keep ? scale : 0;
        if (mask)
            mask[i] = keep;
    }
    timer_stop(TMR_DROPOUT_FW);

    if (isFirst)
    {
        CHECK(cudaMemcpy(input_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
    }
    else
    {
        CHECK(cudaMemcpy(layer1_var2_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
    }
    if (mask)
        CHECK(cudaMemcpy(mask_gpu, mask, sizeof(int) * in->data.size(), cudaMemcpyHostToDevice));
}

__global__ void gpu_dropout_backward(float *in_grad, int *mask, const int scale, const int idx_max)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= idx_max) return;

    in_grad[idx] *= mask[idx] ? scale : 0;
}

void Dropout::backward()
{
    if (!mask)
        return;
    
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);

    dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    if (isFirst)
    {
        gpu_dropout_backward<<<blocksPerGrid, threadsPerBlock>>>(input_grad, mask_gpu, scale, in->data.size());
    }
    else
    {
        gpu_dropout_backward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_grad, mask_gpu, scale, in->data.size());
    }
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################
