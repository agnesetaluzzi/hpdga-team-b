#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include <vector>

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

/* Check to kernel call */
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

float *input_data, *layer1_var1_data, *layer1_var1_grad, *layer1_var2_data, *layer1_var2_grad, *layer2_var1, *output;
// ################################################################################################################

/**
 * Dense matrix multiplication layer.
 */
Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) : a(a), b(b), c(c), m(m), n(n), p(p)
{
    CHECK(cudaMalloc(&b_data, b->data.size() * sizeof(float)));
    CHECK(cudaMalloc(&c_data, c->data.size() * sizeof(float)));

    CHECK(cudaMalloc(&a_grad, a->grad.size() * sizeof(float)));
    CHECK(cudaMalloc(&b_grad, b->grad.size() * sizeof(float)));
    CHECK(cudaMalloc(&c_grad, c->grad.size() * sizeof(float)));
}

Matmul::~Matmul()
{
    CHECK(cudaFree(b_data));
    CHECK(cudaFree(c_data));
    CHECK(cudaFree(a_grad));
    CHECK(cudaFree(b_grad));
    CHECK(cudaFree(c_grad));
}

__global__ void gpu_matmul_forward(float *a_gpu, float *b_gpu, float *c_gpu, const int m, const int n, const int p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    c_gpu[i * p + k] = 0;

    for (int j = 0; j < n; j++)
        c_gpu[i * p + k] += a_gpu[i * n + j] * b_gpu[j * p + k];
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);

    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(m, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, b_data, c_data, m, n, p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&c->data[0], c_data, sizeof(float) * c->data.size(), cudaMemcpyDeviceToHost));
    timer_stop(TMR_MATMUL_FW);
}

__global__ void gpu_matmul_backward1(float *a_grad, float *a_data, float *b_data, float *b_grad, float *c_grad, const int m, const int n, const int p)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    a_grad[i * n + j] = 0;

    for (int k = 0; k < p; k++)
    {
        a_grad[i * n + j] += c_grad[i * p + k] * b_data[j * p + k];
    }
}

__global__ void gpu_matmul_backward2(float *a_grad, float *a_data, float *b_data, float *b_grad, float *c_grad, const int m, const int n, const int p)
{
    int j = blockIdx.x;
    int k = threadIdx.x;

    b_grad[j * p + k] = 0;

    for (int i = 0; i < m; i++)
    {
        b_grad[j * p + k] += c_grad[i * p + k] * a_data[i * n + j];
    }
}

void Matmul::backward()
{
    timer_start(TMR_MATMUL_BW);
    CHECK(cudaMemcpy(a_grad, &(a->grad[0]), sizeof(float) * a->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_grad, &(b->grad[0]), sizeof(float) * b->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c_grad, &(c->grad[0]), sizeof(float) * c->grad.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid1(m, 1, 1);
    dim3 threadsPerBlock1(n, 1, 1);
    gpu_matmul_backward1<<<blocksPerGrid1, threadsPerBlock1>>>(a_grad, layer1_var2_data, b_data, b_grad, c_grad, m, n, p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    dim3 blocksPerGrid2(n, 1, 1);
    dim3 threadsPerBlock2(p, 1, 1);
    gpu_matmul_backward2<<<blocksPerGrid2, threadsPerBlock2>>>(a_grad, layer1_var2_data, b_data, b_grad, c_grad, m, n, p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&a->grad[0], a_grad, sizeof(float) * a->grad.size(), cudaMemcpyDeviceToHost));
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
}

SparseMatmul::~SparseMatmul()
{
    CHECK(cudaFree(b_data));
    CHECK(cudaFree(b_grad));
    // CHECK(cudaFree(c_grad));
    CHECK(cudaFree(sp_indptr));
    CHECK(cudaFree(sp_indices));
}

__global__ void gpu_sparse_matmul_forward(float *a_data, float *b_data, float *c_data, int *sp_indptr, int *sp_indices, const int p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    for (int jj = sp_indptr[i]; jj < sp_indptr[i + 1]; jj++)
    {
        int j = sp_indices[jj];
        c_data[i * p + k] += a_data[jj] * b_data[j * p + k];
    }
}

void SparseMatmul::forward(bool training)
{
    timer_start(TMR_SPMATMUL_FW);

    CHECK(cudaMemcpy(sp_indptr, &(sp->indptr[0]), sizeof(int) * sp->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_indices, &(sp->indices[0]), sizeof(int) * sp->indices.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(layer1_var1_data, 0, sizeof(float) * c->data.size()));

    dim3 blocksPerGrid(sp->indptr.size() - 1, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_sparse_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(input_data, b_data, layer1_var1_data, sp_indptr, sp_indices, p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&c->data[0], layer1_var1_data, sizeof(float) * c->data.size(), cudaMemcpyDeviceToHost));

    timer_stop(TMR_SPMATMUL_FW);
}

__global__ void gpu_sparse_matmul_backward(float *a_data, float *b_grad, float *c_grad, int *sp_indptr, int *sp_indices, const int p, const int sp_indptr_size)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    for (int jj = sp_indptr[i]; jj < sp_indptr[i + 1]; jj++)
    {
        int j = sp_indices[jj];
        atomicAdd(&b_grad[j * p + k], c_grad[i * p + k] * a_data[jj]);
    }
}

void SparseMatmul::backward()
{
    timer_start(TMR_SPMATMUL_BW);

    CHECK(cudaMemcpy(sp_indptr, &(sp->indptr[0]), sizeof(int) * sp->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_indices, &(sp->indices[0]), sizeof(int) * sp->indices.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(b_grad, 0, sizeof(float) * b->grad.size()));
    CHECK(cudaMemcpy(layer1_var1_grad, &(c->grad[0]), sizeof(float) * c->grad.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(sp->indptr.size() - 1, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_sparse_matmul_backward<<<blocksPerGrid, threadsPerBlock>>>(input_data, b_grad, layer1_var1_grad, sp_indptr, sp_indices, p, sp->indptr.size());
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&b->grad[0], b_grad, sizeof(float) * b->grad.size(), cudaMemcpyDeviceToHost));

    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
 */
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim, bool isFirst) : in(in), out(out), graph(graph), dim(dim), isFirst(isFirst)
{
    if (!isFirst)
        CHECK(cudaMalloc(&in_data, in->data.size() * sizeof(float)));
    if (!isFirst)
        CHECK(cudaMalloc(&out_data, out->data.size() * sizeof(float)));
    if (isFirst)
        CHECK(cudaMalloc(&layer1_var2_data, out->data.size() * sizeof(float)))
    if (!isFirst)
        CHECK(cudaMalloc(&in_grad, in->grad.size() * sizeof(float)));
    CHECK(cudaMalloc(&out_grad, out->grad.size() * sizeof(float)));
    CHECK(cudaMalloc(&graph_indptr, graph->indptr.size() * sizeof(int)));
    CHECK(cudaMalloc(&graph_indices, graph->indices.size() * sizeof(int)));
}

GraphSum::~GraphSum()
{
    if (!isFirst)
        CHECK(cudaFree(in_data));
    CHECK(cudaFree(out_grad));
    if (!isFirst)
        CHECK(cudaFree(in_grad));
    if (!isFirst)
        CHECK(cudaFree(out_data));
    CHECK(cudaFree(graph_indptr));
    CHECK(cudaFree(graph_indices));
}

__global__ void gpu_graph_sum_forward(float *in_data, float *out_data, int *graph_indptr, int *graph_indices, const int dim)
{
    int src = blockIdx.x;
    int j = threadIdx.x;

    out_data[src * dim + j] = 0;

    for (int i = graph_indptr[src]; i < graph_indptr[src + 1]; i++)
    {
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf((graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
        out_data[src * dim + j] += coef * in_data[dst * dim + j];
    }
}

void GraphSum::forward(bool training)
{
    timer_start(TMR_GRAPHSUM_FW);

    CHECK(cudaMemcpy(graph_indptr, &(graph->indptr[0]), sizeof(int) * graph->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(graph_indices, &(graph->indices[0]), sizeof(int) * graph->indices.size(), cudaMemcpyHostToDevice));
    if (!isFirst)
        CHECK(cudaMemcpy(in_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(graph->indptr.size() - 1, 1, 1);
    dim3 threadsPerBlock(dim, 1, 1);
    if (!isFirst)
        gpu_graph_sum_forward<<<blocksPerGrid, threadsPerBlock>>>(in_data, out_data, graph_indptr, graph_indices, dim);
    if (isFirst)
        gpu_graph_sum_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var1_data, layer1_var2_data, graph_indptr, graph_indices, dim);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    if (!isFirst)
        CHECK(cudaMemcpy(&out->data[0], out_data, sizeof(float) * out->data.size(), cudaMemcpyDeviceToHost));
    if (isFirst)
        CHECK(cudaMemcpy(&out->data[0], layer1_var2_data, sizeof(float) * out->data.size(), cudaMemcpyDeviceToHost));
    timer_stop(TMR_GRAPHSUM_FW);
}

__global__ void gpu_graph_sum_backward(float *in_grad, float *out_grad, int *graph_indptr, int *graph_indices, const int dim)
{
    int src = blockIdx.x;
    int j = threadIdx.x;

    in_grad[src * dim + j] = 0;
    
    for (int i = graph_indptr[src]; i < graph_indptr[src + 1]; i++)
    {
        int dst = graph_indices[i];
        float coef = 1.0 / sqrtf(
                               (graph_indptr[src + 1] - graph_indptr[src]) * (graph_indptr[dst + 1] - graph_indptr[dst]));
        in_grad[src * dim + j] += coef * out_grad[dst * dim + j];
    }
}

void GraphSum::backward()
{
    timer_start(TMR_GRAPHSUM_BW);

    CHECK(cudaMemcpy(graph_indptr, &(graph->indptr[0]), sizeof(int) * graph->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(graph_indices, &(graph->indices[0]), sizeof(int) * graph->indices.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(out_grad, &(out->grad[0]), sizeof(float) * out->grad.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(graph->indptr.size() - 1, 1, 1);
    dim3 threadsPerBlock(dim, 1, 1);
    if (!isFirst)
        gpu_graph_sum_backward<<<blocksPerGrid, threadsPerBlock>>>(in_grad, out_grad, graph_indptr, graph_indices, dim);
    if (isFirst)
        gpu_graph_sum_backward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var1_grad, out_grad, graph_indptr, graph_indices, dim);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    if (!isFirst)
        CHECK(cudaMemcpy(&in->grad[0], in_grad, sizeof(float) * in->grad.size(), cudaMemcpyDeviceToHost));
    if (isFirst)
        CHECK(cudaMemcpy(&in->grad[0], layer1_var1_grad, sizeof(float) * in->grad.size(), cudaMemcpyDeviceToHost));
    
    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

CrossEntropyLoss::~CrossEntropyLoss()
{
}

void CrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    if (training) logits->zero_grad();
    for (int i = 0; i < logits->data.size() / num_classes; i++) {
        if (truth[i] < 0) continue;
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;
        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training) {
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss = total_loss / count;
    if (training)
        for (float & i : logits->grad)
            i /= count;
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
	
	CHECK(cudaMalloc(&in_grad, in->grad.size() * sizeof(float)));
	CHECK(cudaMalloc(&mask_gpu, in->data.size() * sizeof(bool)));
}

ReLU::~ReLU()
{
    delete[] mask;
    CHECK(cudaFree(in_grad));
    CHECK(cudaFree(mask_gpu));
}

__global__ void gpu_relu_forward(float *in_data, bool *mask, const bool training){
	int i = blockIdx.x;
	
    bool keep = in_data[i] > 0;
    if (training)
        mask[i] = keep;
    if (!keep)
        in_data[i] = 0;	
}

void ReLU::forward(bool training)
{
    timer_start(TMR_RELU_FW);
	
	dim3 blocksPerGrid(in->data.size(), 1, 1);
    dim3 threadsPerBlock(1, 1, 1);
    gpu_relu_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, mask_gpu, training);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
	
	CHECK(cudaMemcpy(&in->data[0], layer1_var2_data, sizeof(float) * in->data.size(), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(mask, mask_gpu, in->data.size() * sizeof(bool), cudaMemcpyDeviceToHost));
	
    timer_stop(TMR_RELU_FW);
}

__global__ void gpu_relu_backward(float *in_grad, bool *mask){
	int i = blockIdx.x;
	
    if (!mask[i])
        in_grad[i] = 0;
}

void ReLU::backward()
{
    timer_start(TMR_RELU_BW);
	
	CHECK(cudaMemcpy(in_grad, &(in->grad[0]), sizeof(float) * in->grad.size(), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mask_gpu, mask, in->data.size() * sizeof(bool), cudaMemcpyHostToDevice));
	
	dim3 blocksPerGrid(in->data.size(), 1, 1);
    dim3 threadsPerBlock(1, 1, 1);
    gpu_relu_backward<<<blocksPerGrid, threadsPerBlock>>>(in_grad, mask_gpu);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
	
	CHECK(cudaMemcpy(&in->grad[0], in_grad, sizeof(float) * in->grad.size(), cudaMemcpyDeviceToHost));

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
        mask = new int[in->data.size()];
    else
        mask = nullptr;

    if (isFirst)
        CHECK(cudaMalloc(&input_data, in->data.size() * sizeof(float)));
}

Dropout::~Dropout()
{
    if (mask)
        delete[] mask;
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
        CHECK(cudaMemcpy(input_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
    if (!isFirst)
        CHECK(cudaMemcpy(layer1_var2_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
}

void Dropout::backward()
{
    if (!mask)
        return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################