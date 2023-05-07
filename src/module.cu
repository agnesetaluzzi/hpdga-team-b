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

// ################################################################################################################
/**
 * Dense matrix multiplication layer.
 */

bool value_to_gpu = false;

float *a_data;
float *b_data;
float *c_data;
int *M, *N, *P;

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) : a(a), b(b), c(c), m(m), n(n), p(p) {
	cudaMallocManaged(&a_data, m * n * sizeof(float));
	cudaMallocManaged(&b_data, n * p * sizeof(float));
	cudaMallocManaged(&c_data, m * p * sizeof(float));
	cudaMallocManaged(&M, sizeof(int));
	cudaMallocManaged(&N, sizeof(int));
	cudaMallocManaged(&P, sizeof(int));

	cudaMemPrefetchAsync(a_data, m * n * sizeof(float), 0);
	cudaMemPrefetchAsync(b_data, n * p * sizeof(float), 0);
	cudaMemPrefetchAsync(c_data, m * p * sizeof(float), 0);
	cudaMemPrefetchAsync(M, sizeof(int), 0);
	cudaMemPrefetchAsync(N, sizeof(int), 0);
	cudaMemPrefetchAsync(P, sizeof(int), 0);

	CHECK(cudaDeviceSynchronize());

	//cudaMemPrefetchAsync(d_a, N*sizeof(float), 0);
}

__global__ void gpu_matmul_forward(float *a_data, float *b_data, float *c_data, int *data, int *N, int *P)
{
    int i = blockIdx.x;
    int k = threadIdx.x;
    c_data[i * (*P) + k] = 0;
    for (int j = 0; j < (*N); j++)
        c_data[i * (*P) + k] += a_data[i * (*N) + j] * b_data[j * (*P) + k];
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);
	CHECK(cudaDeviceSynchronize());

	for(int i = 0; i < m * n; i++){
		a_data[i] = a->data[i];
	}

	for(int i = 0; i < n * p; i++){
		b_data[i] = b->data[i];
	}

	*M = m;
	*N = n;
	*P = p;

	cudaMemPrefetchAsync(a_data, m * n * sizeof(float), 0);
	cudaMemPrefetchAsync(b_data, n * p * sizeof(float), 0);
	cudaMemPrefetchAsync(c_data, m * p * sizeof(float), 0);
	cudaMemPrefetchAsync(M, sizeof(int), 0);
	cudaMemPrefetchAsync(N, sizeof(int), 0);
	cudaMemPrefetchAsync(P, sizeof(int), 0);

	CHECK(cudaDeviceSynchronize());

    dim3 blocksPerGrid(m, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);

    gpu_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(a_data, b_data, c_data, M, N, P);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

	for(int i = 0; i < (*M) * (*P); i++){
		c->data[i] = c_data[i];
	}

    timer_stop(TMR_MATMUL_FW);
}

void Matmul::backward()
{
    timer_start(TMR_MATMUL_BW);
    a->zero_grad();
    b->zero_grad();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            float tmp = 0;
            for (int k = 0; k < p; k++)
            {
                tmp += c->grad[i * p + k] * b->data[j * p + k];
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
            }
            a->grad[i * n + j] = tmp;
        }
    timer_stop(TMR_MATMUL_BW);
}

// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
 */
SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) : a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void SparseMatmul::forward(bool training)
{
    timer_start(TMR_SPMATMUL_FW);
    c->zero();
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++)
        {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
    timer_stop(TMR_SPMATMUL_FW);
}

void SparseMatmul::backward()
{
    timer_start(TMR_SPMATMUL_BW);
    b->zero_grad();
    int row = 0;
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++)
        {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
        }
    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
 */
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) : in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training)
{
    timer_start(TMR_GRAPHSUM_FW);
    out->zero();
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++)
        {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                                   (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst]));
            for (int j = 0; j < dim; j++)
                // This only works for undirected graphs. Should be out[dst] += coef * in[src]
                out->data[src * dim + j] += coef * in->data[dst * dim + j];
        }
    timer_stop(TMR_GRAPHSUM_FW);
}

void GraphSum::backward()
{
    timer_start(TMR_GRAPHSUM_BW);
    in->zero_grad();
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++)
        {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                                   (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst]));
            for (int j = 0; j < dim; j++)
                in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
        }
    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss.
 */
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) : logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training)
{
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    if (training)
        logits->zero_grad();
    for (int i = 0; i < logits->data.size() / num_classes; i++)
    {
        if (truth[i] < 0)
            continue;
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;
        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
        for (int j = 0; j < num_classes; j++)
        {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training)
        {
            for (int j = 0; j < num_classes; j++)
            {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss = total_loss / count;
    if (training)
        for (float &i : logits->grad)
            i /= count;
    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward()
{
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
}

ReLU::~ReLU()
{
    delete[] mask;
}

void ReLU::forward(bool training)
{
    timer_start(TMR_RELU_FW);
    for (int i = 0; i < in->data.size(); i++)
    {
        bool keep = in->data[i] > 0;
        if (training)
            mask[i] = keep;
        if (!keep)
            in->data[i] = 0;
    }
    timer_stop(TMR_RELU_FW);
}

void ReLU::backward()
{
    timer_start(TMR_RELU_BW);
    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i])
            in->grad[i] = 0;
    timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at each step during training time to prevent overfitting.
 * Inputs that are not set to 0 are scaled up by 1/(1-P).
 */
Dropout::Dropout(Variable *in, float p)
{
    this->in = in;
    this->p = p;
    if (!in->grad.empty())
        mask = new int[in->data.size()];
    else
        mask = nullptr;
}

Dropout::~Dropout()
{
    if (mask)
        delete[] mask;
}

void Dropout::forward(bool training)
{
    if (!training)
        return;
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