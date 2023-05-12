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

__global__ void gpu_zero(float *data, int *p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    data[i * (*p) + k] = 0;
}

float *a_gpu, *b_gpu, *c_gpu;
int *m_gpu, *n_gpu, *p_gpu;

// ################################################################################################################
/**
 * Dense matrix multiplication layer.
 */
Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) : a(a), b(b), c(c), m(m), n(n), p(p) {}

__global__ void gpu_matmul_forward(float *a_gpu, float *b_gpu, float *c_gpu, int *m, int *n, int *p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    c_gpu[i * (*p) + k] = 0;

    for (int j = 0; j < (*n); j++)
        c_gpu[i * (*p) + k] += a_gpu[i * (*n) + j] * b_gpu[j * (*p) + k];
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);

    // GPUs do not support std::vector, etc. so I have to use arrays
    // since the spec now guarantees vectors store their elements contiguously, to transform a std::vector to an array:
    // std::vector<double> v;
    // double *a = &v[0];

    /*dim3 blocksPerGrid(m, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, m_gpu, n_gpu, p_gpu);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());*/

    c->zero();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
        }

    // convert an array to a std::vector:
    // int src[] = { 1, 2, 3, 4, 5 };
    // int n = sizeof(src) / sizeof(src[0]);
    // std::vector<int> dest(src, src + n);

    /*float *c_data_from_gpu;
    c_data_from_gpu = (float *)malloc(m * p * sizeof(float));
    CHECK(cudaMemcpy(c_data_from_gpu, c_gpu, sizeof(float) * m * p, cudaMemcpyDeviceToHost));
    std::vector<float> c_data_from_gpu_vector(c_data_from_gpu, c_data_from_gpu + m * p);
    c->data = c_data_from_gpu_vector;

    CHECK(cudaFree(a_gpu));
    CHECK(cudaFree(b_gpu));
    CHECK(cudaFree(c_gpu));
    CHECK(cudaFree(m_gpu));
    CHECK(cudaFree(n_gpu));
    CHECK(cudaFree(p_gpu));*/

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

__global__ void gpu_sparse_matmul_forward(float *a_gpu, float *b_gpu, float *c_gpu, int *sp_indptr_gpu, int *sp_indices_gpu, int *p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    for (int jj = sp_indptr_gpu[i]; jj < sp_indptr_gpu[i + 1]; jj++)
    {
        int j = sp_indices_gpu[jj];
        c_gpu[i * (*p) + k] += a_gpu[jj] * b_gpu[j * (*p) + k];
    }
}

void SparseMatmul::forward(bool training)
{
    timer_start(TMR_SPMATMUL_FW);

    int *sp_indptr_gpu, *sp_indices_gpu;

    CHECK(cudaMalloc(&a_gpu, sizeof(float) * m * n));
    CHECK(cudaMalloc(&b_gpu, sizeof(float) * n * p));
    CHECK(cudaMalloc(&c_gpu, sizeof(float) * m * p));
    CHECK(cudaMalloc(&m_gpu, sizeof(int)));
    CHECK(cudaMalloc(&n_gpu, sizeof(int)));
    CHECK(cudaMalloc(&p_gpu, sizeof(int)));

    CHECK(cudaMalloc(&sp_indptr_gpu, sizeof(int) * sp->indptr.size()));
    CHECK(cudaMalloc(&sp_indices_gpu, sizeof(int) * sp->indices.size()));

    CHECK(cudaMemcpy(a_gpu, &(a->data[0]), sizeof(float) * a->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_gpu, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c_gpu, &(c->data[0]), sizeof(float) * c->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(m_gpu, &m, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(n_gpu, &n, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p_gpu, &p, sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(sp_indptr_gpu, &(sp->indptr[0]), sizeof(int) * sp->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_indices_gpu, &(sp->indices[0]), sizeof(int) * sp->indices.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(sp->indptr.size() - 1, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    dim3 blocksPerGridZero(m, 1, 1);
    gpu_zero<<<blocksPerGridZero, threadsPerBlock>>>(c_gpu, p_gpu);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    gpu_sparse_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, sp_indptr_gpu, sp_indices_gpu, p_gpu);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    /*c->zero();
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++)
        {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }*/

    // convert an array to a std::vector:
    // int src[] = { 1, 2, 3, 4, 5 };
    // int n = sizeof(src) / sizeof(src[0]);
    // std::vector<int> dest(src, src + n);

    float *c_data_from_gpu;
    c_data_from_gpu = (float *)malloc(m * p * sizeof(float));
    CHECK(cudaMemcpy(c_data_from_gpu, c_gpu, sizeof(float) * m * p, cudaMemcpyDeviceToHost));
    std::vector<float> c_data_from_gpu_vector(c_data_from_gpu, c_data_from_gpu + m * p);
    c->data = c_data_from_gpu_vector;

    CHECK(cudaFree(a_gpu));
    CHECK(cudaFree(b_gpu));
    CHECK(cudaFree(c_gpu));
    CHECK(cudaFree(m_gpu));
    CHECK(cudaFree(n_gpu));
    CHECK(cudaFree(p_gpu));
    CHECK(cudaFree(sp_indptr_gpu));
    CHECK(cudaFree(sp_indices_gpu));

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