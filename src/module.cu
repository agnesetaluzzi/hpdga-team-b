#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include <cmath>

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

float *a_data, *b_data, *c_data;
float *a_grad, *b_grad, *c_grad;
int *M, *N, *P;

/**
 * Dense matrix multiplication layer. 
*/
Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p): a(a), b(b), c(c), m(m), n(n), p(p) {
	cudaMalloc(&a_data, a->data.size() * sizeof(float));
	cudaMalloc(&b_data, b->data.size() * sizeof(float));
	cudaMalloc(&c_data, c->data.size() * sizeof(float));
	
	cudaMalloc(&a_grad, a->grad.size() * sizeof(float));
	cudaMalloc(&b_grad, b->grad.size() * sizeof(float));
	cudaMalloc(&c_grad, c->grad.size() * sizeof(float));
	
	cudaMalloc(&M, sizeof(int));
	cudaMalloc(&N, sizeof(int));
	cudaMalloc(&P, sizeof(int));
}

__global__ void gpu_matmul_forward(float *a_gpu, float *b_gpu, float *c_gpu, int *m, int *n, int *p)
{
    int i = blockIdx.x;
    int k = threadIdx.x;

    c_gpu[i * (*p) + k] = 0;

    for (int j = 0; j < (*n); j++)
        c_gpu[i * (*p) + k] += a_gpu[i * (*n) + j] * b_gpu[j * (*p) + k];
}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);

    CHECK(cudaMemcpy(a_data, &(a->data[0]), sizeof(float) * a->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c_data, &(c->data[0]), sizeof(float) * c->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(M, &m, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(N, &n, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(P, &p, sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 blocksPerGrid(m, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(a_data, b_data, c_data, M, N, P);
    CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	
    CHECK(cudaMemcpy(&c->data[0], c_data, sizeof(float) * c->data.size(), cudaMemcpyDeviceToHost));
    timer_stop(TMR_MATMUL_FW);
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

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    CHECK(cudaMemcpy(a_grad, &(a->grad[0]), sizeof(float) * a->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a_data, &(a->data[0]), sizeof(float) * a->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_grad, &(b->grad[0]), sizeof(float) * b->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c_grad, &(c->grad[0]), sizeof(float) * c->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(M, &m, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(N, &n, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(P, &p, sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 blocksPerGrid1(m, 1, 1);
    dim3 threadsPerBlock1(n, 1, 1);
    gpu_matmul_backward1<<<blocksPerGrid1, threadsPerBlock1>>>(a_grad, a_data, b_data, b_grad, c_grad, M, N, P);
	CHECK(cudaDeviceSynchronize());
		
	dim3 blocksPerGrid2(n, 1, 1);
    dim3 threadsPerBlock2(p, 1, 1);
	gpu_matmul_backward2<<<blocksPerGrid2, threadsPerBlock2>>>(a_grad, a_data, b_data, b_grad, c_grad, M, N, P);
	CHECK(cudaDeviceSynchronize());
		
	CHECK(cudaMemcpy(&a->grad[0], a_grad, sizeof(float) * a->grad.size(), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(&b->grad[0], b_grad, sizeof(float) * b->grad.size(), cudaMemcpyDeviceToHost));
	
    /*a->zero_grad();
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
        }*/
	
    timer_stop(TMR_MATMUL_BW);
}


// ################################################################################################################

float *a2_data, *b2_data, *c2_data;
float *a2_grad, *b2_grad, *c2_grad;
int *sp_indptr, *sp_indices;
int *M2, *N2, *P2;

/**
 * A sparse matrix multiplication layer.
*/

SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p): a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {
	cudaMalloc(&a2_data, a->data.size() * sizeof(float));
	cudaMalloc(&b2_data, b->data.size() * sizeof(float));
	cudaMalloc(&c2_data, c->data.size() * sizeof(float));
	
	cudaMalloc(&a2_grad, a->grad.size() * sizeof(float));
	cudaMalloc(&b2_grad, b->grad.size() * sizeof(float));
	cudaMalloc(&c2_grad, c->grad.size() * sizeof(float));
	
	cudaMalloc(&sp_indptr, sp->indptr.size() * sizeof(float));
	cudaMalloc(&sp_indices, sp->indices.size() * sizeof(float));
	
	cudaMalloc(&M2, sizeof(int));
	cudaMalloc(&N2, sizeof(int));
	cudaMalloc(&P2, sizeof(int));
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

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
	
	CHECK(cudaMemcpy(sp_indptr, &(sp->indptr[0]), sizeof(int) * sp->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_indices, &(sp->indices[0]), sizeof(int) * sp->indices.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a2_data, &(a->data[0]), sizeof(float) * a->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b2_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c2_data, &(c->data[0]), sizeof(float) * c->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(M2, &m, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(N2, &n, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(P2, &p, sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 blocksPerGrid(sp->indptr.size() - 1, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_sparse_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(a2_data, b2_data, c2_data, sp_indptr, sp_indices, P2);
    CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	
    CHECK(cudaMemcpy(&c->data[0], c2_data, sizeof(float) * c->data.size(), cudaMemcpyDeviceToHost));
	
    /*c->zero();
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }*/
    timer_stop(TMR_SPMATMUL_FW);
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

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
	
	// It doesn't work
	/*CHECK(cudaMemcpy(sp_indptr, &(sp->indptr[0]), sizeof(int) * sp->indptr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_indices, &(sp->indices[0]), sizeof(int) * sp->indices.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a2_data, &(a->data[0]), sizeof(float) * a->data.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b2_grad, &(b->grad[0]), sizeof(float) * b->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c2_grad, &(c->grad[0]), sizeof(float) * c->grad.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(M2, &m, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(N2, &n, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(P2, &p, sizeof(int), cudaMemcpyHostToDevice));
	
	int max_sp_indptr = 0;
	for(int i = 0; i < sp->indptr.size(); i++){
		if(sp->indptr[i] > max_sp_indptr) max_sp_indptr = sp->indptr[i];
	}
	
	dim3 blocksPerGrid(max_sp_indptr, 1, 1);
    dim3 threadsPerBlock(p, 1, 1);
    gpu_sparse_matmul_backward<<<blocksPerGrid, threadsPerBlock>>>(a2_data, b2_grad, c2_grad, sp_indptr, sp_indices, P2, sp->indptr.size());
    CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	
	CHECK(cudaMemcpy(&b->grad[0], b2_grad, sizeof(float) * b->grad.size(), cudaMemcpyDeviceToHost));*/
	
    b->zero_grad();
    int row = 0;
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
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
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim): in(in), out(out), graph(graph), dim(dim) {
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

void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    out->zero();
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
            for (int j = 0; j < dim; j++)
                // This only works for undirected graphs. Should be out[dst] += coef * in[src]
                out->data[src * dim + j] += coef * in->data[dst * dim + j];
        }
    timer_stop(TMR_GRAPHSUM_FW);
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

void GraphSum::backward() {
    timer_start(TMR_GRAPHSUM_BW);
    in->zero_grad();
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
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
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

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
ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
    delete[] mask;
}

void ReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = in->data[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in->data[i] = 0;
    }
    timer_stop(TMR_RELU_FW);
}

void ReLU::backward() {
    timer_start(TMR_RELU_BW);
    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i]) in->grad[i] = 0;
    timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at each step during training time to prevent overfitting. 
 * Inputs that are not set to 0 are scaled up by 1/(1-P).
*/
Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    if (!in->grad.empty()) 
        mask = new int[in->data.size()];
    else mask = nullptr;
}

Dropout::~Dropout() {
    if (mask) delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = (int)RAND() >= threshold;
        in->data[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep;
    }
    timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
    if (!mask) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################