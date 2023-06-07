#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include "../include/kernels.cuh"
#include <vector>
#include <algorithm>
#define BLOCK_DIM 256
#define TILE_WIDTH 32

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
int max_dim_dropout = 0;
unsigned long long *rand1_gpu, *rand2_gpu;
int epoch = 0;
float *original_input_data;
int *src_index;

cudaStream_t stream1;

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
	
    if (m < 20000)
    {
        CHECK(cudaMalloc(&b_sum, a->data.size() * b->data.size() * sizeof(float)));
    }

    CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
}

Matmul::~Matmul()
{
    CHECK(cudaFree(b_data));
    CHECK(cudaFree(b_grad));
    if (m < 20000)
    {
        CHECK(cudaFree(b_sum));
    }
    CHECK(cudaStreamDestroy(stream1));
}

void Matmul::forward(bool training)
{
    timer_start(TMR_MATMUL_FW);

    if (epoch == 0 || !training)
        CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));

    if (m < 20000)
    {
        dim3 blocksPerGrid((m * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
        dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
        gpu_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, b_data, layer2_var1_data, m, n, p);
    }
    else
    {
        dim3 blocksPerGrid((p / TILE_WIDTH) + 1, (m / TILE_WIDTH) + 1, 1);
        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
        gpu_matmul_forward2<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, b_data, layer2_var1_data, m, n, n, p, m, p);
    }
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_FW);
}

void Matmul::backward()
{
    timer_start(TMR_MATMUL_BW);
    
    if (epoch == 0)
        CHECK(cudaMemcpyAsync(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice, stream1));

    dim3 blocksPerGrid1((m * n + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock1(BLOCK_DIM, 1, 1);
    gpu_matmul_backward1<<<blocksPerGrid1, threadsPerBlock1, 0, stream1>>>(layer1_var2_grad, b_data, layer2_var1_grad, m, n, p);
    CHECK_KERNELCALL();

    if (m < 20000)
    {
        int multiple32 = m + 32 - 1;
        multiple32 -= (multiple32 % 32);

        dim3 blocksPerGrid0((n * p + BLOCK_DIM - 1) / BLOCK_DIM, multiple32, 1);
        dim3 threadsPerBlock0(BLOCK_DIM, 1, 1);
        gpu_matmul_backward2_copy<<<blocksPerGrid0, threadsPerBlock0>>>(layer1_var2_grad, layer1_var2_data, layer2_var1_grad, m, n, p, b_sum);
        CHECK_KERNELCALL();

        dim3 blocksPerGridSum((n * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
        dim3 threadsPerBlockSum(BLOCK_DIM, 1, 1);

        int dim = m;
        int dim2 = m;

        for (int x = 0; x < ceil(log2(m)); x++)
        {
            dim2 = ceil(dim2 / 2.0);
            multiple32 = dim2 + 32 - 1;
            multiple32 -= (dim2 % 32);
            blocksPerGridSum.y = multiple32;
            gpu_matmul_backward2_sum<<<blocksPerGridSum, threadsPerBlockSum>>>(b_sum, dim, dim2, m, n, p);
            CHECK_KERNELCALL();
            dim = dim2;
        }

        dim3 blocksPerGrid2((n * p + BLOCK_DIM - 1), 1, 1);
        dim3 threadsPerBlock2(BLOCK_DIM, 1, 1);
        gpu_matmul_backward2<<<blocksPerGrid2, threadsPerBlock2>>>(b_grad, n, p, b_sum);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
    }
    else
    {
        dim3 blocksPerGrid2((n * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
        dim3 threadsPerBlock2(BLOCK_DIM, 1, 1);
        gpu_matmul_backward3<<<blocksPerGrid2, threadsPerBlock2>>>(b_grad, layer1_var2_data, layer2_var1_grad, m, n, p);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
    }

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
	
	std::vector<std::pair<int, int> > i_length;
	for(int i = 0; i < sp->indptr.size() - 1; i++){
		i_length.push_back({sp->indptr[i + 1] - sp->indptr[i], i});
	}
	std::sort(i_length.begin(), i_length.end());
	std::reverse(i_length.begin(), i_length.end());
	int *i_length_index;
	i_length_index = (int *)malloc((sp->indptr.size() - 1) * sizeof(int));
	for(int i = 0; i < sp->indptr.size() - 1; i++){
		i_length_index[i] = i_length[i].second;
	}
	CHECK(cudaMalloc(&i_index, (sp->indptr.size() - 1) * sizeof(int)));
	CHECK(cudaMemcpy(i_index, i_length_index, (sp->indptr.size() - 1) * sizeof(int), cudaMemcpyHostToDevice));
}

SparseMatmul::~SparseMatmul()
{
    CHECK(cudaFree(b_data));
    CHECK(cudaFree(b_grad));
    CHECK(cudaFree(sp_indptr));
    CHECK(cudaFree(sp_indices));
    CHECK(cudaFree(i_index));
}

void SparseMatmul::forward(bool training)
{
    timer_start(TMR_SPMATMUL_FW);

    if (epoch == 0 || !training)
        CHECK(cudaMemcpy(b_data, &(b->data[0]), sizeof(float) * b->data.size(), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(((sp->indptr.size() - 1) * p + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_sparse_matmul_forward<<<blocksPerGrid, threadsPerBlock>>>(i_index, input_data, b_data, layer1_var1_data, sp_indptr, sp_indices, p, (sp->indptr.size() - 1) * p);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_FW);
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

/**
 * A specialized sparse matrix multiplication for graphs.
 */
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim, bool isFirst) : in(in), out(out), graph(graph), dim(dim), isFirst(isFirst)
{
    if (isFirst)
    {
        CHECK(cudaMalloc(&layer1_var2_data, out->data.size() * sizeof(float)));
        CHECK(cudaMalloc(&layer1_var2_grad, out->grad.size() * sizeof(float)));
		
		std::vector<std::pair<int, int> > src_length;
		for(int i = 0; i < graph->indptr.size() - 1; i++){
			src_length.push_back({graph->indptr[i + 1] - graph->indptr[i], i});
		}
		std::sort(src_length.begin(), src_length.end());
		std::reverse(src_length.begin(), src_length.end());
		int *src_length_index;
		src_length_index = (int *)malloc((graph->indptr.size() - 1) * sizeof(int));
		for(int i = 0; i < graph->indptr.size() - 1; i++){
			src_length_index[i] = src_length[i].second;
		}
		CHECK(cudaMalloc(&src_index, (graph->indptr.size() - 1) * sizeof(int)));
		CHECK(cudaMemcpy(src_index, src_length_index, (graph->indptr.size() - 1) * sizeof(int), cudaMemcpyHostToDevice));
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
    for(int i = 1; i < graph->indptr.size(); i++)
    {
    	max_diff = max(max_diff, graph->indptr[i] - graph->indptr[i - 1]);
    }
}

GraphSum::~GraphSum()
{
    CHECK(cudaFree(graph_indptr));
    CHECK(cudaFree(graph_indices));
}

void GraphSum::forward(bool training)
{
    timer_start(TMR_GRAPHSUM_FW);

	if (graph->indptr.size() - 1 < 5000) {
		dim3 blocksPerGrid0(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
		dim3 threadsPerBlock0(BLOCK_DIM, 1, 1);
		if (isFirst)
			gpu_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer1_var2_data, dim, (graph->indptr.size() - 1) * dim);
		if (!isFirst)
			gpu_zero<<<blocksPerGrid0, threadsPerBlock0>>>(output_data, dim, (graph->indptr.size() - 1) * dim);
		CHECK_KERNELCALL();

		dim3 blocksPerGrid(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, sqrt(max_diff), 1);
		dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
		if (isFirst)
			gpu_graph_sum_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var1_data, layer1_var2_data, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
		else
			gpu_graph_sum_forward<<<blocksPerGrid, threadsPerBlock>>>(layer2_var1_data, output_data, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
		CHECK_KERNELCALL();
		CHECK(cudaDeviceSynchronize());
	} else {
	    dim3 blocksPerGrid(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
		dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
		if (isFirst)
			gpu_graph_sum_forward2<<<blocksPerGrid, threadsPerBlock>>>(src_index, layer1_var1_data, layer1_var2_data, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
		else
			gpu_graph_sum_forward2<<<blocksPerGrid, threadsPerBlock>>>(src_index, layer2_var1_data, output_data, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
		CHECK_KERNELCALL();
		CHECK(cudaDeviceSynchronize());
	}
	if (!isFirst)
	{
        CHECK(cudaMemcpy(&out->data[0], output_data, sizeof(float) * out->data.size(), cudaMemcpyDeviceToHost));
	}
    timer_stop(TMR_GRAPHSUM_FW);
}

void GraphSum::backward()
{
    timer_start(TMR_GRAPHSUM_BW);

	if (graph->indptr.size() - 1 < 5000) {
		dim3 blocksPerGrid0(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
		dim3 threadsPerBlock0(BLOCK_DIM, 1, 1);
		if (isFirst)
			gpu_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer1_var1_grad, dim, (graph->indptr.size() - 1) * dim);
		if (!isFirst)
			gpu_zero<<<blocksPerGrid0, threadsPerBlock0>>>(layer2_var1_grad, dim, (graph->indptr.size() - 1) * dim);

		dim3 blocksPerGrid(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, sqrt(max_diff), 1);
		dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
		if (isFirst)
			gpu_graph_sum_backward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var1_grad, layer1_var2_grad, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
		else
			gpu_graph_sum_backward<<<blocksPerGrid, threadsPerBlock>>>(layer2_var1_grad, output_grad, graph_indptr, graph_indices, dim, sqrt(max_diff), (graph->indptr.size() - 1) * dim);
		CHECK_KERNELCALL();
		CHECK(cudaDeviceSynchronize());
	} else {
	    dim3 blocksPerGrid(((graph->indptr.size() - 1) * dim + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
		dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
		if (isFirst)
			gpu_graph_sum_backward2<<<blocksPerGrid, threadsPerBlock>>>(src_index, layer1_var1_grad, layer1_var2_grad, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
		else
			gpu_graph_sum_backward2<<<blocksPerGrid, threadsPerBlock>>>(src_index, layer2_var1_grad, output_grad, graph_indptr, graph_indices, dim, (graph->indptr.size() - 1) * dim);
		CHECK_KERNELCALL();
		CHECK(cudaDeviceSynchronize());
	}
    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth_training, int *truth_validation, int *truth_testing, float *loss, int num_classes, int num_epochs) :
        logits(logits), truth_training(truth_training), truth_validation(truth_validation), truth_testing(truth_testing), loss(loss), num_classes(num_classes),  num_epochs(num_epochs)
{
    CHECK(cudaMalloc(&count_gpu, sizeof(int)));
    CHECK(cudaMalloc(&total_loss_gpu, sizeof(float)));

    CHECK(cudaMalloc(&truth_training_gpu, sizeof(int) * (logits->data.size() / num_classes)));
    CHECK(cudaMalloc(&truth_validation_gpu, sizeof(int) * (logits->data.size() / num_classes)));
    CHECK(cudaMalloc(&truth_testing_gpu, sizeof(int) * (logits->data.size() / num_classes)));
    CHECK(cudaMemcpy(truth_training_gpu, truth_training, sizeof(int) * (logits->data.size() / num_classes), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(truth_validation_gpu, truth_validation, sizeof(int) * (logits->data.size() / num_classes), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(truth_testing_gpu, truth_testing, sizeof(int) * (logits->data.size() / num_classes), cudaMemcpyHostToDevice));
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
    CHECK(cudaFree(original_input_data));
    CHECK(cudaFree(rand1_gpu));
	CHECK(cudaFree(rand2_gpu));
	CHECK(cudaFree(src_index));
}

void CrossEntropyLoss::forward(bool training) {
    
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    CHECK(cudaMemset(count_gpu, 0, sizeof(int)));
    CHECK(cudaMemset(total_loss_gpu, 0.0, sizeof(float)));

    dim3 blocksPerGrid(((logits->data.size() / num_classes) + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    if (training)
    {
        gpu_cross_entropy_loss_forward1<<<blocksPerGrid, threadsPerBlock>>>(truth_training_gpu, count_gpu, output_data, total_loss_gpu, output_grad, training, (logits->data.size() / num_classes), num_classes);
    }
    else
    {
        if (epoch < num_epochs)
        {
            gpu_cross_entropy_loss_forward1<<<blocksPerGrid, threadsPerBlock>>>(truth_validation_gpu, count_gpu, output_data, total_loss_gpu, output_grad, training, (logits->data.size() / num_classes), num_classes);
            epoch++;
        }
        else
        {
            gpu_cross_entropy_loss_forward1<<<blocksPerGrid, threadsPerBlock>>>(truth_testing_gpu, count_gpu, output_data, total_loss_gpu, output_grad, training, (logits->data.size() / num_classes), num_classes);
        }
    }
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

void ReLU::forward(bool training)
{
    timer_start(TMR_RELU_FW);
	
    dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    gpu_relu_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, mask_gpu, training, in->data.size());
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
	
    timer_stop(TMR_RELU_FW);
}

void ReLU::backward()
{
    timer_start(TMR_RELU_BW);
	
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
Dropout::Dropout(Variable *in, float p, bool isFirst, std::string input_name) : isFirst(isFirst), input_name(input_name)
{
    this->in = in;
    this->p = p;
    if (!in->grad.empty())
    {
        CHECK(cudaMalloc(&mask_gpu, in->data.size() * sizeof(bool)));
    }

    if (isFirst)
    {
        CHECK(cudaMalloc(&input_data, in->data.size() * sizeof(float)));
        CHECK(cudaMalloc(&input_grad, in->grad.size() * sizeof(float)));
        max_dim_dropout = in->data.size();
        if (input_name != "citeseer") 
        {
            CHECK(cudaMalloc(&original_input_data, in->data.size() * sizeof(float)));
            CHECK(cudaMemcpy(original_input_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
        }
    }
    else
    {
        srand(time(NULL));
        if (in->data.size() > max_dim_dropout)
            max_dim_dropout = in->data.size();
		unsigned long long *rand1, *rand2;
        rand1 = (unsigned long long *)malloc(max_dim_dropout * sizeof(unsigned long long));
        rand2 = (unsigned long long *)malloc(max_dim_dropout * sizeof(unsigned long long));
        for (int i = 0; i < max_dim_dropout; i++)
        {
            rand1[i] = rand();
            rand2[i] = rand();
            while (rand1[i] == 0 || rand2[i] == 0)
            {
                rand1[i] = rand();
                rand2[i] = rand();
            }
        }
        CHECK(cudaMalloc(&rand1_gpu, max_dim_dropout * sizeof(unsigned long long)));
        CHECK(cudaMalloc(&rand2_gpu, max_dim_dropout * sizeof(unsigned long long)));
        CHECK(cudaMemcpy(rand1_gpu, rand1, max_dim_dropout * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(rand2_gpu, rand2, max_dim_dropout * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    }
}

Dropout::~Dropout()
{
    if (!in->grad.empty())
    {
        CHECK(cudaFree(mask_gpu));
    }
}

void Dropout::forward(bool training)
{
    if (!training)
    {
        if (isFirst)
        {
            if (input_name != "citeseer") {
                dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
                dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
                gpu_set_original_input<<<blocksPerGrid, threadsPerBlock>>>(input_data, original_input_data, in->data.size());
                CHECK_KERNELCALL();
            }
            else {
                CHECK(cudaMemcpy(input_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
            }
	    }
        return;
    }
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);

    if (isFirst)
    {
        if (input_name != "citeseer") {
            dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
            dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
            gpu_set_original_input<<<blocksPerGrid, threadsPerBlock>>>(input_data, original_input_data, in->data.size());
            CHECK_KERNELCALL();
        } else {
            CHECK(cudaMemcpy(input_data, &(in->data[0]), sizeof(float) * in->data.size(), cudaMemcpyHostToDevice));
        }
    }

    bool isMask = false;
    if (!in->grad.empty())
    {
        isMask = true;
    }
    
    dim3 blocksPerGrid((in->data.size() + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    if (isFirst)
        gpu_dropout_forward<<<blocksPerGrid, threadsPerBlock>>>(input_data, mask_gpu, isMask, threshold, scale, in->data.size(), rand1_gpu, rand2_gpu);
    else
        gpu_dropout_forward<<<blocksPerGrid, threadsPerBlock>>>(layer1_var2_data, mask_gpu, isMask, threshold, scale, in->data.size(), rand1_gpu, rand2_gpu);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward()
{
    if (in->grad.empty())
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
