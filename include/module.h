#ifndef MODULE_H

#include "variable.h"
#include "sparse.h"

class Module {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};

class Matmul: public Module {
    Variable *a, *b, *c;
    int m, n, p;
    float *a_data, *b_data, *c_data;
    float *a_grad, *b_grad, *c_grad;
public:
    Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p);
    ~Matmul();
    void forward(bool);
    void backward();
};

class SparseMatmul: public Module {
    Variable *a, *b, *c;
    SparseIndex *sp;
    int m, n, p;
    float *a_data, *b_data, *c_data;
    float *b_grad, *c_grad;
    int *sp_indptr, *sp_indices;
public:
    SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p);
    ~SparseMatmul();
    void forward(bool);
    void backward();
};

class GraphSum: public Module {
    Variable *in, *out;
    SparseIndex *graph;
    int dim;
    float *in_data, *out_data;
    float *in_grad, *out_grad;
    int *graph_indptr, *graph_indices;
    bool isFirst;
    int max_diff;
public:
    GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim, bool isFirst);
    ~GraphSum();
    void forward(bool);
    void backward();
};

class CrossEntropyLoss: public Module {
    Variable *logits;
    int *truth;
    float *loss;
    int num_classes;
    int *count_gpu;
    int *truth_gpu;
    float *total_loss_gpu;
public:
    CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes);
    ~CrossEntropyLoss();
    void forward(bool);
    void backward();
};

class ReLU: public Module {
    Variable *in;
    bool *mask;
    float *in_data, *in_grad;
    bool *mask_gpu;
public:
    ReLU(Variable *in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class Dropout: public Module {
    Variable *in;
    int *mask;
    float p;
    bool isFirst;
    int *mask_gpu;
public:
    Dropout(Variable *in, float p, bool isFirst);
    ~Dropout();
    void forward(bool);
    void backward();
};


#define MODULE_H
#endif