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
    float *b_data;
    float *b_grad;
    float *b_sum;
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
    float *b_data;
    float *b_grad;
    int *sp_indptr, *sp_indices;
    int *i_index;
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
    int *truth_training;
    int *truth_validation;
    int *truth_testing;
    float *loss;
    int num_classes;
    int *count_gpu;
    int *truth_training_gpu;
    int *truth_validation_gpu;
    int *truth_testing_gpu;
    float *total_loss_gpu;
    int num_epochs;
public:
    CrossEntropyLoss(Variable *logits, int *truth_training, int *truth_validation, int *truth_testing, float *loss, int num_classes, int num_epochs);
    ~CrossEntropyLoss();
    void forward(bool);
    void backward();
};

class ReLU: public Module {
    Variable *in;
    bool *mask_gpu;
public:
    ReLU(Variable *in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class Dropout: public Module {
    Variable *in;
    float p;
    bool isFirst;
    bool *mask_gpu;
    std::string input_name;
public:
    Dropout(Variable *in, float p, bool isFirst, std::string input_name);
    ~Dropout();
    void forward(bool);
    void backward();
};


#define MODULE_H
#endif
