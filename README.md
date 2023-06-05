# High-Performance Data & Graph Analytics - Spring 2023

The contest of the High-Performance Graph Analytics course aimed to enhance the performance of a sequential CPU implementation of a Graph Convolutional Network (GCN) in C++ for Semi-Supervised Classification. The provided code was based on the original GCN model developed by Kipf et al. The goal was to leverage GPU acceleration without using CUDA libraries while maintaining a satisfactory level of accuracy compared to the baseline implementation.

The contest consisted of two main steps:
1. GPU acceleration: Focus on the acceleration of the workload;
2. Model optimization: Focus on exploring various approaches to modify the model or experiment with different hyperparameters to improve the accuracy of the predictions.

In `/original-code` you can can find the c++ code that we had to optimize, in `/gpu_track` there is the code for the first part, and in `/model-optimize` our solution for the second part.

### Build and run
You can build and execute the existing implementation by running the following commands:

```sh
make
./exec/gcn-seq cora # dataset name from the dataset folder: [cora, pubmed, citeseer, reddit]
```

In every folder there is a `colab.ipynb` file: click on the "open in Colab" button to execute the code on Colab.

###  Project structure
In the `.\src\` folder of `/original-code`, `/gpu_track`, `/model-optimize` you will find the main components of the implementation.
As usual, the core is the `main.cpp` file that parses the selected dataset and creates and object of type `GCN`.
During the initialization phase all the layers for the model are constructed. 
Consequently, the model is then run by calling the function `GCN::run()`.
This function, based on the parameters set by `GCN::get_default()`, will execute a predefined number of epoch during the training phase.
In addition, in `/gpu_track` and `/model-optimize` we have `kernels.cu` where we have the implementation of CUDA kernels functions called by the host.
