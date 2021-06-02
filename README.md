
# An Improved Winograd Convolution on x86 CPUs

> This software is the codes for our article "A Fine-grained Optimization to Winograd Convolution Based on Micro-architectural Features of CPU". 
> It is constructed based on [Intel DNNL](https://github.com/intel/mkl-dnn).

We proposed a noval design about register block, data layout and cache block in Winograd convolution to address the performance limitations we demonstrated from experiments. 

Our experients were carried on two x86 servers. The first server equipped with an Intel Xeon Second
Generation Scalable Processors Gold6230N, and the second server is a dual-socket system which has two Gold6150 processors in its two sockets. Those servers run on Linux system(CentOS 7.6). 

We organize codes with different purpose in different branches. The master-branch contain codes including all our three optimizaions, and the original-branch is the original DNNL codes with some information printed. 
