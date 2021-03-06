
# An Improved Winograd Convolution on x86 CPUs

> This software is the codes for our article "A Fine-grained Optimization to Winograd Convolution Based on Micro-architectural Features of CPU". 
> It is constructed based on [Intel DNNL](https://github.com/intel/mkl-dnn).

We proposed a noval design about register block, data layout and cache block in Winograd convolution to address the performance limitations we demonstrated from experiments. 

Our experients had carried on two x86 servers. The first server equipped with an Intel Xeon Second Generation Scalable Processors Gold6230N, and the second server is a dual-socket system which has two Gold6150 processors in its two sockets. Both of these two servers run on Linux system(CentOS 7.6). 

The codes with different purpose are organized in different branches. The main-branch contains codes including all our three optimizaions, while the original-branch is the original DNNL codes with some information printed. 

To run those experiments, the machine should have DNNL built from source first. Briefly speaking, make a new directory ``build`` in the directory ``oneDNN``, and then run ``cmake ..`` in the directory ``build``. After that, DNNL can be built and installed with command ``make`` and ``sudo make install`` seperately. 
More details can be refered [here](https://oneapi-src.github.io/oneDNN/dev_guide_build.html). And then one can compile the test files with a command ``g++ test.cpp -ldnnl -std=c++11``. Usually manually specifying environment variables such as ``OMP_NUM_THREADS`` and ``OMP_PROC_BIND`` is necessary at runtime. 

# An Improved Winograd Convolution on NUMA system with multiple x86 CPUs

> This software is the codes for our article "******". 
> It is constructed based on [Intel DNNL](https://github.com/intel/mkl-dnn).

We proposed a noval design for Winograd convolution to address the bottleneck problem caused by NUMA features on multi-CPU system. By designing a new computing pattern and three optimizations (i.e., new data layout, cache block searching and new matrix multiplication) based on the micro-architectural features of CPUs, we alleviate the memory bandwidth requirement and therefore improve the performance. The codes are in the branch multi-cpu. 
