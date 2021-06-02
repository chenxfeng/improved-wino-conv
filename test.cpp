
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include "dnnl.hpp"
#include "example_utils.hpp"
using namespace dnnl;
using namespace std::chrono;

#include <random>

// global time recoders
float seconds1 = 0.0f;
float seconds2 = 0.0f;
float seconds3 = 0.0f;
float seconds4 = 0.0f;
const int ExecCount = 10;
float bestSecond = 0;
int bestNblock;

/*
** following conv-layers are picked from popular CNN models.
*/

///DenseNet HW57/.../8 W_SGD
//const memory::dims strides = {1, 1};
//const memory::dims padding = {1, 1};
//const memory::dim BATCH = 512;
//const memory::dim IC = 128, OC = 32;
//const memory::dim IH = 8, KH = 3, OH = 8;
//const memory::dim IW = 8, KW = 3, OW = 8;

///NIN nega-512
//const memory::dims strides = {1, 1};
//const memory::dims padding = {1, 1};
//const memory::dim BATCH = 512;
//const memory::dim IC = 384, OC = 1024;
//const memory::dim IH = 6, KH = 3, OH = 6;
//const memory::dim IW = 6, KW = 3, OW = 6;

///r-cnn(AlexNet) nega-512
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 384, OC = 384;
// const memory::dim IH = 13, KH = 3, OH = 13;
// const memory::dim IW = 13, KW = 3, OW = 13;

///r-cnn nega-perf-512
//const memory::dims strides = {1, 1};
//const memory::dims padding = {1, 1};
//const memory::dim BATCH = 512;
//const memory::dim IC = 256, OC = 384;
//const memory::dim IH = 13, KH = 3, OH = 13;
//const memory::dim IW = 13, KW = 3, OW = 13;

///r-cnn nega-512
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 384, OC = 256;
// const memory::dim IH = 13, KH = 3, OH = 13;
// const memory::dim IW = 13, KW = 3, OW = 13;

///expl custom
//const memory::dims strides = {1, 1};
//const memory::dims padding = {1, 1};
//const memory::dim BATCH = 512;
//const memory::dim IC = 512, OC = 512;
//const memory::dim IH = 28, KH = 3, OH = 28;
//const memory::dim IW = 28, KW = 3, OW = 28;

///VGG conv1_2 W_SGD
//const memory::dims strides = {1, 1};
//const memory::dims padding = {1, 1};
//const memory::dim BATCH = 512;
//const memory::dim IC = 64, OC = 64;
//const memory::dim IH = 224, KH = 3, OH = 224;
//const memory::dim IW = 224, KW = 3, OW = 224;


///VGG16 conv2_2 W_SGD
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 128, OC = 128;
// const memory::dim IH = 112, KH = 3, OH = 112;
// const memory::dim IW = 112, KW = 3, OW = 112;

///VGG16 conv3_1 SGD
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 128, OC = 256;
// const memory::dim IH = 56, KH = 3, OH = 56;
// const memory::dim IW = 56, KW = 3, OW = 56;

///VGG16 conv3_2/3
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 256, OC = 256;
// const memory::dim IH = 56, KH = 3, OH = 56;
// const memory::dim IW = 56, KW = 3, OW = 56;

///VGG16 conv4_1
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 256, OC = 512;
// const memory::dim IH = 28, KH = 3, OH = 28;
// const memory::dim IW = 28, KW = 3, OW = 28;

///VGG16 conv4_2/3 expl 
const memory::dims strides = {1, 1};
const memory::dims padding = {1, 1};
const memory::dim BATCH = 512;
const memory::dim IC = 512, OC = 512;
const memory::dim IH = 28, KH = 3, OH = 28;
const memory::dim IW = 28, KW = 3, OW = 28;

///VGG16 conv5_1/2/3
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 512, OC = 512;
// const memory::dim IH = 14, KH = 3, OH = 14;
// const memory::dim IW = 14, KW = 3, OW = 14;

///first conv layer of VGG (conv1_1): W_SGD
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 3, OC = 64;
// const memory::dim IH = 224, KH = 3, OH = 224;
// const memory::dim IW = 224, KW = 3, OW = 224;

///res2a_branch2b W_SGD/only-3-layers in ResNet152
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 64, OC = 64;
// const memory::dim IH = 56, KH = 3, OH = 56;
// const memory::dim IW = 56, KW = 3, OW = 56;

///res3a_branch2b
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 128, OC = 128;
// const memory::dim IH = 28, KH = 3, OH = 28;
// const memory::dim IW = 28, KW = 3, OW = 28;

///res4a_branch2b ///W_S_G_D better
//const memory::dims strides = {1, 1};
//const memory::dims padding = {1, 1};
//const memory::dim BATCH = 512;
//const memory::dim IC = 256, OC = 256;
//const memory::dim IH = 14, KH = 3, OH = 14;
//const memory::dim IW = 14, KW = 3, OW = 14;

///res5b_branch2b expl
// const memory::dims strides = {1, 1};
// const memory::dims padding = {1, 1};
// const memory::dim BATCH = 512;
// const memory::dim IC = 512, OC = 512;
// const memory::dim IH = 7, KH = 3, OH = 7;
// const memory::dim IW = 7, KW = 3, OW = 7;


void performance_profiling(engine::kind engine_kind, int argc, char **argv) {

    // Initialize engine
    engine eng(engine_kind, 0);
    // Initialize stream
    stream s(eng);

    // initializing non-zero values for user data: src, weights and bias
    // std::vector<float> conv_src(BATCH * IC * IH * IW); // net src
    // std::vector<float> conv_wei(OC * IC * KH * KW);
    // std::vector<float> conv_bia(OC);
    // std::vector<float> conv_dst(BATCH * OC * OH * OW); // net dst
    // std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());

    for (int nblock = 0; nblock < 100; ++nblock) {
        // create memory descriptors for forward convolution
        auto conv_src_md = memory::desc({BATCH, IC, IH, IW}, memory::data_type::f32, memory::format_tag::any);
        auto conv_wei_md = memory::desc({OC, IC, KH, KW}, memory::data_type::f32, memory::format_tag::any);
        auto conv_bia_md = memory::desc({OC}, memory::data_type::f32, memory::format_tag::any);
        auto conv_dst_md = memory::desc({BATCH, OC, OH, OW}, memory::data_type::f32, memory::format_tag::any);
        // create a convolution descriptor & primitive descriptor
        auto conv_d  = convolution_forward::desc(prop_kind::forward_training,
                        algorithm::convolution_auto, conv_src_md, conv_wei_md, 
                        conv_bia_md, conv_dst_md, strides, padding, padding);
        auto conv_pd = convolution_forward::primitive_desc(conv_d, eng);

        auto conv_src_m = memory(conv_pd.src_desc(), eng);
        auto conv_wei_m = memory(conv_pd.weights_desc(), eng);
        auto conv_user_bia_m = memory(conv_pd.bias_desc(), eng);
        auto conv_dst_m = memory(conv_pd.dst_desc(), eng);

        float *data = static_cast<float *>(conv_src_m.get_data_handle());
        for (size_t i = 0; i < BATCH*IC*IH*IW; ++i)
            data[i] = (float)i;//sinf((float)i);
        data = static_cast<float *>(conv_wei_m.get_data_handle());
        for (size_t i = 0; i < OC*IC*KH*KW; ++i)
            data[i] = (float)i;//sinf((float)i);
        data = static_cast<float *>(conv_user_bia_m.get_data_handle());
        for (size_t i = 0; i < conv_user_bia_m.get_desc().get_size(); ++i)
            data[i] = (float)i;//sinf((float)i);

        auto conv = convolution_forward(conv_pd);

        for (int i = 0; i < ExecCount; ++i) {
            auto t3 = high_resolution_clock::now();
            conv.execute(s, {{DNNL_ARG_SRC,    conv_src_m},
                            {DNNL_ARG_WEIGHTS, conv_wei_m},
                            {DNNL_ARG_BIAS,    conv_user_bia_m},
                            {DNNL_ARG_DST,     conv_dst_m}});
            auto t4 = high_resolution_clock::now();
            seconds3 += duration_cast<duration<float>>(t4-t3).count();
            // printf("conv time = %f\n", duration_cast<duration<float>>(t4-t3).count());
        }
        printf("execute forward conv time = %lf\n",      seconds3/ExecCount);
        if (bestSecond == 0 || bestSecond > seconds3/ExecCount) {
            bestSecond = seconds3/ExecCount;
            bestNblock = nblock + 1;
        }
        seconds3 = 0;
    }
    printf("best time comsumeing=%lf, best Nblock=%d\n", bestSecond, bestNblock);

    // ///check answer compare to direct convolution
    // auto dir_conv_d  = convolution_forward::desc(prop_kind::forward_training,
    //                 algorithm::convolution_direct, conv_src_md, conv_wei_md, 
    //                 conv_bia_md, conv_dst_md, strides, padding, padding);
    // auto dir_conv_pd = convolution_forward::primitive_desc(dir_conv_d, eng);
    // auto dir_conv_src_m = memory(dir_conv_pd.src_desc(), eng);
    // auto dir_conv_wei_m = memory(dir_conv_pd.weights_desc(), eng);
    // auto dir_conv_user_bia_m = memory(dir_conv_pd.bias_desc(), eng);
    // auto dir_conv_dst_m = memory(dir_conv_pd.dst_desc(), eng);
    // auto dir_conv = convolution_forward(dir_conv_pd);
    // data = static_cast<float *>(dir_conv_src_m.get_data_handle());
    // for (size_t i = 0; i < BATCH*IC*IH*IW; ++i)
    //     data[i] = (float)i;
    // data = static_cast<float *>(dir_conv_wei_m.get_data_handle());
    // for (size_t i = 0; i < OC*IC*KH*KW; ++i)
    //     data[i] = (float)i;
    // data = static_cast<float *>(conv_user_bia_m.get_data_handle());
    // for (size_t i = 0; i < OC; ++i)
    //     data[i] = (float)i;
    // dir_conv.execute(s, {{DNNL_ARG_SRC,    dir_conv_src_m},
    //                 {DNNL_ARG_WEIGHTS, dir_conv_wei_m},
    //                 {DNNL_ARG_BIAS,    dir_conv_user_bia_m},
    //                 {DNNL_ARG_DST,     dir_conv_dst_m}});
    // data = static_cast<float *>(conv_dst_m.get_data_handle());
    // auto data2 = static_cast<float *>(dir_conv_dst_m.get_data_handle());
    // int numOfOut = 0;
    // for (int i = 0; i < BATCH*OC*OH*OW; ++i) {
    //    //assert(data2[i] >= data[i]*0.8 && data2[i] <= data[i]*1.2);
    //    if (data2[i] < data[i]*0.999 || data2[i] > data[i]*1.001) numOfOut ++;
    // }
    // printf("out of range : %d\n", numOfOut);
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv, 1);
    handle_example_errors(performance_profiling, engine_kind, argc, argv);
}
