#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h> 

#include "BitNetMCU_model.h"
#include "BitNetMCU_inference.c"
#include "BitNetMCU_MNIST_test_data.h"
#include "BitNetMCU_CIFAR10_test_data.h"

/**
 * Performs inference on the MNIST dataset using the BitNetMCU model.
 *
 * @param input The input data for the inference, a 16x16 array of int8_t.
 * @return The predicted digit.
 */

// uint32_t BitMnistInference(int8_t*);
uint32_t BitMnistInference_VGG8_I2_S(int8_t*);

void pad_input_16x16_to_32x32(const int8_t* src, int8_t* dst) {
    memset(dst, 0xEC, 32 * 32); 
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            dst[(y + 8) * 32 + (x + 8)] = src[y * 16 + x];  
        }
    }
}

void main(void) {
    uint32_t output[10];
    uint8_t predicted_label;
    int8_t padded_input[1024];  // 32x32
    int8_t *input_data = input_data_0; // 16x16

    // for 32x32 input
    // pad_input_16x16_to_32x32(input_data, padded_input);
    // predicted_label = BitMnistInference_VGG8_I2_S(padded_input);
    // printf("label: %d predicted: %d\n", label_6, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_0);
    // printf("label: %d predicted: %d\n", label_0, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_1);
    // printf("label: %d predicted: %d\n", label_1, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_2);
    // printf("label: %d predicted: %d\n", label_2, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_3);
    // printf("label: %d predicted: %d\n", label_3, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_4);
    // printf("label: %d predicted: %d\n", label_4, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_5);
    // printf("label: %d predicted: %d\n", label_5, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_6);
    // printf("label: %d predicted: %d\n", label_6, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_7);
    // printf("label: %d predicted: %d\n", label_7, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_8);
    // printf("label: %d predicted: %d\n", label_8, predicted_label);
    // predicted_label = BitMnistInference_VGG8_I2_S(input_data_9);
    // printf("label: %d predicted: %d\n", label_9, predicted_label);
    // for cifar10
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_1);
    printf("label: %d predicted: %d\n", cifar10_label_1, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_1);
    printf("label: %d predicted: %d\n", cifar10_label_1, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_2);
    printf("label: %d predicted: %d\n", cifar10_label_2, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_3);
    printf("label: %d predicted: %d\n", cifar10_label_3, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_4);
    printf("label: %d predicted: %d\n", cifar10_label_4, predicted_label); 
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_5);
    printf("label: %d predicted: %d\n", cifar10_label_5, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_6);
    printf("label: %d predicted: %d\n", cifar10_label_6, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_7);
    printf("label: %d predicted: %d\n", cifar10_label_7, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_8);
    printf("label: %d predicted: %d\n", cifar10_label_8, predicted_label);
    predicted_label = BitMnistInference_VGG8_I2_S(cifar10_data_9);
    printf("label: %d predicted: %d\n", cifar10_label_9, predicted_label);
    
}


/**
 * @brief Performs inference on the VGG model.
 *
 * @param input The input data for the inference.
 * @return The result of the inference.
 */
uint32_t BitMnistInference_VGG8_I2_S(int8_t *input) {

    uint32_t L2_outgoing_shape = L2_outgoing_x * L2_outgoing_y * L2_out_channels;
    int32_t L2_out[L2_outgoing_shape];
    int8_t L3_out[L2_outgoing_shape];
    int8_t L4_out[L2_outgoing_shape/4];

    uint32_t L5_outgoing_shape = L5_outgoing_x * L5_outgoing_y * L5_out_channels;
    int32_t L5_out[L5_outgoing_shape];
    int8_t L6_out[L5_outgoing_shape];
    int8_t L7_out[L5_outgoing_shape/4];

    uint32_t L8_outgoing_shape = L8_outgoing_x * L8_outgoing_y * L8_out_channels;
    int32_t L8_out[L8_outgoing_shape];
    int8_t L9_out[L8_outgoing_shape];

    uint32_t L10_outgoing_shape = L10_outgoing_x * L10_outgoing_y * L10_out_channels;
    int32_t L10_out[L10_outgoing_shape];
    int8_t L11_out[L10_outgoing_shape];

    uint32_t L12_outgoing_shape = L12_outgoing_x * L12_outgoing_y * L12_out_channels;
    int32_t L12_out[L12_outgoing_shape];
    int8_t L13_out[L12_outgoing_shape];
    int8_t L14_out[L12_outgoing_shape/4];

    int32_t fc1_out[L16_outgoing_weights];
    int8_t fc1relu_out[L16_outgoing_weights];

    int32_t fc2_out[L18_outgoing_weights];
    int8_t fc2relu_out[L18_outgoing_weights];

    int32_t fc3_out[L20_outgoing_weights];
    int8_t fc3relu_out[L20_outgoing_weights];

    clock_t start_time, end_time;
    double cpu_time_used;

    // L2 (Convolutional Layer)
    start_time = clock();
    processcvlayer_I2_S(input, L2_packed_weights, L2_bitperweight, L2_in_channels, L2_out_channels,
        L2_incoming_x, L2_incoming_y, L2_outgoing_x, L2_outgoing_y, L2_stride, L2_padding, L2_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; 
    printf("L2 (Conv) execution time: %.3f ms\n", cpu_time_used);

    // L3 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(L2_out, L3_out, L2_outgoing_shape);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L3 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L4 (Max Pooling)
    start_time = clock();
    Maxp(L3_out, L4_out, L2_outgoing_shape / 4, L2_out_channels, L2_out_channels,
        L2_outgoing_x, L2_outgoing_y);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L4 (Maxp) execution time: %.3f ms\n", cpu_time_used);

    // L5 (Convolutional Layer)
    start_time = clock();
    processcvlayer_I2_S(L4_out, L5_packed_weights, L5_bitperweight, L5_in_channels, L5_out_channels,
        L5_incoming_x, L5_incoming_y, L5_outgoing_x, L5_outgoing_y, L5_stride, L5_padding, L5_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L5 (Conv) execution time: %.3f ms\n", cpu_time_used);

    // L6 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(L5_out, L6_out, L5_outgoing_shape);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L6 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L7 (Max Pooling)
    start_time = clock();
    Maxp(L6_out, L7_out, L5_outgoing_shape / 4, L5_out_channels, L5_out_channels,
        L5_outgoing_x, L5_outgoing_y); 
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L7 (Maxp) execution time: %.3f ms\n", cpu_time_used);

    // L8 (Convolutional Layer)
    start_time = clock();
    processcvlayer_I2_S(L7_out, L8_packed_weights, L8_bitperweight, L8_in_channels, L8_out_channels,
        L8_incoming_x, L8_incoming_y, L8_outgoing_x, L8_outgoing_y, L8_stride, L8_padding, L8_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L8 (Conv) execution time: %.3f ms\n", cpu_time_used);

    // L9 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(L8_out, L9_out, L8_outgoing_shape);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L9 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L10 (Convolutional Layer)
    start_time = clock();
    processcvlayer_I2_S(L9_out, L10_packed_weights, L10_bitperweight, L10_in_channels, L10_out_channels,
        L10_incoming_x, L10_incoming_y, L10_outgoing_x, L10_outgoing_y, L10_stride, L10_padding, L10_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L10 (Conv) execution time: %.3f ms\n", cpu_time_used);

    // L11 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(L10_out, L11_out, L10_outgoing_shape);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L11 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L12 (Convolutional Layer)
    start_time = clock();
    processcvlayer_I2_S(L11_out, L12_packed_weights, L12_bitperweight, L12_in_channels, L12_out_channels,
        L12_incoming_x, L12_incoming_y, L12_outgoing_x, L12_outgoing_y, L12_stride, L12_padding, L12_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L12 (Conv) execution time: %.3f ms\n", cpu_time_used);

    // L13 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(L12_out, L13_out, L12_outgoing_shape);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L13 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L14 (Max Pooling)
    start_time = clock();
    Maxp(L13_out, L14_out, L12_outgoing_shape / 4, L12_out_channels, L12_out_channels,
        L12_outgoing_x, L12_outgoing_y); 
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L14 (Maxp) execution time: %.3f ms\n", cpu_time_used);



    // L16 (Fully Connected Layer)
    start_time = clock();
    precessfc_I2_S(L14_out, L16_weights, L16_bitperweight, L16_incoming_weights, L16_outgoing_weights, fc1_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L16 (FC) execution time: %.3f ms\n", cpu_time_used);

    // L17 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(fc1_out, fc1relu_out, L16_outgoing_weights);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L17 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L18 (Fully Connected Layer)
    start_time = clock();
    precessfc_I2_S(fc1relu_out, L18_weights, L18_bitperweight, L18_incoming_weights, L18_outgoing_weights, fc2_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L18 (FC) execution time: %.3f ms\n", cpu_time_used);

    // L19 (ReLU Normalization)
    start_time = clock();
    ReLUNorm(fc2_out, fc2relu_out, L18_outgoing_weights);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L19 (ReLUNorm) execution time: %.3f ms\n", cpu_time_used);

    // L20 (Fully Connected Layer - Output layer)
    start_time = clock();
    precessfc_I2_S(fc2relu_out, L20_weights, L20_bitperweight, L20_incoming_weights, L20_outgoing_weights, fc3_out);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    printf("L20 (FC) execution time: %.3f ms\n", cpu_time_used);

    int max_score = fc3_out[0];
    int predicted_label = 0;

    for (uint32_t i = 0; i < L20_outgoing_weights; i++) {
        if (fc3_out[i] > max_score) {
            max_score = fc3_out[i];
            predicted_label = i;
        }
    }

    return predicted_label;
}
