/*
    BitNetMCU inference functions
    @cpldcpu April 2024

    Performs inference on fully connected layer on a very resource constrained MCU.
    1,2,4 bit weights are supported.

*/

#include <stdint.h>

#include <stdio.h>
#include "BitNetMCU_inference.h"

/**
 * @brief Applies a ReLU activation function to an array of integers and normalizes the result to 8-bit integers.
 *
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_input) {
    int32_t max_val = -INT32_MAX;
    int32_t max_pos = 255;
    uint32_t scale;
    uint32_t shift;
    int32_t rounding;
    int32_t tmp;

    // Find the maximum value in the input array
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_pos = i;
        }
    }

    // Normalization
    // Dynamic shift according to max value in the input array
    scale=max_val>>7;  // define max range, all bits above 7 will be shifted down
    shift=0;

    while (scale>0) {
        shift++;
        scale>>=1;
    }

    // impact of rounding is almost negligible (+0.03% in eval accuracy)
    // But rounding affects mismatch to python inference engine
    rounding   = (1 << (shift))>>1;

    // Apply ReLU activation and normalize to 8-bit
    for (uint32_t i = 0; i < n_input; i++) {
        // Apply ReLU activation
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            tmp=(input[i] + rounding) >> shift;

            // clipping needed to catch overflow from rounding
            if (tmp > 127) {
                output[i] = 127;
            } else {
            output[i] = tmp;
            }
        }
    }
    return max_pos;
}


/**
 * @brief Processes a maxpooling layer.
 * 
 * @param input Pointer to the input array of 8-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

void Maxp(int8_t *input, int8_t *output, uint32_t n_inpu, uint32_t in_channels, 
    uint32_t out_channels, uint32_t incoming_x, uint32_t incoming_y) {

    uint32_t outgoin_x = incoming_x / 2;
    uint32_t outgoin_y = incoming_y / 2;
    
    for (uint32_t c = 0; c < in_channels; c++) {
        for (uint32_t oy = 0; oy < outgoin_y; oy++) {
            for (uint32_t ox = 0; ox < outgoin_x; ox++) {
                int8_t max_val = -128;

                for (uint32_t ky = 0; ky < 2; ky++) {
                    for (uint32_t kx = 0; kx < 2; kx++) {
                        uint32_t ix = ox * 2 + kx;
                        uint32_t iy = oy * 2 + ky;
                        if (ix < incoming_x && iy < incoming_y) {
                            int8_t val = input[c * incoming_y * incoming_x + iy * incoming_x + ix];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }

                output[c * outgoin_y * outgoin_x + oy * outgoin_x + ox] = max_val;
            }
        }
    }
}


/**
 * @brief Processes a fully connected layer in a neural network.
 *
 * This function processes a fully connected layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param n_input The number of input neurons.
 * @param n_output The number of output neurons.
 * @param output Pointer to the output array where the result of the layer is stored.
 */

void processfclayer( int8_t *activations,  const uint32_t *weights, int32_t bits_per_weight, uint32_t n_input, uint32_t n_output, int32_t *output)
{
   const uint32_t *weightidx = weights;

    for (uint32_t i = 0; i < n_output; i++) {
        int8_t *activations_idx = activations;
        int32_t sum = 0;

        if (bits_per_weight == 1) {
            for (uint32_t k = 0; k < n_input; k+=32) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 32; j++) {
                    int32_t in=*activations_idx++;
                    sum += (weightChunk & 0x80000000) ? in : -in;  // Note that sign is flipped for Binary quant (bit set equals positive)
                    weightChunk <<= 1;
                }
            }
        } else if (bits_per_weight == 2 ) {
            for (uint32_t k = 0; k < n_input; k+=16) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 16; j++) {
                    int32_t in=*activations_idx++;
                    int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                    sum += tmpsum;                                  // sign*in*1
                    if (weightChunk & 0x40000000) sum += tmpsum<<1; // sign*in*2
                    weightChunk <<= 2;
                }
            }
        // Multiplier-less inference for RV32EC
#if defined(__riscv) && !defined(__riscv_mul)
        } else if (bits_per_weight == 4 ) {
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    if (in != 0) { // Skip zero activations to speed up inference in layers after first layer
                        int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                        sum += tmpsum;                                  // sign*in*1
                        if (weightChunk & 0x10000000) sum += tmpsum<<1; // sign*in*2
                        if (weightChunk & 0x20000000) sum += tmpsum<<2; // sign*in*4
                        if (weightChunk & 0x40000000) sum += tmpsum<<3; // sign*in*8
                    }
                    weightChunk <<= 4;
                }
            }
#else
        } else if (bits_per_weight == 4 ) {
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    if (in != 0) { // Skip zero activations to speed up inference in layers after first layer
                        int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                        sum += tmpsum * ((weightChunk>>(32-4))&7);                                  // sign*in*1
                    }
                    weightChunk <<= 4;
                }
            }
        } else if (bits_per_weight == 8 ) {   // 8 bit twos-complement
            for (uint32_t k = 0; k < n_input; k+=4) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 4; j++) {
                    int32_t in=*activations_idx++;
                    int32_t weight = (weightChunk) >> (32-8); // extend sign, cut off lower bits
                    sum += in*weight;
                    weightChunk <<= 8;
                }
            }
        } else if (bits_per_weight == 8 + 4 ) {   // 4 bit twos-complement
            for (uint32_t k = 0; k < n_input; k+=8) {
                int32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    int32_t weight = (weightChunk) >> (32-4); // extend sign, cut off lower bits
                    sum += in*weight;
                    weightChunk <<= 4;
                }
            }
        } else if (bits_per_weight == 8 + 8 ) {   // 8 bit twos-complement
            for (uint32_t k = 0; k < n_input; k+=4) {
                int32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 4; j++) {
                    int32_t in=*activations_idx++;
                    int32_t weight = (weightChunk) >> (32-8); // extend sign, cut off lower bits
                    sum += in*weight;
                    weightChunk <<= 8;
                }
            }
#endif
        }  else if (bits_per_weight == 16 + 4 ) {  // 4 bit shift
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    int32_t tmpsum;

                    tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                    sum += tmpsum << ((weightChunk >> 28) & 7); // sign*in*2^log
                    weightChunk <<= 4;
                }
            }
        }   // else printf("Error: unsupported weight bit width %d\n", bits_per_weight);

        output[i] = sum;
        // printf("%d,", output[i]);
    }
}



void precessfc_I2_S(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t incoming_weights, int32_t outgoing_weights, int32_t *output) {
    
    memset(output, 0, sizeof(int32_t) * outgoing_weights);
    const uint32_t *weightidx = weights;

    for (int i = 0; i < outgoing_weights; ++i) {
        int8_t *activations_idx = activations;
        int32_t sum = 0;

        for (int j = 0; j < incoming_weights; ++j) {

            int8_t act = *activations_idx++;

            int bit_pos = (i * incoming_weights + j) * bits_per_weight;
            int word_idx = bit_pos / 32;
            int bit_off  = bit_pos % 32;

            uint32_t wbits = (weights[word_idx] >> bit_off) & 0x3; 

            // encoding: 00→-1, 01→0, 10→+1

            int32_t weight_multiplier = (int32_t)wbits - 1; // {-1, 0, 1}
            // sum += (int32_t)act * weight_multiplier;
                               

            int8_t delta = ( -((int8_t)(wbits == 0x2)) & act ) | ( -((int8_t)(wbits == 0x0)) & (-act) );
            sum += delta;
        }

        output[i] = sum;
    }
}


/**
 * @brief Processes a conv2d layer in a neural network.
 *
 * This function processes a conv2D layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.

 * @param output Pointer to the output array where the result of the layer is stored.
 */
void processcvlayer_I2_S(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t in_channels, uint32_t out_channels, 
                         uint32_t incoming_x, uint32_t incoming_y, uint32_t outgoing_x, uint32_t outgoing_y, uint32_t stride, uint32_t padding, int32_t *output) {
    
    const int kernel_size = 3;
    // memset(output, 0, sizeof(int32_t) * in_channels * outgoing_y * outgoing_x);
                            

    for (int n = 0; n < out_channels; ++n) {
        for (int oy = 0; oy < outgoing_y; ++oy) {
            for (int ox = 0; ox < outgoing_x; ++ox) {
                int32_t sum = 0; // assume no bias
                for (int c = 0; c < in_channels; ++c) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int ix = ox * stride - padding + kx;
                            int iy = oy * stride - padding + ky;

                            if (ix >= 0 && ix < incoming_x && iy >= 0 && iy < incoming_y) {
                                int8_t act = activations[c * incoming_y * incoming_x + iy * incoming_x + ix];

                                int bit_pos = (n * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + ky * kernel_size + kx) * bits_per_weight;
                                int word_idx = bit_pos / 32;
                                int bit_off  = bit_pos % 32;

                                uint32_t wbits = (weights[word_idx] >> bit_off) & 0x3;  // 2-bit
                                // encoding: 00→-1, 01→0, 10→+1
                                // int32_t weight_multiplier = (int32_t)wbits - 1; // {-1, 0, 1}
                                // sum += (int32_t)act * weight_multiplier;
                                int8_t delta = ( -((int8_t)(wbits == 0x2)) & act ) | ( -((int8_t)(wbits == 0x0)) & (-act) );
                                sum += delta;
                                // sum += act * weight_multiplier; // this is the same as above, but more readable
                            }
                        }
                    }
                }
                output[n * outgoing_y * outgoing_x + oy * outgoing_x + ox] = sum;
            }
        }
    }
}

/**
 * @brief Processes a conv2d layer in a neural network.
 *
 * This function processes a conv2D layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.

 * @param output Pointer to the output array where the result of the layer is stored.
 */
void processcvlayer_8bits(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t in_channels, uint32_t out_channels, 
                         uint32_t incoming_x, uint32_t incoming_y, uint32_t outgoing_x, uint32_t outgoing_y, uint32_t stride, uint32_t padding, int32_t *output) {
    
    const int kernel_size = 3;
    // memset(output, 0, sizeof(int32_t) * in_channels * outgoing_y * outgoing_x);
                            

    for (int n = 0; n < out_channels; ++n) {
        for (int oy = 0; oy < outgoing_y; ++oy) {
            for (int ox = 0; ox < outgoing_x; ++ox) {
                int32_t sum = 0; // assume no bias
                for (int c = 0; c < in_channels; ++c) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int ix = ox * stride - padding + kx;
                            int iy = oy * stride - padding + ky;

                            if (ix >= 0 && ix < incoming_x && iy >= 0 && iy < incoming_y) {
                                int8_t act = activations[c * incoming_y * incoming_x + iy * incoming_x + ix];
                                int bit_pos = (n * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + ky * kernel_size + kx) * bits_per_weight;
                                int word_idx = bit_pos / 32;
                                int bit_off  = bit_pos % 32;
                                uint32_t wbits = (weights[word_idx] >> bit_off) & 0xFF;  // 8-bit
                                // encoding: 0→-127, 1→-126, ..., 127→+127
                                if (wbits < 128) {
                                    sum -= act * (127 - wbits); // 0 to 127
                                } else {
                                    sum += act * (wbits - 127); // 128 to 255
                                }
                             }
                        }
                    }
                }
                output[n * outgoing_y * outgoing_x + oy * outgoing_x + ox] = sum;
            }
        }
    }
                                
}

void precessfc_8bits(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t incoming_weights, int32_t outgoing_weights, int32_t *output) {
    
    memset(output, 0, sizeof(int32_t) * outgoing_weights);
    const uint32_t *weightidx = weights;

    for (int i = 0; i < outgoing_weights; ++i) {
        int8_t *activations_idx = activations;
        int32_t sum = 0;

        for (int j = 0; j < incoming_weights; ++j) {

            int8_t act = *activations_idx++;

            int bit_pos = (i * incoming_weights + j) * bits_per_weight;
            int word_idx = bit_pos / 32;
            int bit_off  = bit_pos % 32;   
            uint32_t wbits = (weights[word_idx] >> bit_off) & 0xFF;  // 8-bit
            // encoding: 0→-127, 1→-126, ..., 127→+127
            if (wbits < 128) {
                sum -= act * (127 - wbits); // 0 to 127
            } else {
                sum += act * (wbits - 127); // 128 to 255
            }
        }

        output[i] = sum;
    }
}