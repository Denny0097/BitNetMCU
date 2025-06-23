#ifndef BITNETMCU_INFERENCE_H
#define BITNETMCU_INFERENCE_H

#include <stdint.h>

/**
 * @brief Applies a ReLU activation function to an array of integers and normalizes the result to 8-bit integers.
 * 
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_inpu);

/**
 * @brief Processes a maxpooling layer.
 * 
 * @param input Pointer to the input array of 8-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_inpu The number of elements in the input array.
 * @param in_channels The number of input channels.
 * @param out_channels The number of output channels.
 * @param incoming_x The width of the input feature map.
 * @param incoming_y The height of the input feature map.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

void Maxp(int8_t *input, int8_t *output, uint32_t n_inpu, uint32_t in_channels, 
    uint32_t out_channels, uint32_t incoming_x, uint32_t incoming_y);


/**
 * @brief Processes a fully connected layer in a neural network.
 *
 * This function processes a fully connected layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param incoming_weights The number of input neurons.
 * @param outgoing_weights The number of output neurons.
 * @param output Pointer to the output array where the result of the layer is stored.
 */
void processfclayer(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t incoming_weights, uint32_t outgoing_weights, int32_t *output);

/**
 * @brief Processes a fully connected layer in a neural network with 2-bit weights.
 *
 * This function processes a fully connected layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param incoming_weights The number of input neurons.
 * @param outgoing_weights The number of output neurons.
 * @param output Pointer to the output array where the result of the layer is stored.
 */
void precessfc_I2_S(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t incoming_weights, int32_t outgoing_weights, int32_t *output);

/**
 * @brief Processes a conv2D layer in a neural network.
 *
 * This function processes a conv2D layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param in_channels The number of input channels.
 * @param out_channels The number of output channels.
 * @param incoming_x The width of the input feature map.
 * @param incoming_y The height of the input feature map.
 * @param outgoing_x The width of the output feature map.
 * @param outgoing_y The height of the output feature map.
 * @param stride The stride of the convolution.
 * @param padding The amount of zero-padding applied to the input feature map.
 * @param output Pointer to the output array where the result of the layer is stored.
 */
void processcvlayer_I2_S(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t in_channels, uint32_t out_channels, 
                         uint32_t incoming_x, uint32_t incoming_y, uint32_t outgoing_x, uint32_t outgoing_y, uint32_t stride, uint32_t padding, int32_t *output);

#endif // BITNETMCU_INFERENCE_H


