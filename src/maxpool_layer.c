#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int i, j, c, k;
    for(i = 0; i < in.rows; ++i){
        //each image
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        for(c = 0; c < l.channels; ++c){
            for (j = 0; j < x.cols; ++j) {
                //each spatial loaction in each channel
                int max = 0;
                for (k = 1; k < l.size * l.size; ++k) {
                    if (x.data[c*x.cols*l.size*l.size + max*x.cols + j] < x.data[c*x.cols*l.size*l.size + k*x.cols + j]) {
                        max = k;
                    }
                }
                out.data[i*out.cols + c*x.cols + j] = x.data[c*x.cols*l.size*l.size + max*x.cols + j];
            }
        }
        free_matrix(x);
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int i, j, c, k;
    for(i = 0; i < in.rows; ++i){
        //each image
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        for(c = 0; c < l.channels; ++c){
            for (j = 0; j < x.cols; ++j) {
                //each spatial loaction in each channel
                int max = 0;
                for (k = 1; k < l.size * l.size; ++k) {
                    //find max
                    if (x.data[c*x.cols*l.size*l.size + max*x.cols + j] < x.data[c*x.cols*l.size*l.size + k*x.cols + j]) {
                        x.data[c*x.cols*l.size*l.size + max*x.cols + j] = 0;
                        max = k;
                    } else {
                        x.data[c*x.cols*l.size*l.size + k*x.cols + j] = 0;
                    }
                }

                int top_left_row = j / outw * l.stride - (l.size-1)/2;
                int top_left_col = j % outw * l.stride - (l.size-1)/2;
                int max_row = top_left_row + max / l.size;
                int max_col = top_left_col + max % l.size;
                
                dx.data[i*dx.cols + c*l.width*l.height + max_row*l.width + max_col] += dy.data[i*dy.cols + c*x.cols + j];
            }
        }
        free_matrix(x);
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

