from uwnet import *

# filters * c * size^2 * im.w * im.h / stride^2
# layer 1: 32*32*3*8*9=221184
# layer 2: 16*16*8*16*9=294912
# layer 3: 8*8*16*32*9=294912
# layer 4: 4*4*32*64*9=294912
# layer 5: 256*10=2560
# convnet total: 1,108,480
# connected total: 1,082,624

def connected_net():
    l = [   make_connected_layer(3072, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 128),
            make_activation_layer(RELU),
            make_connected_layer(128, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = connected_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# convnet training accuracy:   0.6756600141525269
# convnet test accuracy:       0.6328999996185303
#
# connected training accuracy: 0.5573800206184387
# connected test accuracy:     0.5051000118255615
#
# Clearly, our convnet performed better because with equal number of operations, the convnet is able to extract
# more useful features through the spatial assumption made in the convolution layers. Although, in theory, fully
# connected networks are just as powerful as convnet, without this spatial assumption, my connected network likely
# wasn't able to converge to a lower local minimum than what convnet was able to.

