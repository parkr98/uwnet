from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def batchnorm_conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .01
momentum = [.9, .7, .5, .3]
decay = [.05, .01, .005, .001]

#m = conv_net()
models = []
#print("training without batch norm...")
#train_image_classifier(m, train, batch, iters, rate, momentum, decay)
#print("done")
#print
for mom in momentum:
    for dec in decay:
        print("training with batch norm...")
        m = batchnorm_conv_net()
        models.append(m)
        train_image_classifier(m, train, batch, 100, 0.1, mom, dec)
        train_image_classifier(m, train, batch, 200, 0.05, mom, dec)
        train_image_classifier(m, train, batch, 200, 0.01, mom, dec)
        print("done")
        print

print("evaluating model...")
#print
#print("without batch norm")
#print("training accuracy: %f", accuracy_net(m, train))
#print("test accuracy:     %f", accuracy_net(m, test))
for m in models:
    print
    print("training accuracy: %f", accuracy_net(m, train))
    print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence?
# How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: 
# without batch norm
#   training accuracy: %f 0.4051400125026703
#   test accuracy:     %f 0.40380001068115234
# with batch norm
#   training accuracy: %f 0.5453799962997437
#   test accuracy:     %f 0.5418999791145325
# The model clearly performed much better with batch normalization; an accuracy boost of about 14%. With batch norm, the model
# converged significantly faster with the hyperparameters unmodified, reaching the same loss that the regular model achieved
# after only about 100 iterations.
