# MLKL
A machine learning framework in KL

========

MLKL is a machine learning framework writtin in [kl](http://fabricengine.com/). It supports two differents machine learning-frameworks :
* [Support vector machine MkSVM](#MkSVM-framework)
* [Depp learning MkCNN](#MkCNN-framework)


## MkSVM framework
- 
- It's still under developement
 
## MkCNN framework
-MLKL has a KL implementation of [tiny-cnn](https://github.com/nyanp/tiny-cnn/wiki), a convoluional neural network) implemented in C++11.
-It purpose is to provide a more efficient that can run both on CPU and GPU.

### supported networks
#### layer-types
* fully-connected layer
* fully-connected layer with dropout
* convolutional layer
* average pooling layer
* max-pooling layer

#### activation functions
* tanh
* sigmoid
* softmax
* rectified linear(relu)
* leaky relu
* identity

#### loss functions
* cross-entropy
* mean-squared-error

#### optimization algorithm
* stochastic gradient descent (with/without L2 normalization and momentum)
* stochastic gradient levenberg marquardt
* adagrad
* rmsprop

### dependencies
#### requirements
FabricEngine, scons and a C++ compiler
 