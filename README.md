# MLKL

- MLKL is a machine learning framework written in [kl](http://fabricengine.com/) used to automaticaally classify images. 
- It supports two differents frameworks 
	* [MkSVM : Support vector machine](#MkSVM-framework)
	* [MkCNN : Deep learning](#MkCNN-framework)
- It purpose is to provide efficient parallel implementations that can run both on CPU and GPU (AMD or CUDA). At the moment, only CPU parallelism is supported but GPU will come soon.
- For now, the implementation works with the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset only. In the futur, more tests will be performed with the [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. Furthemore, code will be released soon to convert images of any format to the right format for the network.



## MkSVM framework
- MkSVM is inspired of [Accord.net](http://accord-framework.net/)
- It's still under developement, not fully tested
 



## MkCNN framework
- MkCNN is a KL implementation of a convolutional neural network.
- MkCNN is inspired of both [tiny-cnn](https://github.com/nyanp/tiny-cnn/wiki) and [Convnet](https://code.google.com/p/cuda-convnet/).

#### Features
- Layers : Fully-connected, Dropout, Convolutional, Pooling (average and max)
- Neurons : TanH, Sigmoid, Softmax, Rectified linear, Identity
- Loss functions : Cross entropy, Mean squared error
- Optimization : Stochastic gradient, Stochastic levenberg marquardt, AdaGrad, RmsProp


#### Building
* Requires [FabricEngine](http://fabricengine.com/get-fabric/), [scons](http://www.scons.org/) and a C++ compiler.
* Configure the environment from config/environment.bat (or .sh) file and set it
* Compile the C++ extension using scons (to read MNIST data) 
	* cd core/exts/MNIST 
	* scons 

#### Sample project
* Configure the network if needed
* Launch the sample project
 