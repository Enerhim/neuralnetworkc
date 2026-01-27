### This is a neural network written in pure C.
The scope of this neural network is to classify the **MNIST handwritten digits dataset** with accuracy 90%+ (and also to learn the fundamentals of neural networks well). 

The network is flexible in terms of layers and number of layers. Acivation functions and both CE & MSE loss allow for both classification and regression. Weights can be initialized randomly, with Xavier or with He initialization. Matrices and matrix operations are implemented using flat arrays.

With the layers structure: {784, 256, 128, 10}, alpha = 0.1, epochs = 500 over 60k images (full batch gradient descent): The model achieved 93.18% accuracy on 10k test images.

To use this code, clone the repository using <br/>
`git clone https://github.com/Enerhim/neuralnetworkc/`

To build: </br>
`gcc -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fomit-frame-pointer -finline-functions -mavx2 -mfma src/math.c src/ml.c src/mnist.c main.c -o build.out -lm && ./build.out` </br>  </br>
(Makefile coming soon. This was my first proper C project hence my code is not the most optimal. My primary focus was to learn neural networks on a deeper level and gain insight into how they are implemented. I also just wanted to learn about memory management and such in general. The model is really slow at training: 6-7~ seconds / epoch which I would like to later on improve with CUDA) 

FURTHER SCOPE: Generalizing as a computational graph and autodifferentiation, testing regression, other optimizers, GPU optimization 
