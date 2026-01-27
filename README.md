### This is a neural network written in pure C.
The scope of this repo was for me to learn neural networks from the inside and particularly using vectorized gradient descent and backpropagation.

The network is flexible in terms of layers and number of layers. Acivation functions and both CE & MSE loss allow for both classification and regression. Weights can be initialized randomly, with Xavier or with He initialization. Matrices and matrix operations are implemented using flat arrays.

For the MNIST Handwritten Digits, the layer structure: {784, 256, 128, 10}, alpha = 0.1, epochs = 500 over 60k images (full batch gradient descent): The model achieved 93.18% accuracy on 10k test images. </br>
With almost the same structure and hyperparameters as above, the MNIST Fashion Dataset acheived 82.75% accuracy on 10k test images. A larger hidden layer structure will likely lead to much better performance due to higher complexity of images despite them being the same size. 

To use this code, clone the repository using <br/>
`git clone https://github.com/Enerhim/neuralnetworkc/`

To build: </br>
`gcc -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fomit-frame-pointer -finline-functions -mavx2 -mfma src/math.c src/ml.c src/mnist.c main.c -o build.out -lm && ./build.out` </br>  </br>
(Makefile coming soon. This was my first proper C project hence my code is not the most optimal. My primary focus was to learn neural networks on a deeper level and gain insight into how they are implemented. I also just wanted to learn about memory management and such in general. The model is really slow at training: 6-7~ seconds / epoch which I would like to later on improve with CUDA) 

FURTHER SCOPE: Generalizing as a computational graph and autodifferentiation, testing regression, other optimizers, GPU optimization 
