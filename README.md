### This is a neural network written in pure C.
This was a learning project of mine for learning the implementation of nns using vectorized gradient descent and backpropagation.

For the *MNIST Handwritten Digits*, the layer structure: {784, 256, 128, 10}, alpha = 0.1 w/ decay, epochs = 1000 over 60k images (full batch gradient descent): The model achieved 93.20% accuracy on 10k test images. </br>
With almost the same structure and hyperparameters as above, the *MNIST Fashion Dataset* acheived 84.15% accuracy on 10k test images. A larger hidden layer structure will likely lead to much better performance due to higher complexity of images despite them being the same size. 

Training was at roughly 6-7 seconds / epoch in this CPU implementation

## Build

To use this code, clone the repository using <br/>
`git clone https://github.com/Enerhim/neuralnetworkc/`

To build: </br>
`gcc -O3 -march=native -mtune=native src/math.c src/ml.c src/mnist.c src/pcg.c main.c -o build.out -lm && ./build.out` </br>  </br>
(Makefile coming soon)

FURTHER SCOPE: Generalizing as a computational graph and autodifferentiation, testing regression, other optimizers, GPU optimization, arena allocation
