### This is a neural network written in pure C.
The scope of this neural network is to classify the **MNIST handwritten digits dataset** with a good level of accuracy. 

The neural network is implemented using two hidden layers with the ReLu activation and the Softmax activation for final probabilities.
I have also implemented additional activation functions and attempted to generalize the network to the best of my ability so that it could be used elsewhere. 

To use this code, clone the repository using <br/>
`git clone https://github.com/Enerhim/neuralnetworkc/`

To build: </br>
`gcc -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto \                                                                                                ─╯
    -fomit-frame-pointer -finline-functions -mavx2 -mfma \
    src/math.c src/ml.c src/mnist.c main.c -o build.out -lm && ./build.out` </br>  </br>
(Makefile coming soon. This was my first proper C project hence my code is not the most optimal. My primary focus was to learn neural networks on a deeper level and gain insight into how they are implemented. I also just wanted to learn about memory management and such in general) 

