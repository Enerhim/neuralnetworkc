### This is a neural network written in pure C.
The scope of this neural network is to classify the **MNIST handwritten digits dataset** with a good level of accuracy. 

The neural network is implemented using two hidden layers with the ReLu activation and the Softmax activation for final probabilities.
I have also implemented additional activation functions and attempted to generalize the network to the best of my ability so that it could be used elsewhere. 

To use this code, clone the repository using <br/>
`git clone https://github.com/Enerhim/neuralnetworkc/`

To build: </br>
`gcc src/math.c src/ml.c src/mnist.c main.c -o build.out -lm -march=native -ffast-math; ./build.out` </br>
(I will make a MakeFile soon...) </br>
Also forgive me for the non optimal C code, my focus was to get a working neural network written from scratch and not to master systems programming.
