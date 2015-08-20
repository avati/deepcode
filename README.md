# deepcode

Experiment to generate erasure code data using deep learning.


The goal is to erasure code N nodes/disks of data into N+M nodes/disks, but tolerate M+delta failures (instead of the typical M failure toleration), and this is done by a deep inspection and "learning" the actual sequence of bits and bytes in order to generate a customized and specific scheme of encoding.

Running 'make test' in the cloned directory will build/execute and successfully train the neural network on a small sample random data.

Layer 1 - input data of N nodes
Layer 2 - encoded data of N+M nodes
Layer 3 - (output layer) is segmented into (M+N)Choose(N-delta) segments each of size N, each of which are fully connected with the corresponding subsets of Layer 2 in order to model all the possible scenarios having M+delta node failures

TODO:


- Still figuring out ways to make the activation function (ACT() in the code) of the hidden layer generate categorical numbers for the encoded representation.

- Implement a decoder to actually demonstrate the recovery "works" (the successful training proves this in theory)

- Add documentation and some diagrams soon to help visualize this network architecture.

- Make the learning rate adaptive to make training on larger data sets run in reasonable amount of time.
