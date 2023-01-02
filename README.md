# NeuralJS
Simple Neural Network for NodeJS inspired by Brain.js using gradient descent and momentum

## Installation

```npm -g install neuraljs```


## Initialize Network Architecture

Setup your network architecture, some fields already have defaults.

```
const NeuralNetwork = require('neuraljs');

const nn = new NeuralNetwork({
    name : 'heart_network',
    inputs : 13,
    outputs : 1,
    layers : [13], //an array with the size of each layer
    learningRate : .3,
    momentum : .1,
    epochs : 20000, //number of training iterations
    errorThreshold : .005, // goal error rate for network
    activation : 'sigmoid' | 'tanh' | 'relu',
    type : 'regression'
});

```

## Train Network

The model takes an json object with input/output data. 

### Run the network through training

```
//example training data with xor gate

const nn = new NeuralNetwork({
    name : 'xor_network', //name of the network
    inputs : 2, //two binary inputs
    outputs : 1, //output of the gate prediction
    layers : [3], // one hidden layer with 2 neurons
});

nn.train({
    "input" : [[0, 0],[0, 1],[1, 0],[1, 1]],
    "output" : [[0],[1],[1],[0]]
});
```

## Predict 

The model will make a prediction for given input. (Should train first, or import json model from previous training)

```
const nn = new NeuralNetwork({
    name : 'xor_network', //name of the network
    inputs : 2, //two binary inputs
    outputs : 1, //output of the gate prediction
    layers : [3], // one hidden layer with 2 neurons
});

const output = nn.predict([0,1]);
console.log('prediction: ',output);
```

## Export 

You can export a json file of the model after training and re-use later using import. (layers,weights, biases)

```
...traing and testing logic

const path = './backup/model.json';
nn.export(path);
```

## Import 

You can import the model you have already trained (layers,weights,biases)

```
const path = './backup/model.json';

const nn = new NeuralNetwork();
nn.import(path);
```

