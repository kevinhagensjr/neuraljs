const fs = require('fs');
const NeuralNetwork = require('./../neuraljs');

/*
const trainingfile = fs.readFileSync('./heart_train.json');
const trainingData = JSON.parse(trainingfile);
*/
const nn = new NeuralNetwork({
    inputs : 13,
    outputs : 1,
    layers : [13]
});

nn.train(trainingData);

const output = nn.predict([0.3333333333333333,"1","0",0.4528301886792453,0.4178082191780822,"0","0",0.5801526717557252,"1","0","1","3","3"]);
console.log('prediction: ',output);