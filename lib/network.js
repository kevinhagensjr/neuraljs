const activation = require('./activation');
const derivative = require('./derivative');
const fs = require('fs');

class NeuralNetwork{
    constructor(opt={}){
        this.name = opt.name || 'network';
        this.inputSize = opt.inputs //size of input data
        this.outputSize = opt.outputs //size of output probabilities
        this.layerDepth = opt.layers || [3]; //size and count of each hidden layer
        this.learningRate = opt.learningRate || .3; //velocity of learning
        this.momentum = opt.momentum || .1; //helps learning move towards trend
        this.epochs = opt.epochs || 20000; //max number of training repetitions
        this.errorThreshold = opt.errorThreshold || 0.005; //error goal from training 
        this.func = opt.activation || 'sigmoid'; //activation function for each neuron
        this.type = opt.type || 'regression'; //type of nn, regression, classification, binary classification
        this.biases = []; //array of bias vectors for each layer
        this.weights = []; //array of weight vectors for each layer
        this.outputs = []; //array of outputs for each layer
        this.errors = []; //array of errors for each layer for training
        this.gradients = [] //array of gradients (rate of change of the errors)
        this.steps = [] //array to keep track of convergence

        this.layers = [this.inputSize,this.layerDepth,this.outputSize].flat(); //network in order, by size
    
        //initialize network, set weights and biases for each layer based on depth
        for(let layer=0; layer<this.layers.length; layer++){
            const layerSize = this.layers[layer];
            this.errors[layer] = new Array(layerSize).fill(0);
            this.outputs[layer] = new Array(layerSize).fill(0);
            this.gradients[layer] = new Array(layerSize).fill(0);
            
            if(layer > 0){   
                const previousLayerSize = this.layers[layer-1];
            
                //initialize biases/weights for each neuron
                this.biases[layer] = new Array(layerSize).fill(0).map((e) => this.randomize());
                this.weights[layer] = new Array(layerSize); 
                this.steps[layer] = new Array(layerSize);
                
                //initialize weights/connections for each neuron to previous layer
                for(let neuron=0; neuron < layerSize; neuron++){
                    this.weights[layer][neuron] = new Array(previousLayerSize).fill(0).map((e) => this.randomize());
                    this.steps[layer][neuron] = new Array(previousLayerSize).fill(0);
                }
            }
        }
    }

    predict(input){
        this.outputs[0] = input; //output of each layer becomes the input of next layer
        for(let layer=1; layer<this.layers.length; layer++){
            const layerSize = this.layers[layer];
            const previousLayerSize = this.layers[layer-1];

            for(let neuron=0;neuron<layerSize;neuron++){
                //get the bias for neuron and all weights connected to it
                const weights = this.weights[layer][neuron];

                let bias = this.biases[layer][neuron]; 
                //weight * input + bias
                for(let conn=0; conn<weights.length;conn++){
                    bias += input[conn] * weights[conn];
                }

                //save the neurons output
                this.outputs[layer][neuron] = activation[this.func](bias);
            }

            //save the results of the output
            input = this.outputs[layer];
        }

        return this.outputs[this.layers.length-1];
    }

    train(data){
        let error = 1;
        let iter = 0;
        //train network until error threshold/max iterations met
        for(let i=0; i<this.epochs && error > this.errorThreshold;i++){
            let totalError = 0;
            for(let x=0;x<data['input'].length;x++){     
                //forward propogate through the network
                const output = this.predict(data['input'][x]); 

                //backward propogate to get error and derivatives
                this.backPropogate(data['input'][x],output,data['output'][x]);

                //update weights
                this.gradientDescent();

                //get mean squared error
                let outputErrors = this.errors[this.layers.length-1];
                let errorSum = outputErrors.reduce((a,c) => a + Math.pow(c,2),0);
                totalError  +=  (errorSum / outputErrors.length);
            }

            error = totalError / data['input'].length;
            if(i % 1000 == 0){
                console.log('epoch: %i error rate: %f total error: %f',i,error,totalError);
            }
            iter++;
        }

        return {
            error: error,
            iterations: iter
        };
    }

    backPropogate(input,output,target){
        for(let layer = (this.layers.length-1); layer >= 0; layer--){
            for(let neuron=0;neuron<this.layers[layer];neuron++){
                let error = 0;

                //get the error from each neuron
                if(layer != this.layers.length-1){
                    for(let i=0; i<this.gradients[layer+1].length;i++){
                        error += this.gradients[layer+1][i] * this.weights[layer+1][i][neuron];
                    }
                }else{
                    error = target[neuron] - this.outputs[layer][neuron];
                } 

                //calculate the gradient for each neuron
                this.errors[layer][neuron] = error;
                this.gradients[layer][neuron] = error * derivative[this.func](this.outputs[layer][neuron]); 
            }
        }
    }

    gradientDescent(){
        for(let layer= 1; layer <= this.layers.length-1; layer++){
            let inputs = this.outputs[layer-1];
            //update all the weights for a neuron based on momentum and learning rate
            for(let neuron=0; neuron < this.layers[layer]; neuron++){  
                for(let conn=0;conn<inputs.length;conn++){

                    //calculate descent 
                    let step = this.steps[layer][neuron][conn];
                    step = (this.learningRate * this.gradients[layer][neuron] * inputs[conn]) + (this.momentum * step);
                    
                    //update new weight
                    this.weights[layer][neuron][conn] += step;
                    this.steps[layer][neuron][conn] = step;
                }

                //update the bias of the gradient and learning rate
                this.biases[layer][neuron] += this.learningRate * this.gradients[layer][neuron];
            }
        }
    }

    randomize(){
        return Math.random(); //generate a random number between 0-1
    }

    export(){
        const filename = this.name + '.json';
        fs.writeFile(filename, JSON.stringify({
            'layers' : this.layers, 
            'weights' : this.weights,
            'biases' : this.biases
        }));
        console.log(filename + 'saved to json file');
    }

    import(path){
        if(!path){return;}

        try{
            const importFile = fs.readFileSync(path);
            if(!importFile){
                return;
            }

            const network = JSON.parse(importFile);
            if(!network){
                return false;
            }

            if(network.layers){
                this.layers = network.layers;

                this.inputSize = this.layers[0];
                this.outputSize = this.layers[network.layers.length -1];
            }
            if(network.weights){
                this.weights = network.weights;
            }
            if(this.biases){
                this.biases = network.bias;
            }

            console.log('network initialized from file');

        }catch(e){
            console.error(e);
        }
    }
}

module.exports = NeuralNetwork;
