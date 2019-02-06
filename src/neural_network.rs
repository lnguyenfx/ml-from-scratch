extern crate rand;

use rand::Rng;

use super::matrix::Matrix;
use super::math::sigmoid;

pub type TrainingData = Vec<(Vec<f64>, Vec<f64>)>;

pub struct NeuralNetwork {
    num_inputs: usize,
    num_layers: usize,
    weights: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(num_inputs: usize, num_layers: usize) -> NeuralNetwork {
        NeuralNetwork {
            num_inputs,
            num_layers,
            weights: Vec::<Matrix>::new(),
        }
    }

    pub fn get_weights(&self) -> &Vec<Matrix> {
        return &self.weights;
    }

    pub fn set_weights(&mut self, new_weights: &Vec<Matrix>) {
        self.weights = Vec::new();
        for weight in new_weights {
            self.weights.push(weight.clone());
        }
    }

    pub fn randomize_weights(&mut self) {
        let mut rng = rand::thread_rng();
        let num_inputs = self.num_inputs;
        for _ in 0..self.num_layers {
            let mut weights: Vec<f64> = Vec::new();
            for _ in 0..(num_inputs * num_inputs) {
                let bound: f64 = 1.0 / (self.num_inputs as f64).sqrt();
                weights.push(rng.gen_range(-bound, bound));
            }
            self.weights.push(Matrix::from_vec(&weights, num_inputs, num_inputs));
        }
    }

    pub fn get_outputs(&self, inputs: &Vec<f64>) -> Vec<Matrix> {
        let mut result: Vec<Matrix> = Vec::new();

        let mut inputs = Matrix::from_vec(&inputs, self.num_inputs, 1);
        result.push(inputs.clone());
        for i in 0..self.num_layers {
            let mut outputs = self.weights[i].dot_prod(&inputs);
            outputs.map(sigmoid);
            inputs = outputs.clone();
            result.push(outputs);
        }

        return result;
    }

    pub fn get_errors(&self, inputs: &Vec<f64>, targets: &Vec<f64>) -> Vec<Matrix> {
        let mut result: Vec<Matrix> = Vec::new();

        let weights = self.get_weights();
        let outputs = self.get_outputs(inputs);
        let targets = Matrix::from_vec(targets, outputs[0].size().0, outputs[0].size().1); 

        let errors = targets.subtract(&outputs[outputs.len() - 1]);
        result.push(errors);

        for i in (0..weights.len()).rev() {
            let errors = weights[i].dot_prod(&result[result.len() - 1]);
            result.push(errors);
        }

        result.reverse();
        return result;
    }

    pub fn get_deltas(&self, inputs: &Vec<f64>, targets: &Vec<f64>, learning_rate: f64) -> Vec<Matrix> {
        let mut result: Vec<Matrix> = Vec::new();

        let outputs = self.get_outputs(inputs);
        
        let errors = self.get_errors(inputs, targets);

        for layer in 0..self.num_layers {
            let error = errors[layer + 1].as_vec();
            let prev_output = outputs[layer].as_vec();
            let output = outputs[layer + 1].as_vec() ;

            let mut delta_vec: Vec<f64> = Vec::new();
            for j in 0..error.len() {
                for k in 0..output.len() {
                    let delta: f64 = learning_rate * error[j] * output[j] * (1.0 - output[j]) * prev_output[k];
                    delta_vec.push(delta); 
                }
            }
            let delta_matrix = Matrix::from_vec(&delta_vec, inputs.len(), inputs.len());
            // println!("-->{}", delta_matrix.to_string_fmt(10));
            result.push(delta_matrix);
        }

        return result;
    }

    pub fn train(&mut self, data: &TrainingData, learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for pair in data {
                let inputs: &Vec<f64> = &pair.0;
                let targets: &Vec<f64> = &pair.1;
                
                let deltas = self.get_deltas(inputs, targets, learning_rate);
                
                for i in 0..self.num_layers {
                    self.weights[i] = self.weights[i].add(&deltas[i]);
                }
            }        
        }
        
    }

    pub fn execute(&self, inputs: &Vec<f64>) -> Matrix {
        let outputs: Vec<Matrix> = self.get_outputs(inputs);
        return outputs.last().unwrap().clone();
    }

}

#[cfg(test)]
#[path = "tests/test_neural_network.rs"]
mod test;