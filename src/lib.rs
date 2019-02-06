extern crate rand;

pub mod math;

pub mod matrix;

use rand::Rng;

use matrix::Matrix;
use math::sigmoid;

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
mod tests {
    use super::matrix::Matrix;
    use super::NeuralNetwork;

    #[test]
    fn test_nn_set_weights() {
        let num_inputs = 2;
        let num_layers = 2;
        let mut nn = NeuralNetwork::new(num_inputs, num_layers);
   
        let weights_layer1 = Matrix::from_vec(&vec![1.0, 0.75, 0.5, 0.25], num_inputs, num_inputs);
        let weights_layer2 = Matrix::from_vec(&vec![0.25, 0.5, 0.75, 1.0], num_inputs, num_inputs);
        nn.set_weights(&vec![weights_layer1, weights_layer2]);

        let weights = nn.get_weights();
        let expected: Vec<String> = vec![
            String::from("[[1.00,0.75],[0.50,0.25]]"),
            String::from("[[0.25,0.50],[0.75,1.00]]"),
        ];
        for i in 0..num_layers {
            assert_eq!(weights[i].to_string_fmt(2), expected[i]);
        }
    }

    #[test]
    fn test_nn_randomize_weights() {
        let num_inputs = 2;
        let num_layers = 1;
        let mut nn = NeuralNetwork::new(num_inputs, num_layers);
        nn.randomize_weights();

        let bound: f64 = 0.70710678118654752440084436210485; // 1 / (sqrt(2)

        for weights in nn.get_weights() {
            for i in 0..num_inputs {
                for j in 0..num_inputs {
                    let weight = weights.get_at_index(i, j);
                    assert!(weight >= -bound && weight <= bound);
                }
            }
        }
    }

    #[test]
    fn test_nn_get_outputs_for_one_layer() {
        let num_inputs = 2;
        let num_layers = 1;
        let mut nn = NeuralNetwork::new(num_inputs, num_layers);
        
        let inputs = vec![1.0, 1.0];

        let weights = Matrix::from_vec(&vec![1.0, 0.75, 0.5, 0.25], num_inputs, num_inputs);
        nn.set_weights(&vec![weights]);

        let outputs = nn.get_outputs(&inputs);
        assert_eq!(outputs.len(), 2);

        let expected = vec![
            String::from("[[1.0000000000],[1.0000000000]]"),
            String::from("[[0.8519528020],[0.6791786992]]")
        ];

        for i in 0..outputs.len() {
            assert_eq!(outputs[i].to_string_fmt(10), expected[i]);
        }
    }

    #[test]
    fn test_nn_get_outputs_for_more_than_one_layer() {
        let num_inputs = 2;
        let num_layers = 2;
        let mut nn = NeuralNetwork::new(num_inputs, num_layers);
        
        let inputs = vec![1.0, 1.0];

        let weights_layer_1 = Matrix::from_vec(&vec![1.0, 0.75, 0.5, 0.25], num_inputs, num_inputs);
        let weights_layer_2 = Matrix::from_vec(&vec![0.25, 0.5, 0.75, 1.0], num_inputs, num_inputs);
        nn.set_weights(&vec![weights_layer_1, weights_layer_2]);

        let outputs = nn.get_outputs(&inputs);
        assert_eq!(outputs.len(), 3);
        
        let expected = vec![
            String::from("[[1.0000000000],[1.0000000000]]"),
            String::from("[[0.8519528020],[0.6791786992]]"),
            String::from("[[0.6347333953],[0.7888726343]]"),
        ];

        for i in 0..outputs.len() {
            assert_eq!(outputs[i].to_string_fmt(10), expected[i]);
        }
    }

    #[test]
    fn test_nn_get_errors() {
        let num_inputs = 2;
        let num_layers = 2;
        let mut nn = NeuralNetwork::new(num_inputs, num_layers);
       
        let weights_layer_1 = Matrix::from_vec(&vec![1.0, 0.75, 0.5, 0.25], num_inputs, num_inputs);
        let weights_layer_2 = Matrix::from_vec(&vec![0.25, 0.5, 0.75, 1.0], num_inputs, num_inputs);
        nn.set_weights(&vec![weights_layer_1, weights_layer_2]);
        
        let inputs = vec![1.0, 1.0];
        let targets = vec![0.0, 1.0];
        let errors = nn.get_errors(&inputs, &targets);

        let expected = vec![
            String::from("[[-0.2518116765],[-0.0927905032]]"),
            String::from("[[-0.0531196660],[-0.2649226808]]"),
            String::from("[[-0.6347333953],[0.2111273657]]"),
        ];

        for i in 0..errors.len() {
            assert_eq!(errors[i].to_string_fmt(10), expected[i]);
        }
    }

    #[test]
    fn test_nn_get_deltas() {
        let num_inputs = 2;
        let num_layers = 2;
        let mut nn = NeuralNetwork::new(num_inputs, num_layers);
        
        let inputs = vec![1.0, 1.0];
        let targets = vec![0.0, 1.0];

        let weights_layer1 = Matrix::from_vec(&vec![1.0, 0.75, 0.5, 0.25], num_inputs, num_inputs);
        let weights_layer2 = Matrix::from_vec(&vec![0.25, 0.5, 0.75, 1.0], num_inputs, num_inputs);
        nn.set_weights(&vec![weights_layer1, weights_layer2]);
        
        let learning_rate = 0.1;
        let deltas = nn.get_deltas(&inputs, &targets, learning_rate);

        assert_eq!(deltas.len(), 2);

        let expected = vec![
            String::from("[[-0.0006699942,-0.0006699942],[-0.0057725326,-0.0057725326]]"),
            String::from("[[-0.0125374207,-0.0099948601],[0.0029957908,0.0023882512]]"),
        ];

        for i in 0..deltas.len() {
            assert_eq!(deltas[i].to_string_fmt(10), expected[i]);
        }
    }
}
