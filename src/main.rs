extern crate ml_from_scratch;

use ml_from_scratch::matrix::Matrix;
use ml_from_scratch::neural_network::{TrainingData, NeuralNetwork};

fn main() {
    neural_network_example();
}

fn neural_network_example() {
    println!("Example: Learning the XOR Gate using a Neural Network");

    let num_inputs = 2;
    let num_layers = 2;
    let mut nn = NeuralNetwork::new(num_inputs, num_layers);
    
    nn.set_weights(&vec![
        Matrix::from_str("[[-0.1,0.1],[-0.1,0.1]]"),
        Matrix::from_str("[[0.1,-0.1],[0.1,-0.1]]"),
    ]);

    for i in 0..num_layers {
        println!("Initial weights for layer {}: {}", i, nn.get_weights()[i].to_string_fmt(1));
    }

    // Training XOR Gate
    let inputs: Vec<Vec<f64>> = vec![
        vec![1.0, 1.0], // T T
        vec![1.0, 0.0], // T F
        vec![0.0, 1.0], // F T
        vec![0.0, 0.0], // F F
    ];

    let targets: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0], // T T -> F
        vec![1.0, 0.0], // T F -> T
        vec![1.0, 0.0], // F T -> T
        vec![0.0, 1.0], // F F -> F
    ];
    
    let mut data: TrainingData = Vec::new();
    for i in 0..inputs.len() {
        data.push((inputs[i].clone(), targets[i].clone()));
    }

    let learning_rate = 0.1;
    let epochs = 500000;
    
    nn.train(&data, learning_rate, epochs);

    for i in 0..num_layers {
        println!("Final weights for layer {}: {}", i, nn.get_weights()[i].to_string_fmt(5));
    }

    // Test the model after training
    println!("\nEvaluation model using final weights: ");
    println!("T T -> {}", nn.execute(&vec![1.0, 1.0]).to_string_fmt(5));
    println!("T F -> {}", nn.execute(&vec![1.0, 0.0]).to_string_fmt(5));
    println!("F T -> {}", nn.execute(&vec![0.0, 1.0]).to_string_fmt(5));
    println!("F F -> {}", nn.execute(&vec![0.0, 0.0]).to_string_fmt(5));
}
