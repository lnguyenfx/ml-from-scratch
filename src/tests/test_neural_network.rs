use super::super::matrix::Matrix;
use super::super::neural_network::NeuralNetwork;

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