pub fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
} 

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_sigmoid() {
        let inputs = [1.0, 0.5, 0.25];
        let expected = [
            String::from("0.7310585786"),
            String::from("0.6224593312"),
            String::from("0.5621765009"),
        ];

        for i in 0..inputs.len() {
            let actual = format!("{:.*}", 10, sigmoid(inputs[i]));
            assert_eq!(actual, expected[i]);
        }
    }
}
