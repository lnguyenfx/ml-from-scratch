use super::super::matrix::Matrix;

#[test]
fn test_empty_matrix() {
    let matrix_a: Matrix = Matrix::new();
    assert_eq!(matrix_a.size(), (0, 0));
}

#[test]
fn test_create_matrix_from_vector() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let matrix_a: Matrix = Matrix::from_vec(&data, 3, 3);
    let expected = String::from("[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]");
    assert_eq!(matrix_a.to_string_fmt(1), expected);
}

#[test]
fn test_return_matrix_as_vector() {
    let matrix_a = Matrix::from_vec(&vec![1.0, 2.0, 3.0], 1, 3);
    let vec_matrix_a = matrix_a.as_vec();
    assert_eq!(vec_matrix_a.len(), 3);
}

#[test]
fn test_set_matrix_valid() {
    let mut matrix_a: Matrix = Matrix::new();
    assert_eq!(matrix_a.rows_count, 0);
    assert_eq!(matrix_a.cols_count, 0);

    assert_eq!(matrix_a.to_string(), "[]");

    matrix_a.set("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]");
    assert_eq!(matrix_a.rows_count, 3);
    assert_eq!(matrix_a.cols_count, 3);

    assert_eq!(matrix_a.to_string(), "[[1,2,3],[4,5,6],[7,8,9]]");
    
    matrix_a.set("[1, 2, 3]");
    assert_eq!(matrix_a.rows_count, 1);
    assert_eq!(matrix_a.cols_count, 3);

    assert_eq!(matrix_a.to_string(), "[[1,2,3]]");

    matrix_a.set("[[1, 2, 3]]");
    assert_eq!(matrix_a.rows_count, 1);
    assert_eq!(matrix_a.cols_count, 3);

    assert_eq!(matrix_a.to_string(), "[[1,2,3]]");

    matrix_a.set("[[1], [2]");
    assert_eq!(matrix_a.rows_count, 2);
    assert_eq!(matrix_a.cols_count, 1);

    assert_eq!(matrix_a.to_string(), "[[1],[2]]");
}

#[test]
fn test_set_matrix_at_valid_index() {
    let mut matrix_a = Matrix::new();
    matrix_a.set("[[1, 2, 3], [4, 5, 6]]");
    matrix_a.set_at_index(1, 0, 8.0);

    assert_eq!(matrix_a.to_string(), "[[1,2,3],[8,5,6]]");

    assert_eq!(matrix_a.get_at_index(1, 2), 6.0);
}

#[should_panic]
#[test]
fn test_set_matrix_at_invalid_index() {
    let mut matrix_a = Matrix::new();
    matrix_a.set("[[1, 2, 3], [4, 5, 6]]");
    matrix_a.set_at_index(5, 0, 8.0);
}

#[test]
fn test_matrix_zero_fill() {
    let mut matrix_a = Matrix::new();
    matrix_a.zero_fill(3, 3);
    assert_eq!(matrix_a.to_string(), "[[0,0,0],[0,0,0],[0,0,0]]");
}

#[test]
fn test_matrix_add_valid() {
    let mut matrix_a = Matrix::new();
    let mut matrix_b = Matrix::new();

    matrix_a.set("[[1, 2], [1, 2]]");
    matrix_b.set("[[1, 1], [1, 1]]");

    let matrix_c = matrix_a.add(&matrix_b);

    assert_eq!(matrix_c.to_string(), "[[2,3],[2,3]]");
}

#[test]
fn test_matrix_subtract_valid() {
    let mut matrix_a = Matrix::new();
    let mut matrix_b = Matrix::new();

    matrix_a.set("[[1, 2], [1, 2]]");
    matrix_b.set("[[1, 1], [1, 1]]");

    let matrix_c = matrix_a.subtract(&matrix_b);

    assert_eq!(matrix_c.to_string(), "[[0,1],[0,1]]");
}

#[should_panic]
#[test]
fn test_matrix_add_invalid_dimensions() {
    let mut matrix_a = Matrix::new();
    let mut matrix_b = Matrix::new();

    matrix_a.set("[[1, 2], [1, 2]]");
    matrix_b.set("[[1, 1]");

    matrix_a.add(&matrix_b);
}

#[should_panic]
#[test]
fn test_matrix_subtract_invalid_dimensions() {
    let mut matrix_a = Matrix::new();
    let mut matrix_b = Matrix::new();

    matrix_a.set("[[1, 2], [1, 2]]");
    matrix_b.set("[[1, 1]");

    matrix_a.subtract(&matrix_b);
}

#[should_panic]
#[test]
fn test_matrix_dot_prod_invalid_dimensions() {
    let mut matrix_a = Matrix::new();
    let mut matrix_b = Matrix::new();

    matrix_a.set("[[1, 2, 3], [1, 2, 3]]");
    matrix_b.set("[[1, 1, 1]");

    matrix_a.dot_prod(&matrix_b);
}

#[test]
fn test_matrix_dot_prod_valid() {
    let mut matrix_a = Matrix::new();
    let mut matrix_b = Matrix::new();

    matrix_a.set("[[1, 2], [1, 2]]");
    matrix_b.set("[[1, 1], [1, 1]]");

    let mut matrix_c = matrix_a.dot_prod(&matrix_b);
    
    assert_eq!(matrix_c.to_string(), "[[3,3],[3,3]]");

    matrix_a.set("[[1, 2, 3]");
    matrix_b.set("[[4],[5],[6]]");
    matrix_c = matrix_a.dot_prod(&matrix_b);

    assert_eq!(matrix_c.to_string(), "[[32]]");

    // matrix_a.set("[[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]]");
    // matrix_b.set("[[0.761],[0.603],[0.650]]");
    // matrix_c = matrix_a.dot_prod(&matrix_b);

    // assert_eq!(matrix_c.to_string(), "[[0.975, 0.888, 1.254]]");
}

#[test]
fn test_matrix_transpose() {
    let matrix_a = Matrix::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let matrix_b = matrix_a.transpose();
    assert_eq!(matrix_a.to_string(), "[[1,2,3],[4,5,6]]");
    assert_eq!(matrix_b.to_string(), "[[1,4],[2,5],[3,6]]");
    let matrix_a = Matrix::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    let matrix_b = matrix_a.transpose();
    assert_eq!(matrix_a.to_string(), "[[1,2,3],[4,5,6],[7,8,9]]");
    assert_eq!(matrix_b.to_string(), "[[1,4,7],[2,5,8],[3,6,9]]");
}

#[test]
fn test_matrix_clone() {
    let matrix_a = Matrix::from_vec(&vec![1.0, 2.0, 3.0], 3, 1);
    let matrix_b = matrix_a.clone();
    assert_eq!(matrix_b.to_string(), "[[1],[2],[3]]");
}

#[test]
fn test_matrix_map() {
    let mut matrix_a = Matrix::from_vec(&vec![1.0, 2.0, 3.0], 3, 1);
    matrix_a.map(|x| x + 2f64);
    assert_eq!(matrix_a.to_string(), "[[3],[4],[5]]");
}

#[test]
fn test_matrix_map_with_index() {
    let mut matrix_a = Matrix::from_vec(&vec![1.0, 2.0, 3.0], 3, 1);
    matrix_a.map_with_index(|x, i| x + i as f64);
    assert_eq!(matrix_a.to_string(), "[[1],[3],[5]]");
}