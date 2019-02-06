pub struct Matrix {
    data: Vec<f64>,
    rows_count: usize,
    cols_count: usize,
}

// Basic Operations
impl Matrix {
    pub fn new() -> Matrix {
        Matrix {
            data: Vec::new(),
            rows_count: 0,
            cols_count: 0,
        }
    }

    pub fn from_vec(data: &Vec<f64>, rows_count: usize, cols_count: usize) -> Matrix {
        Matrix {
            data: data.clone(),
            rows_count,
            cols_count,
        }    
    }

    pub fn from_str(input: &str) -> Matrix {
        let input = input.replace(" ", "");
        let temp = input.split("],[").collect::<Vec<&str>>();
        let temp = temp.iter().map(|&s| s.replace("[", "").replace("]", "")).collect::<Vec<String>>();
        let rows_count = temp.len();
        let mut cols_count = 0;
        let mut data: Vec<f64> = Vec::new();
        for row in temp {
            let row_split = row.split(",").collect::<Vec<&str>>();
            cols_count = row_split.len(); 
            for num in row_split {
                data.push(num.parse().unwrap());
            }
        }
        return Matrix {
            data,
            rows_count,
            cols_count,
        }
    }

    pub fn set(&mut self, input: &str) {
        self.clear();
        let temp = Matrix::from_str(input);
        self.data = temp.data;
        self.rows_count = temp.rows_count;
        self.cols_count = temp.cols_count;
    }

    pub fn set_at_index(&mut self, row: usize, col: usize, value: f64) {
        if row > self.rows_count - 1 || col > self.cols_count -1 {
            panic!("Index out of bound!");
        }
        let index = row * self.cols_count + col;
        self.data[index] = value;
    }

    pub fn get_at_index(&self, row: usize, col: usize) -> f64 {
        if row > self.rows_count - 1 || col > self.cols_count -1 {
            panic!("Index out of bound!");
        }
        let index = row * self.cols_count + col;
        return self.data[index];
    }

    pub fn zero_fill(&mut self, rows_count: usize, cols_count: usize) {
        self.clear();
        
        self.data = vec![0f64;rows_count * cols_count];
        self.rows_count = rows_count;
        self.cols_count = cols_count;
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.rows_count = 0;
        self.cols_count = 0;
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows_count, self.cols_count)
    }

    pub fn to_string(&self) -> String {
        return self.to_string_fmt(0);
    }

    pub fn to_string_fmt(&self, decimal_places: usize) -> String {
        let mut result = String::new();
        result.push_str("[");
        for (i, num) in self.data.iter().enumerate() {
            if i % self.cols_count == 0 {
                result.push_str("[");
            }
            result.push_str(&format!("{:.*}", decimal_places, num));
            if (i + 1) % self.cols_count == 0 {
                result.push_str("]");
                if i < self.data.len() - 1 {
                    result.push_str(",");
                }
            } else {
                result.push_str(",");
            }
        }
        result.push_str("]");
        return result;
    }

    pub fn as_vec(&self) -> Vec<f64> {
        return self.data.clone();
    }
}

// Advanced Operations
impl Matrix {
    pub fn add(&self, other: &Matrix) -> Matrix {
        if (self.rows_count != other.rows_count) ||
           (self.cols_count != other.cols_count) {
            panic!("Incompatible Matrix Dimensions!");
        }
        let mut result = Matrix::new();
        result.zero_fill(self.rows_count, self.cols_count);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        return result;
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if (self.rows_count != other.rows_count) ||
           (self.cols_count != other.cols_count) {
            panic!("Incompatible Matrix Dimensions!");
        }
        let mut result = Matrix::new();
        result.zero_fill(self.rows_count, self.cols_count);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        return result;
    }

    pub fn dot_prod(&self, other: &Matrix) -> Matrix {
        if self.cols_count != other.rows_count {
            panic!("Incompatible Matrix Dimensions!");
        }

        let mut result = Matrix::new();
        result.rows_count = self.rows_count;
        result.cols_count = other.cols_count;

        for i in 0..self.rows_count {
            for k in 0..other.cols_count {
                let mut prod_sum = 0.0;
                for j in 0..self.cols_count {
                    prod_sum += self.get_at_index(i, j) * other.get_at_index(j, k); 
                }
                result.data.push(prod_sum);
            }
        }
        return result;        
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new();
        result.zero_fill(self.cols_count, self.rows_count);
        for i in 0..self.rows_count {
            for j in 0..self.cols_count {
                result.set_at_index(j, i, self.get_at_index(i, j));
            }
        }
        return result;
    }

    pub fn clone(&self) -> Matrix {
        let result = Matrix::from_vec(&self.data, self.rows_count, self.cols_count);
        return result;
    }

    pub fn map<F>(&mut self, func: F) 
        where F: Fn(f64) -> f64 {
        for i in 0..self.data.len() {
            self.data[i] = func(self.data[i]);
        }
    }

    pub fn map_with_index<F>(&mut self, func: F) 
        where F: Fn(f64, usize) -> f64 {
        for i in 0..self.data.len() {
            self.data[i] = func(self.data[i], i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

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
}
