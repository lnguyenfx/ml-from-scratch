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
#[path = "tests/test_matrix.rs"]
mod test;
