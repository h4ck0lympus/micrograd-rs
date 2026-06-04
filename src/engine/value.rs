use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

pub struct Value {
    data: f64,
}

impl Value {
    pub fn new(data:f64) -> Self {
        Self { data }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={0:.4})", self.data)
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        Value::new(self.data + rhs.data)
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        Value::new(self.data - rhs.data)
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        Value::new(self.data * rhs.data)
    }
}

impl Div for Value { 
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.data == 0.0 {
            panic!("Division by 0 not allowed")
        }
        Value::new(self.data / rhs.data)
    }
}
