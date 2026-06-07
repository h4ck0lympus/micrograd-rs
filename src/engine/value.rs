use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Tanh,
    Noop,
}


struct Val {
    data: f64,
    grad: f64,
    _parents: Vec<Rc<RefCell<Val>>>,
    _op: Op,
    _backward: Rc<dyn Fn()>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<Val>>);

impl Value {
    pub fn new(data:f64) -> Self {
        Value(Rc::new(RefCell::new(Val {
            data: data,
            grad: 0.0,
            _parents: Vec::new(),
            _op: Op::Noop,
            _backward: Rc::new(|| {})
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn get_grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn tanh(&self) -> Self {
        let out = Value(Rc::new(RefCell::new(Val {
            data: self.data().tanh(),
            grad: 0.0,
            _parents: vec![self.0.clone()],
            _op: Op::Tanh,
            _backward: Rc::new(|| {})
        })));

        let out_node = out.0.clone();
        let left = self.0.clone();

        out.0.borrow_mut()._backward = Rc::new(move || {
            let out_data = out_node.borrow().data;
            let out_grad = out_node.borrow().grad;
            left.borrow_mut().grad += (1.0 - out_data * out_data) * out_grad;
        });

        out
    }

    pub fn backward(&self) {
        (self.0.borrow()._backward)();
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={0:.4}, grad={1:.4})", self.data(), self.get_grad())
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let data = self.data() + rhs.data();
        let out = Value(Rc::new(RefCell::new(Val {
            data,
            grad: 0.0,
            _parents: vec![self.0.clone(), rhs.0.clone()],
            _op: Op::Add,
            _backward: Rc::new(|| {})
        })));

        let left = self.0.clone();
        let right = rhs.0.clone();
        let out_node = out.0.clone();

        out.0.borrow_mut()._backward = Rc::new(move ||{
            let out_grad = out_node.borrow().grad;
            left.borrow_mut().grad += 1.0 * out_grad;
            right.borrow_mut().grad += 1.0 * out_grad;
        });

        out
    }
}


impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.data() * rhs.data();
        let out = Value(Rc::new(RefCell::new(Val {
            data,
            grad: 0.0,
            _parents: vec![self.0.clone(), rhs.0.clone()],
            _op: Op::Mul,
            _backward: Rc::new(|| {})
        })));

        let left = self.0.clone();
        let right = rhs.0.clone();
        let out_node = out.0.clone();

        out.0.borrow_mut()._backward = Rc::new( move || {
            let out_grad = out_node.borrow().grad;
            let right_data = right.borrow().data;
            let left_data = left.borrow().data;
            left.borrow_mut().grad +=  out_grad * right_data;
            right.borrow_mut().grad +=  out_grad * left_data;
        });

        out
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        let new_val = self + Value::new(-1.0) * rhs;
        new_val.0.borrow_mut()._op = Op::Sub;
        new_val
    }
}


impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.data() == 0.0 {
            panic!("Division by 0 not allowed")
        }
        let data = self.data() / rhs.data();
        Value(Rc::new(RefCell::new(Val {
            data,
            grad: 0.0,
            _parents: vec![self.0.clone(), rhs.0.clone()],
            _op: Op::Div,
            _backward: Rc::new(|| {})
        })))
    }
}

