use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Exp,
    Tanh,
    Relu,
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

    pub fn get_data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn get_grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn tanh(&self) -> Self {
        let out = Value(Rc::new(RefCell::new(Val {
            data: self.get_data().tanh(),
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

    pub fn relu(&self) -> Self {
        let data = if self.get_data() < 0.0 { 0.0 } else { self.get_data() };

        let out = Value(Rc::new(RefCell::new(Val {
            data,
            grad: 0.0,
            _parents: vec![self.0.clone()],
            _op: Op::Relu,
            _backward: Rc::new(|| {})
        })));

        let out_node = out.0.clone();
        let left = self.0.clone();

        out.0.borrow_mut()._backward = Rc::new(move || {
            let out_data = out_node.borrow().data;
            let out_grad = out_node.borrow().grad;
            let local_grad = if out_data > 0.0 { 1.0 } else { 0.0 };
            left.borrow_mut().grad += local_grad * out_grad;
        });

        out
    }

    pub fn exp(&self) -> Self {
        let out = Value(Rc::new(RefCell::new(Val {
            data: self.get_data().exp(),
            grad: 0.0,
            _parents: vec![self.0.clone()],
            _op: Op::Exp,
            _backward: Rc::new(|| {})
        })));

        let out_node = out.0.clone();
        let left = self.0.clone();
        
        out.0.borrow_mut()._backward = Rc::new(move || {
            let out_grad = out_node.borrow().grad;
            let out_data = out_node.borrow().data;
            left.borrow_mut().grad += out_data * out_grad;
        });

        out
    }

    pub fn pow(&self, power: f64) -> Self {
        let out = Value(Rc::new(RefCell::new(Val {
            data: self.get_data().powf(power),
            grad: 0.0,
            _parents: vec![self.0.clone()],
            _op: Op::Pow,
            _backward: Rc::new(|| {})
        })));

        let out_node = out.0.clone();
        let left = self.0.clone();
        
        out.0.borrow_mut()._backward = Rc::new(move || {
            let out_grad = out_node.borrow().grad;
            let local_grad = power * left.borrow().data.powf(power-1.0);
            left.borrow_mut().grad += local_grad * out_grad;
        });
        out
    }

    // topo sort - dfs approach
    fn topo_sort(
        node: &Rc<RefCell<Val>>, 
        visited: &mut HashSet<*const RefCell<Val>>, 
        topo: &mut Vec<Rc<RefCell<Val>>>
    ) {
        let ptr = Rc::as_ptr(node);
        if visited.contains(&ptr) {
            return;
        }

        visited.insert(ptr);

        for parent in node.borrow()._parents.iter() {
            Self::topo_sort(parent, visited, topo);
        }
        topo.push(node.clone());
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        Self::topo_sort(&self.0, &mut visited, &mut topo);

        self.set_grad(1.0);
        for node in topo.into_iter().rev() {
            let backward = node.borrow()._backward.clone();
            backward();
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={0:.8}, grad={1:.8})", self.get_data(), self.get_grad())
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let data = self.get_data() + rhs.get_data();
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
            left.borrow_mut().grad += out_grad;
            right.borrow_mut().grad += out_grad;
        });

        out
    }
}


impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.get_data() * rhs.get_data();
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
        if rhs.get_data() == 0.0 {
            panic!("Division by 0 not allowed")
        }
        let data = self.get_data() / rhs.get_data();
        let out = Value(Rc::new(RefCell::new(Val {
            data,
            grad: 0.0,
            _parents: vec![self.0.clone(), rhs.0.clone()],
            _op: Op::Div,
            _backward: Rc::new(|| {})
        })));

        let left = self.0.clone();
        let right = rhs.0.clone();
        let out_node = out.0.clone();

        out.0.borrow_mut()._backward = Rc::new(move || {
            let out_grad = out_node.borrow().grad;
            let left_data = left.borrow().data;
            let right_data = right.borrow().data;
            left.borrow_mut().grad += out_grad * (1.0 / right_data);
            right.borrow_mut().grad += out_grad * (-left_data / (right_data * right_data));
        });

        out
    }
}

