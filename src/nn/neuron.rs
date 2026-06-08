use crate::engine::value::Value;
use rand::RngExt;
// neuron = activation(x1w1 + x2w2 + ... + xNwN + b)
// x1,x2, ... xN are inputs to neuron
// w1,w2, ... wN are weights of neuron (importance of each input)
// b is bias . We have some bias value for each neuron, on which neuron should be activated.
// activation function can be tanh/relu/sigmoid/etc.
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(ninp: usize) -> Self {
        let mut rng = rand::rng();
        Neuron {
            weights: (0..ninp)
                .map(|_| { Value::new(rng.random_range(-1.0..1.0)) })
                .collect(),
            bias: Value::new(rng.random_range(-1.0..1.0))
        }
    }

    pub fn forward(&self, x:Vec<Value>) -> Value {
        // w * x + b
        let data = self.weights.iter()
            .zip(x.clone().iter())
            .fold(self.bias.clone(), |acc, (w, x)| { 
                acc + w.clone() * x.clone() 
            }); // returns a Value
        data.tanh()
    }
}
