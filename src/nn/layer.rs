use crate::engine::value::Value;

use super::neuron::Neuron;

// a layer is collection of neurons 
// ninp = number of inputs each neuron of layer expects
// num = number of neurons in the layer
pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(ninp: usize, num: usize) -> Self {
       Layer {
           neurons: (0..num)
               .map(|_| { Neuron::new(ninp) })
               .collect()
       } 
    }

    pub fn forward(&self, x: Vec<Value> ) -> Vec<Value> {
        self.neurons.iter()
            .map(|n| { n.forward(x.clone()) })
            .collect()
    }
}
