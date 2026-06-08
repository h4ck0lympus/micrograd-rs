use super::layer::Layer;
use crate::engine::value::Value;
use crate::nn::module::Module;
// MLP - Multi Layer Perceptron
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    // number of inputs for each neuron of layer
    // vector denoting size of each layer
    pub fn new(ninp: usize, nlayers: Vec<usize>) -> Self {
        assert!(nlayers.len() > 0); // no underflow possible now
        let mut sizes = Vec::new();
        sizes.push(ninp);
        sizes.extend(nlayers);
        let layers = (0..sizes.len()-1)
            .map(|i| { Layer::new(sizes[i], sizes[i+1]) })
            .collect();
        MLP { layers }
    }

    pub fn forward(&self, mut x: Vec<Value>) -> Vec<Value> {
        for l in &self.layers{
            x = l.forward(x);
        }
        x
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::<Value>::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}
