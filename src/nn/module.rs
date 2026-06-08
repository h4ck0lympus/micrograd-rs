use crate::engine::value::Value;

pub trait Module {
    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.set_grad(0.0);
        }
    }
}
