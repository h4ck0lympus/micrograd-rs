use micrograd_rs::engine::value::{Value};

fn main() {
    let x1 = Value::new(2.0); let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0); let w2 = Value::new(1.0);
    let b = Value::new(6.8813735);
    let x1w1 = x1.clone() * w1.clone(); let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let o = n.clone().tanh();
    o.backward();
    println!("x1 = {}, x2 = {}", x1, x2);
    println!("w1 = {}, w2 = {}", w1, w2);
    println!("x1w1 = {}, x2w2 = {}", x1w1, x2w2);
    println!("x1w1x2w2 = {}", x1w1x2w2);
    println!("n = {}", n)
}
