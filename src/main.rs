use micrograd_rs::engine::value::Value;
use micrograd_rs::nn::mlp::MLP;

fn main() {
    let x = vec![
        Value::new(2.0),
        Value::new(3.0),
        Value::new(-1.0),
    ];

    let model = MLP::new(3, vec![4, 4, 1]);

    let out = model.forward(x);

    println!("output:");
    for v in &out {
        println!("{}", v);
    }

    let loss = out[0].clone();
    loss.backward();

    println!("after backward:");
    println!("loss = {}", loss);
}
