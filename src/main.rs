use micrograd_rs::engine::value::{Value};

fn main() {
    println!("Hello, world!");
    let a = Value::new(3.0);
    let b = Value::new(2.0);
    println!("a = {}", a);
    println!("b = {}", b);
    println!("a * b = {}", a * b);
    // println!("a = {}", a);
}

