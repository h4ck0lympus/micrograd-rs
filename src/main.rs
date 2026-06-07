use micrograd_rs::engine::value::{Value};

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let f = Value::new(-2.0);
    let e = a.clone() * b.clone();
    let d = e.clone() + c.clone();
    let l = d.clone() * f.clone();
    l.set_grad(1.0);
    l.backward();
    d.backward();
    f.backward();
    c.backward();
    e.backward();
    a.backward();
    b.backward();
    println!("a = {}, b = {}, c = {}", a, b, c);
    println!("d = {}, e = {}, f = {}", d, e, f);
    println!("l = {}", l);
}
