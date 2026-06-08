use micrograd_rs::engine::value::Value;
use micrograd_rs::nn::mlp::MLP;
use micrograd_rs::nn::module::Module;

fn main() {
    // inputs for model
    let _x = vec![
        Value::new(2.0),
        Value::new(3.0),
        Value::new(-1.0),
    ];

    // we have 3 inputs for model and 3 layers
    // 2 intermediate layers and 1 output layer
    // https://cs231n.github.io/assets/nn1/neural_net.jpeg
    let model = MLP::new(3, vec![4, 4, 1]);

    // sample dataset
    // sample inputs from dataset
    let xs = vec![
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];

    // expected output 
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    for i in 0..200 {
        // forward pass
        let ypred: Vec<Value> = xs
            .iter()
            .map(|x| {
                let x_values = x
                    .iter()
                    .map(|v| Value::new(*v))
                    .collect::<Vec<Value>>();
                model.forward(x_values)[0].clone()
            })
        .collect();

        // calculate loss
        let loss: Value = ypred
            .iter()
            .zip(ys.iter())
            .fold(Value::new(0.0), |acc, (yp, y)|{
                acc + (yp.clone() - Value::new(*y)).pow(2.0)
            });

        model.zero_grad();
        // backpropogate
        loss.backward();

        if i % 10 == 0 {
            println!("y predictions for x values(run-{}):", i);
            for y in &ypred {
                println!("{}", y);
            }
            println!("loss = {}", loss);
            println!("-----------")
        }

        // update
        let step = 0.1; // learning rate
        for param in model.parameters() {
            param.set_data(param.get_data() - step * param.get_grad()); // gradient descent
        }
    }
}
