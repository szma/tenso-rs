use tenso_rs::nn::MLP;
use tenso_rs::tensor::{Context, Tensor};

fn sgd_step(params: &[Tensor], lr: f32) {
    for p in params {
        if let Some(grad) = p.grad() {
            let new_data = p.data() - &(grad * lr);
            p.set_data(new_data);
        }
    }
}

fn main() {
    let ctx = Context::new();

    // XOR dataset: [4 samples, 2 features]
    let x = ctx.tensor(&[0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]);

    // XOR targets: [4 samples, 1 output]
    let y_true = ctx.tensor(&[0., 1., 1., 0.], &[4, 1]);

    // Network: 2 -> 4 -> 1
    let mlp = MLP::new(&ctx, &[2, 4, 1]);
    let lr = 0.05;

    for epoch in 0..100 {
        ctx.zero_grad();

        let y_pred = mlp.forward(x);
        let loss = (y_pred - y_true).pow(2.0).sum();

        loss.backward();

        if epoch % 10 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss.data()[[0]]);
        }

        sgd_step(&mlp.params(), lr);
    }

    // Final prediction
    ctx.zero_grad();
    let y_pred = mlp.forward(x);

    println!("\nFinal predictions:");
    println!("Input [0,0] -> {:.3}", y_pred.data()[[0, 0]]);
    println!("Input [0,1] -> {:.3}", y_pred.data()[[1, 0]]);
    println!("Input [1,0] -> {:.3}", y_pred.data()[[2, 0]]);
    println!("Input [1,1] -> {:.3}", y_pred.data()[[3, 0]]);
    println!("\nExpected: 0, 1, 1, 0");
}
