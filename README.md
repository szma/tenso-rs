# tenso-rs

Minimal autograd library in Rust for neural networks.

## Features

- Automatic differentiation (backpropagation)
- Tensor operations: `+`, `-`, `*`, `matmul`, `relu`, `pow`, `sum`
- Neural network layers: `Linear`, `MLP`

## Example

```rust
use tenso_rs::nn::MLP;
use tenso_rs::tensor::Context;

fn main() {
    let ctx = Context::new();

    // Data
    let x = ctx.tensor(&[0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]);
    let y_true = ctx.tensor(&[0., 1., 1., 0.], &[4, 1]);

    // Network: 2 -> 4 -> 1
    let mlp = MLP::new(&ctx, &[2, 4, 1]);

    for _ in 0..1000 {
        ctx.zero_grad();
        let y_pred = mlp.forward(x);
        let loss = (y_pred - y_true).pow(2.0).sum();
        loss.backward();

        // SGD update
        for p in mlp.params() {
            if let Some(grad) = p.grad() {
                p.set_data(p.data() - &(grad * 0.5));
            }
        }
    }
}
```

## Run

```bash
cargo run --example xor
```
