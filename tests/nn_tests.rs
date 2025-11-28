use tenso_rs::{Context, Linear, MLP, sgd_step};

#[test]
fn test_linear_forward() {
    let ctx = Context::new();
    let linear = Linear::new(&ctx, 2, 3);

    let x = ctx.tensor(&[1.0, 2.0], &[1, 2]);
    let y = linear.forward(x);

    assert_eq!(y.shape(), vec![1, 3]);
}

#[test]
fn test_linear_params() {
    let ctx = Context::new();
    let linear = Linear::new(&ctx, 2, 3);

    let params = linear.params();
    assert_eq!(params.len(), 2); // weight + bias

    assert_eq!(params[0].shape(), vec![2, 3]); // weight
    assert_eq!(params[1].shape(), vec![1, 3]); // bias
}

#[test]
fn test_mlp_creation() {
    let ctx = Context::new();
    let mlp = MLP::new(&ctx, &[2, 4, 1]);

    assert_eq!(mlp.layers.len(), 2);
}

#[test]
fn test_mlp_forward() {
    let ctx = Context::new();
    let mlp = MLP::new(&ctx, &[2, 4, 1]);

    let x = ctx.tensor(&[1.0, 2.0], &[1, 2]);
    let y = mlp.forward(x);

    assert_eq!(y.shape(), vec![1, 1]);
}

#[test]
fn test_mlp_params() {
    let ctx = Context::new();
    let mlp = MLP::new(&ctx, &[2, 4, 1]);

    // Layer 1: weight(2,4) + bias(1,4) = 2 params
    // Layer 2: weight(4,1) + bias(1,1) = 2 params
    let params = mlp.params();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_sgd_step() {
    let ctx = Context::new();
    let x = ctx.tensor(&[1.0], &[1]);
    let target = ctx.tensor(&[0.0], &[1]);

    // loss = (x - target)^2 = x^2
    // dloss/dx = 2x = 2
    let loss = (x - target).pow(2.0);
    loss.backward();

    let params = vec![x];
    sgd_step(&params, 0.1);

    // x_new = x - lr * grad = 1.0 - 0.1 * 2.0 = 0.8
    assert!((x.data()[[0]] - 0.8).abs() < 1e-6);
}

#[test]
fn test_training_reduces_loss() {
    let ctx = Context::new();

    // Simple regression: learn to output 0.5 from input [1, 1]
    let x = ctx.tensor(&[1.0, 1.0], &[1, 2]);
    let target = ctx.tensor(&[0.5], &[1, 1]);

    let mlp = MLP::new(&ctx, &[2, 4, 1]);

    // Initial loss
    let initial_pred = mlp.forward(x);
    let initial_loss = (initial_pred - target).pow(2.0).sum();
    let initial_loss_val = initial_loss.data()[[0]];

    // Train for a few steps
    for _ in 0..100 {
        ctx.zero_grad();
        let pred = mlp.forward(x);
        let loss = (pred - target).pow(2.0).sum();
        loss.backward();
        sgd_step(&mlp.params(), 0.1);
    }

    // Final loss should be lower
    let final_pred = mlp.forward(x);
    let final_loss = (final_pred - target).pow(2.0).sum();
    let final_loss_val = final_loss.data()[[0]];

    assert!(
        final_loss_val < initial_loss_val,
        "Loss should decrease after training: {} -> {}",
        initial_loss_val,
        final_loss_val
    );
}
