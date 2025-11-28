use tenso_rs::{Context, TransformerBlock};

#[test]
fn test_scale_forward_backward() {
    let ctx = Context::new();
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.scale(2.0);

    assert_eq!(y.data().as_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]);

    y.sum().backward();
    // d/dx (2x) = 2
    let grad = x.grad().unwrap();
    assert_eq!(grad.as_slice().unwrap(), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn test_exp_forward_backward() {
    let ctx = Context::new();
    let x = ctx.tensor(&[0.0, 1.0, 2.0, 0.0], &[2, 2]);
    let y = x.exp();

    let data = y.data();
    let slice = data.as_slice().unwrap();
    assert!((slice[0] - 1.0).abs() < 1e-5); // e^0 = 1
    assert!((slice[1] - std::f32::consts::E).abs() < 1e-5); // e^1 = e
    assert!((slice[2] - std::f32::consts::E.powi(2)).abs() < 1e-4); // e^2

    y.sum().backward();
    // d/dx (e^x) = e^x
    let grad = x.grad().unwrap();
    let grad_slice = grad.as_slice().unwrap();
    assert!((grad_slice[0] - 1.0).abs() < 1e-5);
    assert!((grad_slice[1] - std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn test_softmax_forward() {
    let ctx = Context::new();
    // Row-wise softmax
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    let y = x.softmax();

    let data = y.data();

    // First row: softmax([1, 2, 3])
    let row0_sum: f32 = (0..3).map(|j| data[[0, j]]).sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");

    // Second row: softmax([1, 1, 1]) = [1/3, 1/3, 1/3]
    let row1_sum: f32 = (0..3).map(|j| data[[1, j]]).sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");
    assert!(
        (data[[1, 0]] - 1.0 / 3.0).abs() < 1e-5,
        "Uniform input -> uniform output"
    );
}

#[test]
fn test_softmax_backward() {
    let ctx = Context::new();
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.softmax();
    let loss = y.sum();
    loss.backward();

    // Softmax gradients should exist and not be NaN
    let grad = x.grad().unwrap();
    for &g in grad.as_slice().unwrap() {
        assert!(!g.is_nan(), "Gradient should not be NaN");
    }
}

#[test]
fn test_layer_norm_forward() {
    let ctx = Context::new();
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 4]);
    let y = x.layer_norm(1e-5);

    let data = y.data();

    // Each row should have mean ≈ 0 and std ≈ 1
    for row in 0..2 {
        let mean: f32 = (0..4).map(|j| data[[row, j]]).sum::<f32>() / 4.0;
        let var: f32 = (0..4).map(|j| (data[[row, j]] - mean).powi(2)).sum::<f32>() / 4.0;
        let std = var.sqrt();

        assert!(
            mean.abs() < 1e-5,
            "LayerNorm mean should be ~0, got {}",
            mean
        );
        assert!(
            (std - 1.0).abs() < 1e-4,
            "LayerNorm std should be ~1, got {}",
            std
        );
    }
}

#[test]
fn test_layer_norm_backward() {
    let ctx = Context::new();
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.layer_norm(1e-5);
    let loss = y.sum();
    loss.backward();

    let grad = x.grad().unwrap();
    for &g in grad.as_slice().unwrap() {
        assert!(!g.is_nan(), "Gradient should not be NaN");
        assert!(g.is_finite(), "Gradient should be finite");
    }
}

#[test]
fn test_transpose_forward() {
    let ctx = Context::new();
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.transpose();

    // Shape should be [3, 2]
    assert_eq!(y.shape(), vec![3, 2]);

    // Check values
    let data = y.data();
    assert_eq!(data[[0, 0]], 1.0);
    assert_eq!(data[[0, 1]], 4.0);
    assert_eq!(data[[1, 0]], 2.0);
    assert_eq!(data[[2, 1]], 6.0);
}

#[test]
fn test_transpose_backward() {
    let ctx = Context::new();
    // Use a simple setup: matmul uses transpose internally
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = ctx.tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // a @ b.T tests transpose in chain
    let c = tenso_rs::matmul(a, b.transpose());
    c.mean().backward();

    // Both should have gradients
    assert!(a.grad().is_some(), "a should have gradient");
    assert!(b.grad().is_some(), "b should have gradient");
}

#[test]
fn test_transformer_forward() {
    let ctx = Context::new();
    let d_model = 8;
    let d_ff = 32;
    let batch_size = 4;

    let transformer = TransformerBlock::new(&ctx, d_model, d_ff);

    // Random input
    let input_data: Vec<f32> = (0..(batch_size * d_model))
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let x = ctx.tensor(&input_data, &[batch_size, d_model]);

    let out = transformer.forward(x);

    // Output shape should match input shape
    assert_eq!(out.shape(), vec![batch_size, d_model]);

    // Check no NaNs or Infs
    for &v in out.data().as_slice().unwrap() {
        assert!(!v.is_nan(), "Output should not contain NaN");
        assert!(v.is_finite(), "Output should be finite");
    }
}

#[test]
fn test_transformer_backward() {
    let ctx = Context::new();
    let d_model = 8;
    let d_ff = 32;
    let batch_size = 4;

    let transformer = TransformerBlock::new(&ctx, d_model, d_ff);

    let input_data: Vec<f32> = (0..(batch_size * d_model))
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let x = ctx.tensor(&input_data, &[batch_size, d_model]);

    let out = transformer.forward(x);
    let loss = out.mean();
    loss.backward();

    // Check that at least some parameters have gradients
    let params = transformer.params();
    let mut params_with_grad = 0;
    for param in &params {
        if let Some(grad) = param.grad() {
            params_with_grad += 1;
            for &g in grad.iter() {
                assert!(!g.is_nan(), "Gradient should not be NaN");
                assert!(g.is_finite(), "Gradient should be finite");
            }
        }
    }
    assert!(
        params_with_grad > 0,
        "At least some parameters should have gradients"
    );
    println!(
        "Parameters with gradients: {}/{}",
        params_with_grad,
        params.len()
    );
}
