use tenso_rs::Context;

/// Tests for broadcasting support in backward pass
/// When tensors with different shapes are combined (e.g., [batch, features] + [1, features]),
/// gradients must be summed over broadcasted dimensions.

#[test]
fn test_add_broadcast_forward() {
    let ctx = Context::new();
    // [2, 3] + [1, 3] should broadcast
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = ctx.tensor(&[10.0, 20.0, 30.0], &[1, 3]);

    let c = a + b;
    assert_eq!(c.shape(), vec![2, 3]);
    assert_eq!(
        c.data().as_slice().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
}

#[test]
fn test_add_broadcast_backward() {
    let ctx = Context::new();
    // Simulates: batch [2, 3] + bias [1, 3]
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = ctx.tensor(&[10.0, 20.0, 30.0], &[1, 3]);

    let c = a + b;
    let loss = c.sum();
    loss.backward();

    // grad_a should be all 1s (same shape as a)
    assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
    assert_eq!(
        a.grad().unwrap().as_slice().unwrap(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    // grad_b should be summed over batch dimension: [2.0, 2.0, 2.0]
    assert_eq!(b.grad().unwrap().shape(), &[1, 3]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[2.0, 2.0, 2.0]);
}

#[test]
fn test_sub_broadcast_backward() {
    let ctx = Context::new();
    // Simulates: batch [2, 3] - bias [1, 3]
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = ctx.tensor(&[10.0, 20.0, 30.0], &[1, 3]);

    let c = a - b;
    let loss = c.sum();
    loss.backward();

    // grad_a should be all 1s
    assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
    assert_eq!(
        a.grad().unwrap().as_slice().unwrap(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    // grad_b should be summed and negated: [-2.0, -2.0, -2.0]
    assert_eq!(b.grad().unwrap().shape(), &[1, 3]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[-2.0, -2.0, -2.0]);
}

#[test]
fn test_mul_broadcast_backward() {
    let ctx = Context::new();
    // [2, 3] * [1, 3]
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = ctx.tensor(&[2.0, 3.0, 4.0], &[1, 3]);

    let c = a * b;
    let loss = c.sum();
    loss.backward();

    // grad_a = b broadcasted = [2, 3, 4, 2, 3, 4]
    assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
    assert_eq!(
        a.grad().unwrap().as_slice().unwrap(),
        &[2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
    );

    // grad_b = sum over batch of a = [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_eq!(b.grad().unwrap().shape(), &[1, 3]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_linear_layer_batch_gradient() {
    use tenso_rs::matmul;

    let ctx = Context::new();

    // Simulates a linear layer: y = x @ W + b
    // x: [batch=2, in=3]
    // W: [in=3, out=2]
    // b: [1, out=2]
    let x = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let b = ctx.tensor(&[0.1, 0.2], &[1, 2]);

    let xw = matmul(x, w);
    let y = xw + b;
    let loss = y.sum();
    loss.backward();

    // b gradient should be summed over batch: [2.0, 2.0]
    assert_eq!(b.grad().unwrap().shape(), &[1, 2]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[2.0, 2.0]);
}

#[test]
fn test_batch_size_4() {
    let ctx = Context::new();
    // Larger batch to ensure reduction works correctly
    // [4, 2] + [1, 2]
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    let b = ctx.tensor(&[10.0, 20.0], &[1, 2]);

    let c = a + b;
    let loss = c.sum();
    loss.backward();

    // grad_b should be [4.0, 4.0] (summed over 4 batch samples)
    assert_eq!(b.grad().unwrap().shape(), &[1, 2]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[4.0, 4.0]);
}
