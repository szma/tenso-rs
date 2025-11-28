use tenso_rs::{Context, matmul};

#[test]
fn test_tensor_creation() {
    let ctx = Context::new();
    let t = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(t.data().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_add() {
    let ctx = Context::new();
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = ctx.tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let c = a + b;
    assert_eq!(c.data().as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_tensor_sub() {
    let ctx = Context::new();
    let a = ctx.tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let b = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = a - b;
    assert_eq!(c.data().as_slice().unwrap(), &[4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_tensor_mul() {
    let ctx = Context::new();
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = ctx.tensor(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);

    let c = a * b;
    assert_eq!(c.data().as_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_matmul() {
    let ctx = Context::new();
    // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
    // [3, 4] @ [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = ctx.tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let c = matmul(a, b);
    assert_eq!(c.data().as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_relu() {
    let ctx = Context::new();
    let a = ctx.tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);

    let b = a.relu();
    assert_eq!(b.data().as_slice().unwrap(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sum() {
    let ctx = Context::new();
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let s = a.sum();
    assert_eq!(s.data().as_slice().unwrap(), &[10.0]);
}

#[test]
fn test_mean() {
    let ctx = Context::new();
    let a = ctx.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let m = a.mean();
    assert_eq!(m.data().as_slice().unwrap(), &[2.5]);
}

#[test]
fn test_pow() {
    let ctx = Context::new();
    let a = ctx.tensor(&[1.0, 2.0, 3.0], &[3]);

    let b = a.pow(2.0);
    assert_eq!(b.data().as_slice().unwrap(), &[1.0, 4.0, 9.0]);
}

#[test]
fn test_backward_simple() {
    let ctx = Context::new();
    let a = ctx.tensor(&[2.0, 3.0], &[2]);
    let b = ctx.tensor(&[4.0, 5.0], &[2]);

    let c = a + b;
    c.backward();

    // d(c)/d(a) = 1, d(c)/d(b) = 1
    assert_eq!(a.grad().unwrap().as_slice().unwrap(), &[1.0, 1.0]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[1.0, 1.0]);
}

#[test]
fn test_backward_mul() {
    let ctx = Context::new();
    let a = ctx.tensor(&[2.0, 3.0], &[2]);
    let b = ctx.tensor(&[4.0, 5.0], &[2]);

    let c = a * b;
    c.backward();

    // d(c)/d(a) = b, d(c)/d(b) = a
    assert_eq!(a.grad().unwrap().as_slice().unwrap(), &[4.0, 5.0]);
    assert_eq!(b.grad().unwrap().as_slice().unwrap(), &[2.0, 3.0]);
}

#[test]
fn test_backward_chain() {
    let ctx = Context::new();
    let x = ctx.tensor(&[2.0], &[1]);

    // y = x^2 => dy/dx = 2x = 4
    let y = x.pow(2.0);
    y.backward();

    assert_eq!(x.grad().unwrap().as_slice().unwrap(), &[4.0]);
}

#[test]
fn test_zero_grad() {
    let ctx = Context::new();
    let a = ctx.tensor(&[2.0], &[1]);
    let b = ctx.tensor(&[3.0], &[1]);

    let c = a + b;
    c.backward();

    assert!(a.grad().is_some());

    ctx.zero_grad();

    assert!(a.grad().is_none());
}

#[test]
fn test_context_len_and_prune() {
    let ctx = Context::new();

    assert!(ctx.is_empty());
    assert_eq!(ctx.len(), 0);

    let _a = ctx.tensor(&[1.0], &[1]);
    let _b = ctx.tensor(&[2.0], &[1]);

    assert_eq!(ctx.len(), 2);

    let pruned = ctx.prune(1);
    assert_eq!(pruned, 1);
    assert_eq!(ctx.len(), 1);
}
