use tenso_rs::tensor::{Context, matmul};
fn main() {
    let ctx = Context::new();
    let w = ctx.tensor(&[1., 2., 3., 4., 5., 6.], &[3, 2]);
    let x = ctx.tensor(&[1., 2.], &[2, 1]);

    let y = matmul(w, x).relu();
    y.backward();

    println!("{}", ctx);
}
