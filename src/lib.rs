mod nn;
mod optim;
mod tensor;

pub use nn::{Linear, MLP};
pub use optim::sgd_step;
pub use tensor::{Context, Tensor, matmul};
