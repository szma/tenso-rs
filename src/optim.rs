use crate::tensor::Tensor;

pub fn sgd_step(params: &[Tensor], lr: f32) {
    for p in params {
        if let Some(grad) = p.grad() {
            let new_data = p.data() - &(grad * lr);
            p.set_data(new_data);
        }
    }
}
