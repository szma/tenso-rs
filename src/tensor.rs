#![allow(dead_code)]
use std::{
    cell::RefCell,
    fmt,
    ops::{Add, Mul, Sub},
};

use ndarray::ArrayD;

#[derive(Debug)]
pub struct Context {
    tensors: RefCell<Vec<TensorData>>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            tensors: RefCell::new(Vec::new()),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn zero_grad(&self) {
        for t in self.tensors.borrow_mut().iter_mut() {
            t.grad = None;
        }
    }

    /// Prune all tensors after the given index, keeping only the first `keep` tensors.
    /// Use this to remove intermediate computation tensors while preserving parameters.
    /// Returns the number of pruned tensors.
    pub fn prune(&self, keep: usize) -> usize {
        let mut tensors = self.tensors.borrow_mut();
        let old_len = tensors.len();
        tensors.truncate(keep);
        old_len - keep
    }

    /// Returns the current number of tensors in the arena.
    pub fn len(&self) -> usize {
        self.tensors.borrow().len()
    }

    /// Returns true if the arena contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.borrow().is_empty()
    }

    pub fn tensor(&self, data: &[f32], shape: &[usize]) -> Tensor<'_> {
        let data = ArrayD::from_shape_vec(shape, data.to_vec()).unwrap();

        let idx = TensorIdx(self.tensors.borrow().len());
        self.tensors.borrow_mut().push(TensorData {
            data,
            grad: None,
            op: Op::None,
        });
        Tensor { idx, ctx: self }
    }

    fn backward(&self, idx: TensorIdx) {
        let mut tensors = self.tensors.borrow_mut();

        let shape = tensors[idx.0].data.shape().to_vec();
        tensors[idx.0].grad = Some(ArrayD::ones(shape));

        for i in (0..=idx.0).rev() {
            let grad = tensors[i].grad.clone();
            if let Some(grad) = grad {
                match tensors[i].op {
                    Op::None => {}
                    Op::Add(a, b) => {
                        // Handle broadcasting: sum gradients over broadcasted dimensions
                        let a_shape = tensors[a.0].data.shape().to_vec();
                        let b_shape = tensors[b.0].data.shape().to_vec();

                        // grad_a: reduce if a was broadcasted
                        let a_grad = if a_shape != grad.shape() {
                            // Sum over the batch dimension (axis 0) if shapes differ
                            grad.sum_axis(ndarray::Axis(0))
                                .insert_axis(ndarray::Axis(0))
                        } else {
                            grad.clone()
                        };
                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_grad;
                        } else {
                            tensors[a.0].grad = Some(a_grad);
                        }

                        // grad_b: reduce if b was broadcasted
                        let b_grad = if b_shape != grad.shape() {
                            grad.sum_axis(ndarray::Axis(0))
                                .insert_axis(ndarray::Axis(0))
                        } else {
                            grad.clone()
                        };
                        if let Some(ref mut g) = tensors[b.0].grad {
                            *g += &b_grad;
                        } else {
                            tensors[b.0].grad = Some(b_grad);
                        }
                    }
                    Op::Mul(a, b) => {
                        // Handle broadcasting: sum gradients over broadcasted dimensions
                        let a_shape = tensors[a.0].data.shape().to_vec();
                        let b_shape = tensors[b.0].data.shape().to_vec();

                        let a_delta_raw = tensors[b.0].data.clone() * &grad;
                        let b_delta_raw = tensors[a.0].data.clone() * &grad;

                        // grad_a += grad * b (reduced if broadcasted)
                        let a_delta = if a_shape != a_delta_raw.shape() {
                            a_delta_raw
                                .sum_axis(ndarray::Axis(0))
                                .insert_axis(ndarray::Axis(0))
                        } else {
                            a_delta_raw
                        };
                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }

                        // grad_b += grad * a (reduced if broadcasted)
                        let b_delta = if b_shape != b_delta_raw.shape() {
                            b_delta_raw
                                .sum_axis(ndarray::Axis(0))
                                .insert_axis(ndarray::Axis(0))
                        } else {
                            b_delta_raw
                        };
                        if let Some(ref mut g) = tensors[b.0].grad {
                            *g += &b_delta;
                        } else {
                            tensors[b.0].grad = Some(b_delta);
                        }
                    }
                    Op::MatMul(a, b) => {
                        let grad_2d = grad.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                        let a_2d = tensors[a.0]
                            .data
                            .view()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap();
                        let b_2d = tensors[b.0]
                            .data
                            .view()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap();

                        // grad_A = grad_C @ B^T
                        let a_delta = grad_2d.dot(&b_2d.t()).into_dyn();
                        // grad_B = A^T @ grad_C
                        let b_delta = a_2d.t().dot(&grad_2d).into_dyn();

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                        if let Some(ref mut g) = tensors[b.0].grad {
                            *g += &b_delta;
                        } else {
                            tensors[b.0].grad = Some(b_delta);
                        }
                    }
                    Op::ReLU(a) => {
                        let a_delta =
                            &tensors[a.0].data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * &grad;

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Sum(a) => {
                        // grad is a scalar, distributed to all elements
                        let scalar = grad[[0]];
                        let a_shape = tensors[a.0].data.shape().to_vec();
                        let a_delta = ArrayD::from_elem(a_shape, scalar);

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Mean(a, n) => {
                        // grad is a scalar, distributed to all elements divided by n
                        let scalar = grad[[0]] / n as f32;
                        let a_shape = tensors[a.0].data.shape().to_vec();
                        let a_delta = ArrayD::from_elem(a_shape, scalar);

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Sub(a, b) => {
                        // Handle broadcasting: sum gradients over broadcasted dimensions
                        let a_shape = tensors[a.0].data.shape().to_vec();
                        let b_shape = tensors[b.0].data.shape().to_vec();

                        // grad_a += grad (reduced if broadcasted)
                        let a_grad = if a_shape != grad.shape() {
                            grad.sum_axis(ndarray::Axis(0))
                                .insert_axis(ndarray::Axis(0))
                        } else {
                            grad.clone()
                        };
                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_grad;
                        } else {
                            tensors[a.0].grad = Some(a_grad);
                        }

                        // grad_b -= grad (reduced if broadcasted)
                        let neg_grad = grad.mapv(|x| -x);
                        let b_grad = if b_shape != neg_grad.shape() {
                            neg_grad
                                .sum_axis(ndarray::Axis(0))
                                .insert_axis(ndarray::Axis(0))
                        } else {
                            neg_grad
                        };
                        if let Some(ref mut g) = tensors[b.0].grad {
                            *g += &b_grad;
                        } else {
                            tensors[b.0].grad = Some(b_grad);
                        }
                    }
                    Op::Pow(a, exp) => {
                        // d/dx(x^n) = n * x^(n-1)
                        let a_delta = &tensors[a.0].data.mapv(|x| exp * x.powf(exp - 1.0)) * &grad;

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Scale(a, scalar) => {
                        // d/dx(x * c) = c
                        let a_delta = grad.mapv(|g| g * scalar);

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Exp(a) => {
                        // d/dx(e^x) = e^x = output
                        let a_delta = &tensors[i].data * &grad;

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Softmax(a) => {
                        // softmax backward: grad_input = softmax * (grad - sum(grad * softmax))
                        let softmax_out = &tensors[i].data;
                        let shape = softmax_out.shape();
                        let rows = shape[0];
                        let cols = shape[1];
                        let mut a_delta = softmax_out.clone();

                        for row in 0..rows {
                            // sum(grad * softmax) for this row
                            let dot: f32 = (0..cols)
                                .map(|j| grad[[row, j]] * softmax_out[[row, j]])
                                .sum();
                            for j in 0..cols {
                                a_delta[[row, j]] = softmax_out[[row, j]] * (grad[[row, j]] - dot);
                            }
                        }

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::LayerNorm(a, eps) => {
                        // LayerNorm backward is complex but well-defined
                        // y = (x - mean) / std
                        // We need to compute dx given dy (grad)
                        let input = &tensors[a.0].data;
                        let shape = input.shape();
                        let rows = shape[0];
                        let cols = shape[1];
                        let n = cols as f32;
                        let mut a_delta = input.clone();

                        for row in 0..rows {
                            // Recompute mean and std for this row
                            let mean: f32 = (0..cols).map(|j| input[[row, j]]).sum::<f32>() / n;
                            let var: f32 = (0..cols)
                                .map(|j| (input[[row, j]] - mean).powi(2))
                                .sum::<f32>()
                                / n;
                            let std = (var + eps).sqrt();

                            // Compute intermediate terms
                            let dy_sum: f32 = (0..cols).map(|j| grad[[row, j]]).sum();
                            let dy_xhat_sum: f32 = (0..cols)
                                .map(|j| grad[[row, j]] * (input[[row, j]] - mean) / std)
                                .sum();

                            for j in 0..cols {
                                let x_hat = (input[[row, j]] - mean) / std;
                                a_delta[[row, j]] = (1.0 / std)
                                    * (grad[[row, j]] - dy_sum / n - x_hat * dy_xhat_sum / n);
                            }
                        }

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                    Op::Transpose(a) => {
                        // Transpose backward: just transpose the gradient back
                        let grad_2d = grad.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                        let a_delta = grad_2d.t().to_owned().into_dyn();

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TensorIdx(usize);

#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    idx: TensorIdx,
    ctx: &'a Context,
}

#[derive(Debug)]
enum Op {
    None,
    Add(TensorIdx, TensorIdx),
    Sub(TensorIdx, TensorIdx),
    Mul(TensorIdx, TensorIdx),
    MatMul(TensorIdx, TensorIdx),
    ReLU(TensorIdx),
    Sum(TensorIdx),
    Mean(TensorIdx, usize), // stores input idx and number of elements
    Pow(TensorIdx, f32),
    Scale(TensorIdx, f32),
    Exp(TensorIdx),
    Softmax(TensorIdx),        // row-wise softmax
    LayerNorm(TensorIdx, f32), // eps for numerical stability
    Transpose(TensorIdx),
}

#[derive(Debug)]
struct TensorData {
    data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    op: Op,
}

impl<'a> Tensor<'a> {
    pub fn shape(&self) -> Vec<usize> {
        self.ctx.tensors.borrow()[self.idx.0].data.shape().to_vec()
    }

    pub fn backward(&self) {
        self.ctx.backward(self.idx);
    }

    pub fn relu(&self) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            tensors[self.idx.0].data.mapv(|x| x.max(0.0))
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::ReLU(self.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }

    pub fn sum(&self) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            let sum = tensors[self.idx.0].data.sum();
            ArrayD::from_elem(vec![1], sum)
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Sum(self.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }

    pub fn mean(&self) -> Tensor<'a> {
        let (result_data, n) = {
            let tensors = self.ctx.tensors.borrow();
            let data = &tensors[self.idx.0].data;
            let n = data.len();
            let mean = data.sum() / n as f32;
            (ArrayD::from_elem(vec![1], mean), n)
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Mean(self.idx, n),
        });

        Tensor { idx, ctx: self.ctx }
    }

    pub fn pow(&self, exp: f32) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            tensors[self.idx.0].data.mapv(|x| x.powf(exp))
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Pow(self.idx, exp),
        });

        Tensor { idx, ctx: self.ctx }
    }

    pub fn scale(&self, scalar: f32) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            tensors[self.idx.0].data.mapv(|x| x * scalar)
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Scale(self.idx, scalar),
        });

        Tensor { idx, ctx: self.ctx }
    }

    pub fn exp(&self) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            tensors[self.idx.0].data.mapv(|x| x.exp())
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Exp(self.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }

    /// Row-wise softmax: softmax(x)_ij = exp(x_ij) / sum_k(exp(x_ik))
    pub fn softmax(&self) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            let data = &tensors[self.idx.0].data;
            // Numerically stable softmax: subtract max per row
            let shape = data.shape();
            let rows = shape[0];
            let cols = shape[1];
            let mut result = data.clone();
            for i in 0..rows {
                let row_max = (0..cols)
                    .map(|j| result[[i, j]])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut row_sum = 0.0;
                for j in 0..cols {
                    result[[i, j]] = (result[[i, j]] - row_max).exp();
                    row_sum += result[[i, j]];
                }
                for j in 0..cols {
                    result[[i, j]] /= row_sum;
                }
            }
            result
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Softmax(self.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }

    /// Layer normalization over the last dimension (features)
    /// Normalizes each row to have mean=0 and std=1
    pub fn layer_norm(&self, eps: f32) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            let data = &tensors[self.idx.0].data;
            let shape = data.shape();
            let rows = shape[0];
            let cols = shape[1];
            let mut result = data.clone();
            for i in 0..rows {
                // Compute mean
                let mean: f32 = (0..cols).map(|j| result[[i, j]]).sum::<f32>() / cols as f32;
                // Compute variance
                let var: f32 = (0..cols)
                    .map(|j| (result[[i, j]] - mean).powi(2))
                    .sum::<f32>()
                    / cols as f32;
                let std = (var + eps).sqrt();
                // Normalize
                for j in 0..cols {
                    result[[i, j]] = (result[[i, j]] - mean) / std;
                }
            }
            result
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::LayerNorm(self.idx, eps),
        });

        Tensor { idx, ctx: self.ctx }
    }

    /// Transpose a 2D tensor (swap rows and columns)
    pub fn transpose(&self) -> Tensor<'a> {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            let data = &tensors[self.idx.0].data;
            let view_2d = data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            view_2d.t().to_owned().into_dyn()
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Transpose(self.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.ctx.tensors.borrow()[self.idx.0].data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.ctx.tensors.borrow()[self.idx.0].grad.clone()
    }

    pub fn set_data(&self, data: ArrayD<f32>) {
        self.ctx.tensors.borrow_mut()[self.idx.0].data = data;
    }
}

impl<'a> Add for Tensor<'a> {
    type Output = Tensor<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            &tensors[self.idx.0].data + &tensors[rhs.idx.0].data
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Add(self.idx, rhs.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }
}

impl<'a> Sub for Tensor<'a> {
    type Output = Tensor<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            &tensors[self.idx.0].data - &tensors[rhs.idx.0].data
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Sub(self.idx, rhs.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }
}

impl<'a> Mul for Tensor<'a> {
    type Output = Tensor<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let result_data = {
            let tensors = self.ctx.tensors.borrow();
            &tensors[self.idx.0].data * &tensors[rhs.idx.0].data
        };

        let mut tensors = self.ctx.tensors.borrow_mut();
        let idx = TensorIdx(tensors.len());
        tensors.push(TensorData {
            data: result_data,
            grad: None,
            op: Op::Mul(self.idx, rhs.idx),
        });

        Tensor { idx, ctx: self.ctx }
    }
}

pub fn matmul<'a>(a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
    let result_data = {
        let tensors = a.ctx.tensors.borrow();
        let a_2d = tensors[a.idx.0]
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let b_2d = tensors[b.idx.0]
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        a_2d.dot(&b_2d).into_dyn()
    };

    let mut tensors = a.ctx.tensors.borrow_mut();
    let idx = TensorIdx(tensors.len());
    tensors.push(TensorData {
        data: result_data,
        grad: None,
        op: Op::MatMul(a.idx, b.idx),
    });

    Tensor { idx, ctx: a.ctx }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tensors = self.tensors.borrow();
        for (i, t) in tensors.iter().enumerate() {
            writeln!(f, "Tensor {}", i)?;
            writeln!(f, "  data:  {:?}", t.data)?;
            if let Some(ref g) = t.grad {
                writeln!(f, "  grad:  {:?}", g)?;
            }
            writeln!(f, "  op:    {:?}", t.op)?;
        }
        Ok(())
    }
}
