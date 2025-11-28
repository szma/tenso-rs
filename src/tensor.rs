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
                        // grad_a += grad
                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &grad;
                        } else {
                            tensors[a.0].grad = Some(grad.clone());
                        }
                        // grad_b += grad
                        if let Some(ref mut g) = tensors[b.0].grad {
                            *g += &grad;
                        } else {
                            tensors[b.0].grad = Some(grad.clone());
                        }
                    }
                    Op::Mul(a, b) => {
                        let a_delta = tensors[b.0].data.clone() * &grad;
                        let b_delta = tensors[a.0].data.clone() * &grad;

                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &a_delta;
                        } else {
                            tensors[a.0].grad = Some(a_delta);
                        }
                        // grad_b += grad * a
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
                    Op::Sub(a, b) => {
                        // grad_a += grad, grad_b -= grad
                        if let Some(ref mut g) = tensors[a.0].grad {
                            *g += &grad;
                        } else {
                            tensors[a.0].grad = Some(grad.clone());
                        }
                        let neg_grad = grad.mapv(|x| -x);
                        if let Some(ref mut g) = tensors[b.0].grad {
                            *g += &neg_grad;
                        } else {
                            tensors[b.0].grad = Some(neg_grad);
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
    Pow(TensorIdx, f32),
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
