use crate::tensor::{Context, Tensor, matmul};
use rand::Rng;

/// A fully connected (linear) layer: y = x @ W + b
pub struct Linear<'a> {
    pub weight: Tensor<'a>,
    pub bias: Tensor<'a>,
}

impl<'a> Linear<'a> {
    /// Creates a new linear layer with Xavier/Glorot initialization
    pub fn new(ctx: &'a Context, in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot initialization
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let bias_data = vec![0.0; out_features];

        Self {
            weight: ctx.tensor(&weight_data, &[in_features, out_features]),
            bias: ctx.tensor(&bias_data, &[1, out_features]),
        }
    }

    /// Forward pass: y = x @ W + b
    pub fn forward(&self, x: Tensor<'a>) -> Tensor<'a> {
        matmul(x, self.weight) + self.bias
    }

    /// Returns all trainable parameters
    pub fn params(&self) -> Vec<Tensor<'a>> {
        vec![self.weight, self.bias]
    }
}

/// A Multi-Layer Perceptron (MLP) with ReLU activations
pub struct MLP<'a> {
    pub layers: Vec<Linear<'a>>,
}

impl<'a> MLP<'a> {
    /// Creates a new MLP with the given layer sizes
    /// Example: &[2, 4, 1] creates a network with 2 inputs, 4 hidden units, 1 output
    pub fn new(ctx: &'a Context, layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|w| Linear::new(ctx, w[0], w[1]))
            .collect();
        Self { layers }
    }

    /// Forward pass with ReLU activations between layers
    pub fn forward(&self, mut x: Tensor<'a>) -> Tensor<'a> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            // ReLU for all layers except the last
            if i < self.layers.len() - 1 {
                x = x.relu();
            }
        }
        x
    }

    /// Returns all trainable parameters from all layers
    pub fn params(&self) -> Vec<Tensor<'a>> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }
}
