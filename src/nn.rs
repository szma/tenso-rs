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

/// Single-head Transformer Block with self-attention and feedforward network
pub struct TransformerBlock<'a> {
    // Attention projections
    q_proj: Linear<'a>,
    k_proj: Linear<'a>,
    v_proj: Linear<'a>,
    out_proj: Linear<'a>,

    // Feedforward network
    ffn1: Linear<'a>,
    ffn2: Linear<'a>,

    // Scaling factor for attention: 1/sqrt(d_model)
    scale: f32,
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

impl<'a> TransformerBlock<'a> {
    /// Creates a new Transformer block
    /// - d_model: dimension of the model (input/output size)
    /// - d_ff: dimension of the feedforward hidden layer (typically 4 * d_model)
    pub fn new(ctx: &'a Context, d_model: usize, d_ff: usize) -> Self {
        Self {
            q_proj: Linear::new(ctx, d_model, d_model),
            k_proj: Linear::new(ctx, d_model, d_model),
            v_proj: Linear::new(ctx, d_model, d_model),
            out_proj: Linear::new(ctx, d_model, d_model),
            ffn1: Linear::new(ctx, d_model, d_ff),
            ffn2: Linear::new(ctx, d_ff, d_model),
            scale: 1.0 / (d_model as f32).sqrt(),
        }
    }

    /// Forward pass through the transformer block
    /// Input shape: [batch_size, d_model]
    /// Output shape: [batch_size, d_model]
    pub fn forward(&self, x: Tensor<'a>) -> Tensor<'a> {
        // Self-attention
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Attention scores: Q @ K^T / sqrt(d_k)
        let scores = matmul(q, k.transpose()).scale(self.scale);

        // Attention weights (softmax)
        let attn_weights = scores.softmax();

        // Attention output
        let attn_out = matmul(attn_weights, v);
        let attn_out = self.out_proj.forward(attn_out);

        // Residual connection + layer norm
        let x = (x + attn_out).layer_norm(1e-5);

        // Feedforward network
        let ffn_out = self.ffn2.forward(self.ffn1.forward(x).relu());

        // Residual connection + layer norm
        (x + ffn_out).layer_norm(1e-5)
    }

    /// Returns all trainable parameters
    pub fn params(&self) -> Vec<Tensor<'a>> {
        let mut params = Vec::new();
        params.extend(self.q_proj.params());
        params.extend(self.k_proj.params());
        params.extend(self.v_proj.params());
        params.extend(self.out_proj.params());
        params.extend(self.ffn1.params());
        params.extend(self.ffn2.params());
        params
    }
}
