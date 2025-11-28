//! Character-Level Language Model Example
//!
//! This example demonstrates a practical application of the TransformerBlock:
//! learning to predict the next character in a sequence.
//!
//! The model learns patterns from a simple text and can then generate new text
//! by repeatedly predicting the most likely next character.

use tenso_rs::{Context, Linear, TransformerBlock, sgd_step};

/// Simple character vocabulary
struct Vocab {
    chars: Vec<char>,
}

impl Vocab {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        Self { chars }
    }

    fn size(&self) -> usize {
        self.chars.len()
    }

    fn char_to_idx(&self, c: char) -> Option<usize> {
        self.chars.iter().position(|&x| x == c)
    }

    fn idx_to_char(&self, idx: usize) -> char {
        self.chars[idx]
    }

    /// One-hot encode a character
    fn one_hot(&self, c: char) -> Vec<f32> {
        let mut vec = vec![0.0; self.size()];
        if let Some(idx) = self.char_to_idx(c) {
            vec[idx] = 1.0;
        }
        vec
    }
}

/// Create training data: (context, next_char) pairs
/// Context is a window of characters, target is the next character
fn create_training_data(text: &str, context_size: usize, vocab: &Vocab) -> (Vec<Vec<f32>>, Vec<usize>) {
    let chars: Vec<char> = text.chars().collect();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..(chars.len() - context_size) {
        // Create input: concatenated one-hot vectors for context window
        let mut input = Vec::new();
        for j in 0..context_size {
            input.extend(vocab.one_hot(chars[i + j]));
        }
        inputs.push(input);

        // Target: index of next character
        if let Some(target_idx) = vocab.char_to_idx(chars[i + context_size]) {
            targets.push(target_idx);
        }
    }

    (inputs, targets)
}

/// Softmax for a slice (used for output probabilities)
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

/// Sample from probability distribution (with temperature)
fn sample(probs: &[f32], temperature: f32) -> usize {
    if temperature < 0.01 {
        // Greedy: pick most likely
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    } else {
        // Temperature sampling
        let scaled: Vec<f32> = probs.iter().map(|p| p.powf(1.0 / temperature)).collect();
        let sum: f32 = scaled.iter().sum();
        let normalized: Vec<f32> = scaled.iter().map(|p| p / sum).collect();

        let r: f32 = rand::random();
        let mut cumsum = 0.0;
        for (i, &p) in normalized.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        normalized.len() - 1
    }
}

/// Cross-entropy loss for a single prediction
fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    let probs = softmax(logits);
    -probs[target].max(1e-10).ln()
}

fn main() {
    let ctx = Context::new();

    // Training text - a simple pattern the model can learn
    let training_text = "hello world hello world hello world hello world \
                         hello world hello world hello world hello world \
                         hi there hi there hi there hi there hi there \
                         hello world hello world hi there hi there ";

    println!("=== Character-Level Language Model ===\n");
    println!("Training text: \"{}...\"", &training_text[..50]);

    // Build vocabulary
    let vocab = Vocab::new(training_text);
    println!("Vocabulary size: {} chars", vocab.size());
    println!("Characters: {:?}\n", vocab.chars);

    // Hyperparameters
    let context_size = 3; // Look at 3 chars to predict the next
    let d_model = vocab.size() * context_size; // Input dimension
    let d_ff = 64; // Feedforward hidden dimension
    let hidden_dim = 32;
    let learning_rate = 0.01;
    let epochs = 50;

    println!("Context size: {} characters", context_size);
    println!("Model dimension: {}", d_model);

    // Create training data
    let (inputs, targets) = create_training_data(training_text, context_size, &vocab);
    println!("Training samples: {}\n", inputs.len());

    // Model architecture:
    // Input (context_size * vocab_size) -> TransformerBlock -> Linear -> vocab_size logits
    let transformer = TransformerBlock::new(&ctx, d_model, d_ff);

    // Project from d_model to hidden, then to vocab size
    let proj1 = Linear::new(&ctx, d_model, hidden_dim);
    let proj2 = Linear::new(&ctx, hidden_dim, vocab.size());

    // Collect all parameters
    let mut params = transformer.params();
    params.extend(proj1.params());
    params.extend(proj2.params());

    let num_params: usize = params.iter().map(|p| p.data().len()).sum();
    println!("Total parameters: {}\n", num_params);

    // Training loop
    println!("Training for {} epochs...\n", epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        // Process each training example with individual updates (online SGD)
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            ctx.zero_grad();

            // Forward pass
            let x = ctx.tensor(input, &[1, d_model]);
            let h = transformer.forward(x);
            let h = proj1.forward(h).relu();
            let logits = proj2.forward(h);

            // Get logits as slice
            let logits_data = logits.data();
            let logits_slice = logits_data.as_slice().unwrap();

            // Compute loss for logging
            total_loss += cross_entropy_loss(logits_slice, target);

            // Check accuracy
            let probs = softmax(logits_slice);
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if predicted == target {
                correct += 1;
            }

            // Create target tensor for gradient computation
            let mut target_vec = vec![0.0; vocab.size()];
            target_vec[target] = 1.0;
            let target_tensor = ctx.tensor(&target_vec, &[1, vocab.size()]);

            // Loss: MSE between softmax output and one-hot target
            let softmax_out = logits.softmax();
            let diff = softmax_out - target_tensor;
            let loss = (diff * diff).sum();
            loss.backward();

            // Update parameters after each sample
            sgd_step(&params, learning_rate);
        }

        let avg_loss = total_loss / inputs.len() as f32;
        let accuracy = 100.0 * correct as f32 / inputs.len() as f32;

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {:3}: loss = {:.4}, accuracy = {:.1}%",
                epoch + 1,
                avg_loss,
                accuracy
            );
        }
    }

    // Text generation
    println!("\n=== Text Generation ===\n");

    let seeds = ["hel", "hi ", "wor", " wo"];

    for seed in seeds {
        print!("Seed '{}' -> \"{}",seed, seed);

        let mut context: Vec<char> = seed.chars().collect();

        // Generate 20 characters
        for _ in 0..20 {
            // Build input from current context
            let mut input = Vec::new();
            for &c in context.iter().rev().take(context_size).collect::<Vec<_>>().iter().rev() {
                input.extend(vocab.one_hot(*c));
            }

            // Pad if context is shorter than context_size
            while input.len() < d_model {
                input.splice(0..0, vec![0.0; vocab.size()]);
            }

            // Forward pass
            let x = ctx.tensor(&input, &[1, d_model]);
            let h = transformer.forward(x);
            let h = proj1.forward(h).relu();
            let logits = proj2.forward(h);

            // Sample next character
            let logits_data = logits.data();
            let logits_slice = logits_data.as_slice().unwrap();
            let probs = softmax(logits_slice);

            // Use low temperature for more deterministic output
            let next_idx = sample(&probs, 0.5);
            let next_char = vocab.idx_to_char(next_idx);

            print!("{}", next_char);
            context.push(next_char);
        }
        println!("\"");
    }

    println!("\n=== Attention Visualization ===\n");
    println!("The transformer learns to attend to relevant parts of the context.");
    println!("For example, after 'hel' it learns to predict 'l' (-> 'hell' -> 'hello')");
    println!("After 'hi ' it learns to predict 't' (-> 'hi t' -> 'hi there')");

    println!("\nDone!");
}