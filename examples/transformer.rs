use tenso_rs::{Context, TransformerBlock, sgd_step};

fn main() {
    let ctx = Context::new();

    let d_model = 16;
    let d_ff = 64;
    let batch_size = 8;

    println!(
        "Creating TransformerBlock with d_model={}, d_ff={}",
        d_model, d_ff
    );
    let transformer = TransformerBlock::new(&ctx, d_model, d_ff);

    // Count parameters
    let num_params: usize = transformer.params().iter().map(|p| p.data().len()).sum();
    println!("Total parameters: {}", num_params);

    // Create random input data
    let input_data: Vec<f32> = (0..(batch_size * d_model))
        .map(|i| ((i as f32 * 0.1).sin() + 0.5) * 0.1)
        .collect();
    let x = ctx.tensor(&input_data, &[batch_size, d_model]);
    println!("Input shape: {:?}", x.shape());

    // Forward pass
    let out = transformer.forward(x);
    println!("Output shape: {:?}", out.shape());

    // Simple "training" loop - minimize mean of output
    // Note: Without pruning, the context grows with each forward pass.
    // In a real application, you'd use separate contexts or implement
    // smarter memory management.
    println!("\nTraining for 5 steps (minimizing output mean):");
    let params = transformer.params();

    for step in 0..5 {
        ctx.zero_grad();

        // Forward pass
        let x = ctx.tensor(&input_data, &[batch_size, d_model]);
        let out = transformer.forward(x);
        let loss = out.mean();
        let loss_val = loss.data()[[0]];

        // Backward pass
        loss.backward();

        // Gradient descent
        sgd_step(&params, 0.01);

        println!("  Step {}: loss = {:.6}", step + 1, loss_val);
    }

    // Final forward pass
    let x = ctx.tensor(&input_data, &[batch_size, d_model]);
    let out = transformer.forward(x);
    println!(
        "\nFinal output sample (first 5 values): {:?}",
        &out.data().as_slice().unwrap()[..5]
    );

    println!("\nTransformer forward and backward pass successful!");
}
