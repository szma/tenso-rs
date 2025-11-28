use tenso_rs::{Context, MLP, sgd_step};

#[path = "dataloader/mnist_loader.rs"]
mod mnist_loader;

fn main() {
    use mnist_loader::MnistData;
    use std::path::Path;

    // Load MNIST data
    let train = MnistData::load(
        Path::new("examples/data/train-images-idx3-ubyte"),
        Path::new("examples/data/train-labels-idx1-ubyte"),
    )
    .expect("Failed to load MNIST training data");

    let test = MnistData::load(
        Path::new("examples/data/t10k-images-idx3-ubyte"),
        Path::new("examples/data/t10k-labels-idx1-ubyte"),
    )
    .expect("Failed to load MNIST test data");

    println!(
        "Loaded {} training, {} test images",
        train.len(),
        test.len()
    );

    let ctx = Context::new();

    // Network: 784 -> 128 -> 10 (batch_size=1, no broadcasting needed)
    let mlp = MLP::new(&ctx, &[784, 16, 10]);
    let params_count = ctx.len();

    let params = mlp.params();
    let total_params: usize = params.iter().map(|p| p.data().len()).sum();
    println!("Model has {} parameter tensors ({} total weights)", params.len(), total_params);

    let lr = 0.1;
    let epochs = 10;
    let num_train_samples = train.len();
    let num_test_samples = test.len();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for i in 0..num_train_samples {
            // Single sample input [1, 784]
            let input_data: Vec<f32> = train.images[i].iter().map(|&x| x as f32).collect();
            let x = ctx.tensor(&input_data, &[1, 784]);

            // One-hot target [1, 10]
            let mut target_data = vec![0.0f32; 10];
            let label = train.labels[i] as usize;
            target_data[label] = 1.0;
            let y_true = ctx.tensor(&target_data, &[1, 10]);

            // Forward pass
            let logits = mlp.forward(x);

            // MSE loss
            let loss = (logits - y_true).pow(2.0).mean();
            total_loss += loss.data()[[0]];

            // Track accuracy
            let logits_data = logits.data();
            let pred = (0..10)
                .max_by(|&a, &b| {
                    logits_data[[0, a]]
                        .partial_cmp(&logits_data[[0, b]])
                        .unwrap()
                })
                .unwrap();
            if pred == label {
                correct += 1;
            }

            // Backward pass
            loss.backward();

            // SGD update
            sgd_step(&mlp.params(), lr);

            // Clear gradients and prune intermediate tensors
            ctx.zero_grad();
            ctx.prune(params_count);
        }

        let accuracy = correct as f64 / num_train_samples as f64 * 100.0;
        println!(
            "Epoch {}: Loss = {:.4}, Train Accuracy = {:.2}%",
            epoch + 1,
            total_loss / num_train_samples as f32,
            accuracy
        );

        // Test accuracy
        let mut test_correct = 0;
        for i in 0..num_test_samples {
            let input_data: Vec<f32> = test.images[i].iter().map(|&x| x as f32).collect();
            let x = ctx.tensor(&input_data, &[1, 784]);

            let logits = mlp.forward(x);
            let logits_data = logits.data();

            let pred = (0..10)
                .max_by(|&a, &b| {
                    logits_data[[0, a]]
                        .partial_cmp(&logits_data[[0, b]])
                        .unwrap()
                })
                .unwrap();
            if pred == test.labels[i] as usize {
                test_correct += 1;
            }

            ctx.prune(params_count);
        }
        println!(
            "         Test Accuracy = {:.2}%",
            test_correct as f64 / num_test_samples as f64 * 100.0
        );
    }
}
