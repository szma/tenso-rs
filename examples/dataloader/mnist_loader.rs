use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub struct MnistData {
    pub images: Vec<Vec<f64>>, // each image is 784 floats (0.0-1.0)
    pub labels: Vec<u8>,       // 0-9
}

impl MnistData {
    pub fn load(images_path: &Path, labels_path: &Path) -> std::io::Result<Self> {
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;
        Ok(Self { images, labels })
    }

    fn load_images(path: &Path) -> std::io::Result<Vec<Vec<f64>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let magic = read_u32_be(&mut reader)?;
        assert_eq!(magic, 2051, "Invalid image file magic number");

        let num_images = read_u32_be(&mut reader)? as usize;
        let rows = read_u32_be(&mut reader)? as usize;
        let cols = read_u32_be(&mut reader)? as usize;
        let pixels_per_image = rows * cols;

        let mut images = Vec::with_capacity(num_images);
        let mut buffer = vec![0u8; pixels_per_image];

        for _ in 0..num_images {
            reader.read_exact(&mut buffer)?;
            let image: Vec<f64> = buffer.iter().map(|&b| b as f64 / 255.0).collect();
            images.push(image);
        }

        Ok(images)
    }

    fn load_labels(path: &Path) -> std::io::Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let magic = read_u32_be(&mut reader)?;
        assert_eq!(magic, 2049, "Invalid label file magic number");

        let num_labels = read_u32_be(&mut reader)? as usize;

        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        Ok(labels)
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

fn read_u32_be<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}
