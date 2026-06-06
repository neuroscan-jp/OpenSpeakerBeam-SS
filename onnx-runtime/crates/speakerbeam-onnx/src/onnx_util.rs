use std::path::Path;

use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

pub fn open_session(path: &Path, threads: usize) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads.max(1))?
        .commit_from_file(path)
}

pub fn tensor_from_array3(array: Array3<f32>) -> ort::Result<Tensor<f32>> {
    Tensor::from_array(array)
}

pub fn tensor_from_array2(array: Array2<f32>) -> ort::Result<Tensor<f32>> {
    Tensor::from_array(array)
}

pub fn tensor_to_vec(tensor: &ort::value::DynValue) -> ort::Result<Vec<f32>> {
    let arr = tensor.try_extract_array::<f32>()?;
    Ok(arr.iter().copied().collect())
}
