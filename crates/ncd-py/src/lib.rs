//! Python bindings. Bytes are borrowed zero-copy; the GIL is released while computing.

use std::path::PathBuf;

use ncd_core::{Codec, DistMatrix, Method, View, extract, knn, ncd};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

fn to_py(e: ncd_core::Error) -> PyErr {
    match e {
        ncd_core::Error::Io(io) => PyIOError::new_err(io.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn parse(codec: &str, method: &str) -> PyResult<(Codec, Method)> {
    Ok((
        codec.parse().map_err(to_py)?,
        method.parse().map_err(to_py)?,
    ))
}

fn to_array(py: Python<'_>, m: DistMatrix) -> PyResult<Bound<'_, PyArray2<f32>>> {
    let (r, c) = (m.rows(), m.cols());
    let arr = Array2::from_shape_vec((r, c), m.into_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// C(x) in bytes.
#[pyfunction]
#[pyo3(signature = (data, codec = "zstd:19"))]
fn compressed_len(py: Python<'_>, data: PyBackedBytes, codec: &str) -> PyResult<usize> {
    let codec: Codec = codec.parse().map_err(to_py)?;
    py.detach(|| codec.compressed_len(&data))
        .map_err(|e| PyIOError::new_err(e.to_string()))
}

#[pyfunction(name = "ncd")]
#[pyo3(signature = (x, y, codec = "zstd:19", method = "concat"))]
fn ncd_py(
    py: Python<'_>,
    x: PyBackedBytes,
    y: PyBackedBytes,
    codec: &str,
    method: &str,
) -> PyResult<f64> {
    let (codec, method) = parse(codec, method)?;
    py.detach(|| ncd::ncd_with(codec, method, &x, &y))
        .map_err(to_py)
}

/// (n, n) float32, zero diagonal.
#[pyfunction]
#[pyo3(signature = (objs, codec = "zstd:19", method = "concat"))]
fn pairwise<'py>(
    py: Python<'py>,
    objs: Vec<PyBackedBytes>,
    codec: &str,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (codec, method) = parse(codec, method)?;
    let m = py
        .detach(|| ncd::pairwise(codec, &objs, method))
        .map_err(to_py)?;
    to_array(py, m)
}

/// (m, n) float32, queries x refs.
#[pyfunction]
#[pyo3(signature = (queries, refs, codec = "zstd:19", method = "concat"))]
fn cross<'py>(
    py: Python<'py>,
    queries: Vec<PyBackedBytes>,
    refs: Vec<PyBackedBytes>,
    codec: &str,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (codec, method) = parse(codec, method)?;
    let m = py
        .detach(|| ncd::cross(codec, &queries, &refs, method))
        .map_err(to_py)?;
    to_array(py, m)
}

/// (m,) mean NCD to the k nearest refs.
#[pyfunction]
#[pyo3(signature = (queries, refs, k = 5, codec = "zstd:19", method = "concat"))]
fn anomaly_scores<'py>(
    py: Python<'py>,
    queries: Vec<PyBackedBytes>,
    refs: Vec<PyBackedBytes>,
    k: usize,
    codec: &str,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (codec, method) = parse(codec, method)?;
    let s = py
        .detach(|| ncd::cross(codec, &queries, &refs, method).map(|m| knn::anomaly_scores(&m, k)))
        .map_err(to_py)?;
    Ok(PyArray1::from_vec(py, s))
}

/// View `code`, `no-overlay` or `whole`; falls back to the whole file.
#[pyfunction]
#[pyo3(signature = (path, view = "code", max_bytes = None))]
fn load<'py>(
    py: Python<'py>,
    path: PathBuf,
    view: &str,
    max_bytes: Option<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let view: View = view.parse().map_err(to_py)?;
    let ex = py
        .detach(|| extract::load(&path, view, max_bytes))
        .map_err(to_py)?;
    Ok(PyBytes::new(py, &ex.bytes))
}

/// Codec window in bytes.
#[pyfunction]
#[pyo3(signature = (codec = "zstd:19"))]
fn window(codec: &str) -> PyResult<usize> {
    Ok(codec.parse::<Codec>().map_err(to_py)?.window())
}

#[pymodule]
fn ncdthreat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(compressed_len, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_py, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(anomaly_scores, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(window, m)?)?;
    Ok(())
}
