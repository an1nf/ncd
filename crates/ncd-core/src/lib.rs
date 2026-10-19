//! # ncd-core
//!
//! Normalized Compression Distance (NCD) for threat classification.
//!
//! The Kolmogorov complexity `K(x)` is uncomputable, but any real compressor
//! `C` gives an upper bound `C(x) >= K(x)` (up to an additive constant). The
//! normalized information distance
//!
//! ```text
//! NID(x, y) = max{K(x|y), K(y|x)} / max{K(x), K(y)}
//! ```
//!
//! is universal among normalized admissible distances (Li, Chen, Li, Ma,
//! Vitányi, 2004). Replacing `K` by `C` and using `C(xy) - C(x) ≈ K(y|x)`
//! yields the computable NCD (Cilibrasi & Vitányi, 2005):
//!
//! ```text
//! NCD(x, y) = (C(xy) - min{C(x), C(y)}) / max{C(x), C(y)}
//! ```
//!
//! This crate provides:
//! * [`codec`] — several real compressors behind one interface, with their
//!   window limits (a compressor only "sees" `x` while compressing `y` inside
//!   `xy` if `|x|` fits in its window).
//! * [`ncd`] — cached, parallel pairwise and cross distance matrices, plus a
//!   conditional variant that uses `x` as a zstd raw-content dictionary to
//!   estimate `C(y|x)` directly.
//! * [`knn`] / [`metrics`] — k-nearest-neighbour classification, anomaly
//!   scores and evaluation metrics.
//! * [`extract`] — defensive extraction of PE/ELF code sections from
//!   untrusted binaries (nothing is ever executed).
//! * [`synth`] — a deterministic synthetic "malware family" generator for
//!   tests and demos.
//! * [`npy`] — a minimal `.npy` writer so matrices can be loaded in NumPy.

pub mod codec;
pub mod extract;
pub mod knn;
pub mod metrics;
pub mod ncd;
pub mod npy;
mod rng;
pub mod synth;

pub use codec::Codec;
pub use extract::View;
pub use ncd::{DistMatrix, Method};

/// Errors produced by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid codec spec `{0}` (expected e.g. `zstd:19`, `xz:9`, `bzip2:9`, `deflate:9`)")]
    ParseCodec(String),
    #[error("invalid view `{0}` (expected `whole`, `code` or `no-overlay`)")]
    ParseView(String),
    #[error("invalid method `{0}` (expected `concat`, `concat-sym` or `conditional`)")]
    ParseMethod(String),
    #[error("method `{method}` is not supported by codec `{codec}`")]
    Unsupported { method: String, codec: String },
    #[error("file {path} is {size} bytes, above the {limit}-byte safety limit")]
    TooLarge { path: String, size: u64, limit: u64 },
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;
