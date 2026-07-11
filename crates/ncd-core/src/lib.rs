//! Normalized Compression Distance:
//! `NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`,
//! a computable approximation of the normalized information distance.

pub mod codec;
pub mod ncd;

pub use codec::Codec;
pub use ncd::{DistMatrix, Method};

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
