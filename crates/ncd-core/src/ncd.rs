//! NCD matrices. `C(x)` is cached; rows run in parallel, one context per thread.

use std::io;
use std::str::FromStr;

use rayon::prelude::*;

use crate::codec::{Codec, Worker};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Method {
    /// `C(xy)`, upper triangle only.
    #[default]
    Concat,
    /// `(C(xy) + C(yx)) / 2`.
    ConcatSym,
}

impl Method {
    pub fn name(&self) -> &'static str {
        match self {
            Method::Concat => "concat",
            Method::ConcatSym => "concat-sym",
        }
    }
}

impl FromStr for Method {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "concat" => Ok(Method::Concat),
            "concat-sym" | "sym" | "symmetric" => Ok(Method::ConcatSym),
            _ => Err(Error::ParseMethod(s.to_string())),
        }
    }
}

/// Row-major `f32` matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DistMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DistMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::Invalid(format!(
                "matrix data has {} entries, expected {rows}×{cols}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn select(&self, rows: &[usize], cols: &[usize]) -> DistMatrix {
        let mut data = Vec::with_capacity(rows.len() * cols.len());
        for &i in rows {
            let r = self.row(i);
            data.extend(cols.iter().map(|&j| r[j]));
        }
        DistMatrix {
            rows: rows.len(),
            cols: cols.len(),
            data,
        }
    }
}

/// Not clamped: values outside [0, 1] are a compressor diagnostic.
pub fn ncd_from_lens(cx: usize, cy: usize, cxy: usize) -> f64 {
    let (lo, hi) = if cx <= cy { (cx, cy) } else { (cy, cx) };
    if hi == 0 {
        return 0.0;
    }
    (cxy as f64 - lo as f64) / hi as f64
}

pub fn ncd(codec: Codec, x: &[u8], y: &[u8]) -> Result<f64> {
    let mut w = codec.worker()?;
    let cx = w.compressed_len(x)?;
    let cy = w.compressed_len(y)?;
    let cxy = w.concat_len(x, y)?;
    Ok(ncd_from_lens(cx, cy, cxy))
}

pub fn ncd_with(codec: Codec, method: Method, x: &[u8], y: &[u8]) -> Result<f64> {
    let m = cross(codec, &[x], &[y], method)?;
    Ok(m.get(0, 0) as f64)
}

fn clone_err(e: &io::Error) -> Error {
    Error::Io(io::Error::new(e.kind(), e.to_string()))
}

pub fn compressed_lens<T: AsRef<[u8]> + Sync>(codec: Codec, objs: &[T]) -> Result<Vec<usize>> {
    objs.par_iter()
        .map_init(
            || codec.worker(),
            |w, x| match w {
                Ok(w) => Ok(w.compressed_len(x.as_ref())?),
                Err(e) => Err(clone_err(e)),
            },
        )
        .collect()
}

/// Symmetric `n×n` matrix, zero diagonal.
pub fn pairwise<T: AsRef<[u8]> + Sync>(
    codec: Codec,
    objs: &[T],
    method: Method,
) -> Result<DistMatrix> {
    let n = objs.len();
    let lens = compressed_lens(codec, objs)?;
    let mut data = vec![0f32; n * n];

    let rows: Vec<Vec<(usize, f32)>> = (0..n)
        .into_par_iter()
        .map_init(
            || codec.worker(),
            |w, i| -> Result<Vec<(usize, f32)>> {
                let w: &mut Worker = w.as_mut().map_err(|e| clone_err(e))?;
                let x = objs[i].as_ref();
                ((i + 1)..n)
                    .map(|j| {
                        let y = objs[j].as_ref();
                        let mut cxy = w.concat_len(x, y)? as f64;
                        if method == Method::ConcatSym {
                            cxy = (cxy + w.concat_len(y, x)? as f64) / 2.0;
                        }
                        let (lo, hi) = (lens[i].min(lens[j]), lens[i].max(lens[j]));
                        let d = if hi == 0 {
                            0.0
                        } else {
                            (cxy - lo as f64) / hi as f64
                        };
                        Ok((j, d as f32))
                    })
                    .collect()
            },
        )
        .collect::<Result<_>>()?;

    for (i, row) in rows.into_iter().enumerate() {
        for (j, d) in row {
            data[i * n + j] = d;
            data[j * n + i] = d;
        }
    }
    DistMatrix::new(n, n, data)
}

/// `m×n` matrix, queries × refs.
pub fn cross<Q, R>(codec: Codec, queries: &[Q], refs: &[R], method: Method) -> Result<DistMatrix>
where
    Q: AsRef<[u8]> + Sync,
    R: AsRef<[u8]> + Sync,
{
    let (m, n) = (queries.len(), refs.len());
    let lq = compressed_lens(codec, queries)?;
    let lr = compressed_lens(codec, refs)?;

    let rows: Vec<Vec<f32>> = (0..m)
        .into_par_iter()
        .map_init(
            || codec.worker(),
            |w, i| -> Result<Vec<f32>> {
                let w: &mut Worker = w.as_mut().map_err(|e| clone_err(e))?;
                let q = queries[i].as_ref();
                (0..n)
                    .map(|j| {
                        let r = refs[j].as_ref();
                        let mut cqr = w.concat_len(q, r)? as f64;
                        if method == Method::ConcatSym {
                            cqr = (cqr + w.concat_len(r, q)? as f64) / 2.0;
                        }
                        let (lo, hi) = (lq[i].min(lr[j]), lq[i].max(lr[j]));
                        let d = if hi == 0 {
                            0.0
                        } else {
                            (cqr - lo as f64) / hi as f64
                        };
                        Ok(d as f32)
                    })
                    .collect()
            },
        )
        .collect::<Result<_>>()?;
    DistMatrix::new(m, n, rows.concat())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn formula() {
        assert_eq!(ncd_from_lens(10, 10, 10), 0.0);
        assert_eq!(ncd_from_lens(10, 20, 30), 1.0);
        assert_eq!(ncd_from_lens(0, 0, 0), 0.0);
    }

    #[test]
    fn identical_vs_unrelated() {
        let a = noise(1, 20_000);
        let b = noise(2, 20_000);
        for codec in Codec::all_defaults() {
            let same = ncd(codec, &a, &a).unwrap();
            let diff = ncd(codec, &a, &b).unwrap();
            // bzip2 NCD(x, x) is ~0.22 on random data.
            let tol = if matches!(codec, Codec::Bzip2 { .. }) {
                0.3
            } else {
                0.05
            };
            assert!(same < tol, "{codec}: NCD(x,x) = {same}");
            assert!(diff > 0.9, "{codec}: NCD(x,y) = {diff}");
        }
    }

    #[test]
    fn deflate_window_pitfall() {
        // 64 KiB exceeds DEFLATE's 32 KiB window.
        let a = noise(7, 64 << 10);
        assert!(ncd(Codec::Deflate { level: 9 }, &a, &a).unwrap() > 0.9);
        assert!(ncd(Codec::Zstd { level: 19 }, &a, &a).unwrap() < 0.05);
    }

    #[test]
    fn matrix_matches_single_pairs() {
        let objs: Vec<Vec<u8>> = (0..6).map(|i| noise(i, 3000 + 500 * i as usize)).collect();
        let codec = Codec::Zstd { level: 3 };
        let m = pairwise(codec, &objs, Method::Concat).unwrap();
        for i in 0..6 {
            assert_eq!(m.get(i, i), 0.0);
            for j in (i + 1)..6 {
                let d = ncd(codec, &objs[i], &objs[j]).unwrap() as f32;
                assert!((m.get(i, j) - d).abs() < 1e-6);
                assert_eq!(m.get(i, j), m.get(j, i));
            }
        }
    }
}
