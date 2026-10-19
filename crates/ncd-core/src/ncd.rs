//! Normalized Compression Distance: single pairs, pairwise and cross matrices.
//!
//! Every `C(x)` is computed once and cached; only the `C(xy)` (or `C(y|x)`)
//! terms are computed per pair. Rows are distributed over threads with rayon,
//! each thread owning its own compression context.

use std::io;
use std::str::FromStr;

use rayon::prelude::*;

use crate::codec::{Codec, Worker, zstd_compressor, zstd_len};
use crate::{Error, Result};

/// How the joint/conditional complexity term is estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Method {
    /// `C(xy)` for the upper triangle only. Real compressors are not exactly
    /// symmetric, so `NCD(x,y)` and `NCD(y,x)` can differ slightly.
    #[default]
    Concat,
    /// `(C(xy) + C(yx)) / 2` — symmetric by construction, twice the cost.
    ConcatSym,
    /// zstd only: compress the more complex object using the simpler one as
    /// a raw-content dictionary, estimating `max{K(x|y), K(y|x)}` directly
    /// (the numerator of the normalized information distance). The
    /// dictionary is prepared once per object and reused across a whole row.
    Conditional,
}

impl Method {
    pub fn name(&self) -> &'static str {
        match self {
            Method::Concat => "concat",
            Method::ConcatSym => "concat-sym",
            Method::Conditional => "conditional",
        }
    }
}

impl FromStr for Method {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "concat" => Ok(Method::Concat),
            "concat-sym" | "sym" | "symmetric" => Ok(Method::ConcatSym),
            "conditional" | "cond" | "dict" => Ok(Method::Conditional),
            _ => Err(Error::ParseMethod(s.to_string())),
        }
    }
}

/// Dense row-major `f32` distance matrix.
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

    /// Selects a sub-matrix (e.g. train×train out of a full pairwise matrix).
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

/// `NCD = (C(xy) − min{C(x), C(y)}) / max{C(x), C(y)}`.
///
/// Values are not clamped: slightly negative or >1 results are possible with
/// real compressors and are a useful diagnostic of compressor quality.
pub fn ncd_from_lens(cx: usize, cy: usize, cxy: usize) -> f64 {
    let (lo, hi) = if cx <= cy { (cx, cy) } else { (cy, cx) };
    if hi == 0 {
        return 0.0;
    }
    (cxy as f64 - lo as f64) / hi as f64
}

/// Distance between two objects with the default [`Method::Concat`].
pub fn ncd(codec: Codec, x: &[u8], y: &[u8]) -> Result<f64> {
    let mut w = codec.worker()?;
    let cx = w.compressed_len(x)?;
    let cy = w.compressed_len(y)?;
    let cxy = w.concat_len(x, y)?;
    Ok(ncd_from_lens(cx, cy, cxy))
}

/// Distance between two objects with an explicit [`Method`].
pub fn ncd_with(codec: Codec, method: Method, x: &[u8], y: &[u8]) -> Result<f64> {
    let m = cross(codec, &[x], &[y], method)?;
    Ok(m.get(0, 0) as f64)
}

fn clone_err(e: &io::Error) -> Error {
    Error::Io(io::Error::new(e.kind(), e.to_string()))
}

/// `C(x)` for every object, in parallel.
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

fn check_method(codec: Codec, method: Method) -> Result<()> {
    if method == Method::Conditional && !codec.supports_conditional() {
        return Err(Error::Unsupported {
            method: method.name().into(),
            codec: codec.to_string(),
        });
    }
    Ok(())
}

fn zstd_level(codec: Codec) -> i32 {
    match codec {
        Codec::Zstd { level } => level,
        _ => unreachable!("checked by check_method"),
    }
}

/// `C(target | anchor)` for every target, with `anchor` as a prepared
/// raw-content zstd dictionary.
fn conditional_row<'t>(
    level: i32,
    anchor: &[u8],
    targets: impl Iterator<Item = (usize, &'t [u8])>,
) -> io::Result<Vec<(usize, usize)>> {
    let dict = zstd::dict::EncoderDictionary::copy(anchor, level);
    let mut c = zstd_compressor(level, Some(&dict))?;
    let mut out = Vec::new();
    targets
        .map(|(j, t)| Ok((j, zstd_len(&mut c, &mut out, t)?)))
        .collect()
}

/// Strict total order on objects by (complexity, index): the "anchor" of a
/// pair is the simpler object, whose dictionary conditions the other.
fn simpler(ca: usize, a: usize, cb: usize, b: usize) -> bool {
    (ca, a) < (cb, b)
}

/// Full symmetric `n×n` NCD matrix. The diagonal is set to 0.
pub fn pairwise<T: AsRef<[u8]> + Sync>(
    codec: Codec,
    objs: &[T],
    method: Method,
) -> Result<DistMatrix> {
    check_method(codec, method)?;
    let n = objs.len();
    let lens = compressed_lens(codec, objs)?;
    let mut data = vec![0f32; n * n];

    // Each row yields (j, distance) for the pairs it owns.
    let rows: Vec<Vec<(usize, f32)>> = match method {
        Method::Concat | Method::ConcatSym => (0..n)
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
            .collect::<Result<_>>()?,
        Method::Conditional => {
            let level = zstd_level(codec);
            (0..n)
                .into_par_iter()
                .map(|i| -> Result<Vec<(usize, f32)>> {
                    let targets = (0..n)
                        .filter(|&j| simpler(lens[i], i, lens[j], j))
                        .map(|j| (j, objs[j].as_ref()));
                    let row = conditional_row(level, objs[i].as_ref(), targets)?;
                    Ok(row
                        .into_iter()
                        .map(|(j, c)| (j, ratio(c, lens[j])))
                        .collect())
                })
                .collect::<Result<_>>()?
        }
    };

    for (i, row) in rows.into_iter().enumerate() {
        for (j, d) in row {
            data[i * n + j] = d;
            data[j * n + i] = d;
        }
    }
    DistMatrix::new(n, n, data)
}

fn ratio(num: usize, den: usize) -> f32 {
    if den == 0 {
        0.0
    } else {
        (num as f64 / den as f64) as f32
    }
}

/// `m×n` matrix of distances from each query to each reference.
pub fn cross<Q, R>(codec: Codec, queries: &[Q], refs: &[R], method: Method) -> Result<DistMatrix>
where
    Q: AsRef<[u8]> + Sync,
    R: AsRef<[u8]> + Sync,
{
    check_method(codec, method)?;
    let (m, n) = (queries.len(), refs.len());
    let lq = compressed_lens(codec, queries)?;
    let lr = compressed_lens(codec, refs)?;

    match method {
        Method::Concat | Method::ConcatSym => {
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
        Method::Conditional => {
            let level = zstd_level(codec);
            let mut data = vec![0f32; m * n];
            // Pass 1: queries as anchors, conditioning the more complex refs.
            // Ties go to the query side so that every pair is computed once.
            let by_query: Vec<Vec<(usize, usize)>> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let targets = (0..n)
                        .filter(|&j| lq[i] <= lr[j])
                        .map(|j| (j, refs[j].as_ref()));
                    conditional_row(level, queries[i].as_ref(), targets)
                })
                .collect::<io::Result<_>>()?;
            for (i, row) in by_query.into_iter().enumerate() {
                for (j, c) in row {
                    data[i * n + j] = ratio(c, lr[j]);
                }
            }
            // Pass 2: refs as anchors, conditioning the more complex queries.
            let by_ref: Vec<Vec<(usize, usize)>> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let targets = (0..m)
                        .filter(|&i| lq[i] > lr[j])
                        .map(|i| (i, queries[i].as_ref()));
                    conditional_row(level, refs[j].as_ref(), targets)
                })
                .collect::<io::Result<_>>()?;
            for (j, col) in by_ref.into_iter().enumerate() {
                for (i, c) in col {
                    data[i * n + j] = ratio(c, lq[i]);
                }
            }
            DistMatrix::new(m, n, data)
        }
    }
}

/// Number of pairs `(a, b)` whose concatenation exceeds `window` bytes.
///
/// With `same = true`, `a` and `b` are the same set and only unordered pairs
/// of distinct elements are counted. Runs in `O(n log n)`.
pub fn window_overflows(window: usize, a: &[usize], b: &[usize], same: bool) -> u64 {
    let mut sb = b.to_vec();
    sb.sort_unstable();
    let mut count: u64 = 0;
    for &x in a {
        // number of y in b with x + y > window
        let threshold = window.saturating_sub(x);
        let idx = sb.partition_point(|&y| y <= threshold);
        count += (sb.len() - idx) as u64;
    }
    if same {
        // remove self-pairs (x + x > window) and halve ordered pairs
        let selfs = a.iter().filter(|&&x| x.saturating_mul(2) > window).count() as u64;
        (count - selfs) / 2
    } else {
        count
    }
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
            // BWT does not "see" a verbatim repeat as cheaply as LZ does:
            // bzip2's self-distance on incompressible data is ≈0.2–0.25, a
            // known deviation from the identity axiom of a metric.
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
        // 64 KiB of noise is beyond DEFLATE's 32 KiB window: even NCD(x, x)
        // saturates, while zstd still recognizes the copy.
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

    #[test]
    fn conditional_cross_matches_pairwise() {
        let base = noise(42, 8000);
        let objs: Vec<Vec<u8>> = (0..5)
            .map(|i| {
                let mut v = base.clone();
                v.extend(noise(100 + i, 1000 * (i as usize + 1)));
                v
            })
            .collect();
        let codec = Codec::Zstd { level: 19 };
        let p = pairwise(codec, &objs, Method::Conditional).unwrap();
        let c = cross(codec, &objs, &objs, Method::Conditional).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!((p.get(i, j) - c.get(i, j)).abs() < 1e-6, "({i},{j})");
                }
            }
        }
        // shared 8 KB prefix ⇒ clearly similar
        assert!(p.get(0, 1) < 0.6, "{}", p.get(0, 1));
        // conditioning a copy on itself costs almost nothing
        assert!(c.get(2, 2) < 0.05, "{}", c.get(2, 2));
    }

    #[test]
    fn conditional_rejected_for_other_codecs() {
        let objs = [b"abc".to_vec()];
        assert!(pairwise(Codec::Xz { level: 6 }, &objs, Method::Conditional).is_err());
    }

    #[test]
    fn overflow_counting() {
        let a = [10, 20, 30, 40];
        // pairs with sum > 50: (20,40), (30,40) ⇒ 2
        assert_eq!(window_overflows(50, &a, &a, true), 2);
        // ordered cross pairs a×[25]: 30+25, 40+25 ⇒ 2
        assert_eq!(window_overflows(50, &a, &[25], false), 2);
        let brute = a
            .iter()
            .enumerate()
            .flat_map(|(i, &x)| a[i + 1..].iter().map(move |&y| (x + y > 45) as u64))
            .sum::<u64>();
        assert_eq!(window_overflows(45, &a, &a, true), brute);
    }
}
