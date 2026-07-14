//! Compressors. Only output length is used, so stream codecs write to a counting sink.

use std::fmt;
use std::io::{self, Write};
use std::str::FromStr;

use crate::Error;

/// 128 MiB window (level 19 default is 8 MiB); zstd shrinks it for small inputs.
pub const ZSTD_WINDOW_LOG: u32 = 27;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// Levels 1..=22.
    Zstd { level: i32 },
    /// LZMA2, presets 0..=9, no integrity check.
    Xz { level: u32 },
    /// Levels 1..=9, blocks of 100 kB × level.
    Bzip2 { level: u32 },
    /// Raw DEFLATE, levels 0..=9.
    Deflate { level: u32 },
}

impl Default for Codec {
    fn default() -> Self {
        Codec::Zstd { level: 19 }
    }
}

impl Codec {
    pub fn all_defaults() -> [Codec; 4] {
        [
            Codec::Zstd { level: 19 },
            Codec::Xz { level: 9 },
            Codec::Bzip2 { level: 9 },
            Codec::Deflate { level: 9 },
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Codec::Zstd { .. } => "zstd",
            Codec::Xz { .. } => "xz",
            Codec::Bzip2 { .. } => "bzip2",
            Codec::Deflate { .. } => "deflate",
        }
    }

    /// Max match distance in bytes. Beyond it `C(xy) ≈ C(x) + C(y)` and NCD saturates.
    pub fn window(&self) -> usize {
        match *self {
            Codec::Zstd { .. } => 1 << ZSTD_WINDOW_LOG,
            Codec::Xz { level } => match level {
                0 => 256 << 10,
                1 => 1 << 20,
                2 => 2 << 20,
                3 | 4 => 4 << 20,
                5 | 6 => 8 << 20,
                7 => 16 << 20,
                8 => 32 << 20,
                _ => 64 << 20,
            },
            Codec::Bzip2 { level } => level as usize * 100_000,
            Codec::Deflate { .. } => 32 << 10,
        }
    }

    pub fn supports_conditional(&self) -> bool {
        matches!(self, Codec::Zstd { .. })
    }

    pub fn worker(&self) -> io::Result<Worker> {
        Worker::new(*self)
    }

    pub fn compressed_len(&self, data: &[u8]) -> io::Result<usize> {
        self.worker()?.compressed_len(data)
    }

    fn validate(self) -> Result<Self, Error> {
        let ok = match self {
            Codec::Zstd { level } => (1..=22).contains(&level),
            Codec::Xz { level } => level <= 9,
            Codec::Bzip2 { level } => (1..=9).contains(&level),
            Codec::Deflate { level } => level <= 9,
        };
        if ok {
            Ok(self)
        } else {
            Err(Error::ParseCodec(self.to_string()))
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Codec::Zstd { level } => write!(f, "zstd:{level}"),
            Codec::Xz { level } => write!(f, "xz:{level}"),
            Codec::Bzip2 { level } => write!(f, "bzip2:{level}"),
            Codec::Deflate { level } => write!(f, "deflate:{level}"),
        }
    }
}

impl FromStr for Codec {
    type Err = Error;

    /// `name[:level]`, e.g. `zstd:19`, `xz`, `gzip:9`.
    fn from_str(s: &str) -> Result<Self, Error> {
        let bad = || Error::ParseCodec(s.to_string());
        let (name, level) = match s.split_once(':') {
            Some((n, l)) => (n, Some(l.trim().parse::<i64>().map_err(|_| bad())?)),
            None => (s, None),
        };
        let lvl_u = |default: u32| -> Result<u32, Error> {
            level.map_or(Ok(default), |l| u32::try_from(l).map_err(|_| bad()))
        };
        let codec = match name.trim().to_ascii_lowercase().as_str() {
            "zstd" | "zstandard" => Codec::Zstd {
                level: level.map_or(Ok(19), |l| i32::try_from(l).map_err(|_| bad()))?,
            },
            "xz" | "lzma" | "lzma2" => Codec::Xz { level: lvl_u(9)? },
            "bzip2" | "bz2" => Codec::Bzip2 { level: lvl_u(9)? },
            "deflate" | "gzip" | "zlib" => Codec::Deflate { level: lvl_u(9)? },
            _ => return Err(bad()),
        };
        codec.validate().map_err(|_| bad())
    }
}

#[derive(Default)]
struct Counter(u64);

impl Write for Counter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn zstd_compressor<'a>(
    level: i32,
    dict: Option<&'a zstd::dict::EncoderDictionary<'a>>,
) -> io::Result<zstd::bulk::Compressor<'a>> {
    let mut c = match dict {
        Some(d) => zstd::bulk::Compressor::with_prepared_dictionary(d)?,
        None => zstd::bulk::Compressor::new(level)?,
    };
    c.set_parameter(zstd::zstd_safe::CParameter::WindowLog(ZSTD_WINDOW_LOG))?;
    // Constant frame overhead adds noise to NCD.
    c.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(false))?;
    c.set_parameter(zstd::zstd_safe::CParameter::ContentSizeFlag(false))?;
    c.set_parameter(zstd::zstd_safe::CParameter::DictIdFlag(false))?;
    Ok(c)
}

/// Per-thread compression context.
pub struct Worker {
    codec: Codec,
    zstd: Option<zstd::bulk::Compressor<'static>>,
    out: Vec<u8>,
    concat: Vec<u8>,
}

impl Worker {
    pub fn new(codec: Codec) -> io::Result<Self> {
        let zstd = match codec {
            Codec::Zstd { level } => Some(zstd_compressor(level, None)?),
            _ => None,
        };
        Ok(Self {
            codec,
            zstd,
            out: Vec::new(),
            concat: Vec::new(),
        })
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// `C(x)`
    pub fn compressed_len(&mut self, data: &[u8]) -> io::Result<usize> {
        compress_len(self.codec, self.zstd.as_mut(), &mut self.out, data)
    }

    /// `C(xy)`
    pub fn concat_len(&mut self, x: &[u8], y: &[u8]) -> io::Result<usize> {
        self.concat.clear();
        self.concat.reserve(x.len() + y.len());
        self.concat.extend_from_slice(x);
        self.concat.extend_from_slice(y);
        compress_len(self.codec, self.zstd.as_mut(), &mut self.out, &self.concat)
    }
}

pub(crate) fn zstd_len(
    c: &mut zstd::bulk::Compressor<'_>,
    out: &mut Vec<u8>,
    data: &[u8],
) -> io::Result<usize> {
    out.clear();
    out.reserve(zstd::zstd_safe::compress_bound(data.len()));
    c.compress_to_buffer(data, out)
}

fn compress_len(
    codec: Codec,
    zstd: Option<&mut zstd::bulk::Compressor<'static>>,
    out: &mut Vec<u8>,
    data: &[u8],
) -> io::Result<usize> {
    match codec {
        Codec::Zstd { .. } => {
            let c = zstd.expect("zstd worker always holds a context");
            zstd_len(c, out, data)
        }
        Codec::Xz { level } => {
            // Cap dict size to the input: same output, avoids 16-64 MiB allocs per call.
            let mut opts = xz2::stream::LzmaOptions::new_preset(level).map_err(io::Error::other)?;
            let need = data.len().max(4096).next_power_of_two().min(codec.window());
            opts.dict_size(need as u32);
            let mut filters = xz2::stream::Filters::new();
            filters.lzma2(&opts);
            let stream =
                xz2::stream::Stream::new_stream_encoder(&filters, xz2::stream::Check::None)
                    .map_err(io::Error::other)?;
            let mut enc = xz2::write::XzEncoder::new_stream(Counter::default(), stream);
            enc.write_all(data)?;
            Ok(enc.finish()?.0 as usize)
        }
        Codec::Bzip2 { level } => {
            let mut enc =
                bzip2::write::BzEncoder::new(Counter::default(), bzip2::Compression::new(level));
            enc.write_all(data)?;
            Ok(enc.finish()?.0 as usize)
        }
        Codec::Deflate { level } => {
            let mut enc = flate2::write::DeflateEncoder::new(
                Counter::default(),
                flate2::Compression::new(level),
            );
            enc.write_all(data)?;
            Ok(enc.finish()?.0 as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        for s in ["zstd:19", "xz:9", "bzip2:9", "deflate:9", "zstd:1"] {
            assert_eq!(s.parse::<Codec>().unwrap().to_string(), s);
        }
        assert_eq!(
            "gzip".parse::<Codec>().unwrap(),
            Codec::Deflate { level: 9 }
        );
        assert_eq!("zstd".parse::<Codec>().unwrap(), Codec::Zstd { level: 19 });
        for bad in [
            "zstd:0", "zstd:23", "bzip2:0", "xz:10", "lz4", "zstd:x", "xz:-1",
        ] {
            assert!(bad.parse::<Codec>().is_err(), "{bad} should fail");
        }
    }

    #[test]
    fn redundancy_compresses() {
        let data = b"abcdefgh".repeat(4096);
        for codec in Codec::all_defaults() {
            let c = codec.compressed_len(&data).unwrap();
            assert!(c < data.len() / 20, "{codec}: {c}");
        }
    }

    #[test]
    fn worker_reuse_is_deterministic() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i * 7 % 251) as u8).collect();
        for codec in Codec::all_defaults() {
            let mut w = codec.worker().unwrap();
            let a = w.compressed_len(&data).unwrap();
            let _ = w.concat_len(&data, b"zzz").unwrap();
            let b = w.compressed_len(&data).unwrap();
            assert_eq!(a, b, "{codec}");
        }
    }
}
