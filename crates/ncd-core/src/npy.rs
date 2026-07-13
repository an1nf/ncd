//! `.npy` v1.0 writer.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::ncd::DistMatrix;

pub fn write_f32<W: Write>(mut w: W, rows: usize, cols: usize, data: &[f32]) -> io::Result<()> {
    assert_eq!(rows * cols, data.len());
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // header total length must be a multiple of 64
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let header_len = (dict.len() + pad + 1) as u16;
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    w.write_all(&vec![b' '; pad])?;
    w.write_all(b"\n")?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

pub fn save(path: &Path, m: &DistMatrix) -> io::Result<()> {
    write_f32(
        BufWriter::new(File::create(path)?),
        m.rows(),
        m.cols(),
        m.as_slice(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn header_is_aligned() {
        let mut buf = Vec::new();
        super::write_f32(&mut buf, 2, 3, &[0.0; 6]).unwrap();
        let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(buf.len(), 10 + header_len + 24);
        assert_eq!(buf[10 + header_len - 1], b'\n');
    }
}
