//! Byte views of untrusted PE/ELF files. Bounds-checked, parser under
//! `catch_unwind`, falls back to raw bytes. Nothing is executed.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use goblin::elf::program_header::{PF_X, PT_LOAD};
use goblin::elf::section_header::{SHF_EXECINSTR, SHT_NOBITS};
use goblin::pe::section_table::{IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE};

use crate::{Error, Result};

pub const MAX_FILE_SIZE: u64 = 256 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum View {
    Whole,
    /// Executable sections: PE `CNT_CODE`/`MEM_EXECUTE`, ELF `SHF_EXECINSTR`, else `PF_X` segments.
    #[default]
    Code,
    /// Headers and sections, overlay removed.
    NoOverlay,
}

impl View {
    pub fn name(&self) -> &'static str {
        match self {
            View::Whole => "whole",
            View::Code => "code",
            View::NoOverlay => "no-overlay",
        }
    }
}

impl FromStr for View {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "whole" | "raw" => Ok(View::Whole),
            "code" | "text" => Ok(View::Code),
            "no-overlay" | "nooverlay" => Ok(View::NoOverlay),
            _ => Err(Error::ParseView(s.to_string())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Pe,
    Elf,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct Extracted {
    pub bytes: Vec<u8>,
    pub format: Format,
    /// View unavailable; whole file used.
    pub fell_back: bool,
}

/// `data[off..off + len]`, clamped to EOF.
fn slice_at(data: &[u8], off: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(off).ok()?;
    if start >= data.len() {
        return None;
    }
    let len = usize::try_from(len).unwrap_or(usize::MAX);
    let end = start.saturating_add(len).min(data.len());
    (end > start).then(|| &data[start..end])
}

fn ranges(data: &[u8], view: View) -> (Format, Option<Vec<(u64, u64)>>) {
    match goblin::Object::parse(data) {
        Ok(goblin::Object::PE(pe)) => {
            let secs = &pe.sections;
            let r = match view {
                View::Whole => None,
                View::Code => Some(
                    secs.iter()
                        .filter(|s| {
                            s.characteristics & (IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE) != 0
                        })
                        .map(|s| (s.pointer_to_raw_data as u64, s.size_of_raw_data as u64))
                        .collect(),
                ),
                View::NoOverlay => secs
                    .iter()
                    .map(|s| s.pointer_to_raw_data as u64 + s.size_of_raw_data as u64)
                    .max()
                    .map(|end| vec![(0, end)]),
            };
            (Format::Pe, r)
        }
        Ok(goblin::Object::Elf(elf)) => {
            let r = match view {
                View::Whole => None,
                View::Code => {
                    let secs: Vec<_> = elf
                        .section_headers
                        .iter()
                        .filter(|s| {
                            s.sh_flags & SHF_EXECINSTR as u64 != 0 && s.sh_type != SHT_NOBITS
                        })
                        .map(|s| (s.sh_offset, s.sh_size))
                        .collect();
                    if secs.is_empty() {
                        // No section headers: use executable segments.
                        Some(
                            elf.program_headers
                                .iter()
                                .filter(|p| p.p_type == PT_LOAD && p.p_flags & PF_X != 0)
                                .map(|p| (p.p_offset, p.p_filesz))
                                .collect(),
                        )
                    } else {
                        Some(secs)
                    }
                }
                View::NoOverlay => elf
                    .program_headers
                    .iter()
                    .map(|p| p.p_offset.saturating_add(p.p_filesz))
                    .chain(
                        elf.section_headers
                            .iter()
                            .filter(|s| s.sh_type != SHT_NOBITS)
                            .map(|s| s.sh_offset.saturating_add(s.sh_size)),
                    )
                    .max()
                    .map(|end| vec![(0, end)]),
            };
            (Format::Elf, r)
        }
        _ => (Format::Unknown, None),
    }
}

/// Never panics.
pub fn extract(data: &[u8], view: View) -> Extracted {
    let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ranges(data, view)));
    let (format, ranges) = parsed.unwrap_or((Format::Unknown, None));
    if view == View::Whole {
        return Extracted {
            bytes: data.to_vec(),
            format,
            fell_back: false,
        };
    }
    let mut bytes = Vec::new();
    for (off, len) in ranges.unwrap_or_default() {
        if let Some(s) = slice_at(data, off, len) {
            bytes.extend_from_slice(s);
        }
    }
    if bytes.is_empty() {
        Extracted {
            bytes: data.to_vec(),
            format,
            fell_back: true,
        }
    } else {
        Extracted {
            bytes,
            format,
            fell_back: false,
        }
    }
}

pub fn load(path: &Path, view: View, max_bytes: Option<usize>) -> Result<Extracted> {
    let f = File::open(path)?;
    let size = f.metadata()?.len();
    if size > MAX_FILE_SIZE {
        return Err(Error::TooLarge {
            path: path.display().to_string(),
            size,
            limit: MAX_FILE_SIZE,
        });
    }
    let mut data = Vec::with_capacity(size as usize);
    f.take(MAX_FILE_SIZE + 1).read_to_end(&mut data)?;
    let mut ex = extract(&data, view);
    if let Some(m) = max_bytes {
        ex.bytes.truncate(m);
    }
    Ok(ex)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PE32 with `.text`, `.data` and an overlay.
    pub(crate) fn tiny_pe() -> Vec<u8> {
        let mut f = vec![0u8; 0x400];
        f[0..2].copy_from_slice(b"MZ");
        f[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes()); // e_lfanew
        let pe = 0x80;
        f[pe..pe + 4].copy_from_slice(b"PE\0\0");
        let coff = pe + 4;
        f[coff..coff + 2].copy_from_slice(&0x14cu16.to_le_bytes()); // i386
        f[coff + 2..coff + 4].copy_from_slice(&2u16.to_le_bytes()); // sections
        f[coff + 16..coff + 18].copy_from_slice(&0xe0u16.to_le_bytes()); // opt hdr size
        f[coff + 18..coff + 20].copy_from_slice(&0x0102u16.to_le_bytes());
        let opt = coff + 20;
        f[opt..opt + 2].copy_from_slice(&0x10bu16.to_le_bytes()); // PE32 magic
        f[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // section align
        f[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes()); // file align
        f[opt + 56..opt + 60].copy_from_slice(&0x3000u32.to_le_bytes()); // image size
        f[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes()); // headers size
        f[opt + 92..opt + 96].copy_from_slice(&16u32.to_le_bytes()); // data dirs
        let sec = opt + 0xe0;
        let mut put_section = |at: usize, name: &[u8], va: u32, raw: u32, chars: u32| {
            f[at..at + name.len()].copy_from_slice(name);
            f[at + 8..at + 12].copy_from_slice(&0x100u32.to_le_bytes()); // vsize
            f[at + 12..at + 16].copy_from_slice(&va.to_le_bytes());
            f[at + 16..at + 20].copy_from_slice(&0x100u32.to_le_bytes()); // raw size
            f[at + 20..at + 24].copy_from_slice(&raw.to_le_bytes());
            f[at + 36..at + 40].copy_from_slice(&chars.to_le_bytes());
        };
        put_section(sec, b".text", 0x1000, 0x200, 0x6000_0020);
        put_section(sec + 40, b".data", 0x2000, 0x300, 0xC000_0040);
        f[0x200..0x300].fill(0xCC); // code
        f[0x300..0x400].fill(0xDD); // data
        f.extend(std::iter::repeat_n(0xEE, 0x80)); // overlay
        f
    }

    #[test]
    fn pe_views() {
        let pe = tiny_pe();
        let code = extract(&pe, View::Code);
        assert_eq!(code.format, Format::Pe);
        assert!(!code.fell_back);
        assert_eq!(code.bytes, vec![0xCC; 0x100]);

        let no = extract(&pe, View::NoOverlay);
        assert_eq!(no.bytes.len(), 0x400);
        assert_eq!(extract(&pe, View::Whole).bytes.len(), 0x480);
    }

    #[test]
    fn non_executable_falls_back() {
        let data = b"just some text, not an executable".to_vec();
        let ex = extract(&data, View::Code);
        assert!(ex.fell_back);
        assert_eq!(ex.format, Format::Unknown);
        assert_eq!(ex.bytes, data);
    }

    #[test]
    fn hostile_offsets_are_clamped() {
        let mut pe = tiny_pe();
        // .text raw pointer far past EOF, .data raw size huge
        let sec = 0x80 + 4 + 20 + 0xe0;
        pe[sec + 20..sec + 24].copy_from_slice(&0xFFFF_FF00u32.to_le_bytes());
        pe[sec + 40 + 16..sec + 40 + 20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        for v in [View::Code, View::NoOverlay, View::Whole] {
            let _ = extract(&pe, v); // must not panic
        }
    }

    #[test]
    fn truncated_and_random_inputs_never_panic() {
        let pe = tiny_pe();
        for cut in (0..pe.len()).step_by(7) {
            for v in [View::Code, View::NoOverlay] {
                let ex = extract(&pe[..cut], v);
                assert!(ex.bytes.len() <= cut);
            }
        }
        let mut s: u64 = 0x9e3779b97f4a7c15;
        for _ in 0..300 {
            let mut buf = pe.clone();
            for _ in 0..16 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let i = (s as usize) % buf.len();
                buf[i] = (s >> 32) as u8;
            }
            let _ = extract(&buf, View::Code);
        }
    }
}
