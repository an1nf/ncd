//! Deterministic synthetic "malware family" corpus.
//!
//! Real malware cannot ship in a public repository, so tests and the demo use
//! a generator that mimics the properties NCD exploits:
//!
//! * each family has its own code-like byte source (a sparse order-1 Markov
//!   chain, like opcode sequences of a shared codebase) and a template drawn
//!   from it;
//! * samples are *variants* of the template: point mutations, inserted junk,
//!   deleted, duplicated and reordered blocks (recompilation, polymorphism,
//!   function reordering) and high-entropy padding (packed stubs);
//! * every sample, benign or not, embeds the same "runtime library" blob,
//!   like statically linked CRT code, so similarity is not trivially zero;
//! * benign samples come from a few "vendors" (toolchains/frameworks): each
//!   sample links a random subset of its vendor's shared modules plus code of
//!   its own, so benign software is diverse but not unrelated — as with real
//!   programs built on common frameworks.
//!
//! The corpus is a sanity check for the pipeline, not evidence about real
//! malware — results on it are reported as such.

use crate::rng::SplitMix64;

#[derive(Clone, Debug)]
pub struct SynthConfig {
    pub families: usize,
    pub per_family: usize,
    pub benign: usize,
    /// Number of benign vendors whose library modules benign samples share.
    pub benign_vendors: usize,
    pub template_len: usize,
    /// Fraction of bytes affected by mutations, e.g. 0.05.
    pub mutation_rate: f64,
    pub seed: u64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            families: 6,
            per_family: 20,
            benign: 30,
            benign_vendors: 4,
            template_len: 16 << 10,
            mutation_rate: 0.08,
            seed: 2026,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub label: String,
    pub id: String,
    pub bytes: Vec<u8>,
}

/// Sparse order-1 Markov source over bytes.
struct Chain {
    succ: Vec<[u8; 4]>,
}

impl Chain {
    fn new(rng: &mut SplitMix64) -> Self {
        // A restricted alphabet makes the source compressible like code.
        let alphabet: Vec<u8> = rng.bytes(64);
        let succ = (0..256)
            .map(|_| std::array::from_fn(|_| alphabet[rng.below(alphabet.len())]))
            .collect();
        Self { succ }
    }

    fn emit(&self, rng: &mut SplitMix64, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut state = rng.byte();
        for _ in 0..len {
            let u = rng.unit();
            state = if u < 0.02 {
                rng.byte()
            } else {
                let pick = if u < 0.57 {
                    0
                } else if u < 0.82 {
                    1
                } else if u < 0.97 {
                    2
                } else {
                    3
                };
                self.succ[state as usize][pick]
            };
            out.push(state);
        }
        out
    }
}

fn mutate(rng: &mut SplitMix64, template: &[u8], rate: f64, chain: &Chain) -> Vec<u8> {
    let mut v = template.to_vec();
    // point substitutions
    for b in v.iter_mut() {
        if rng.unit() < rate * 0.4 {
            *b = rng.byte();
        }
    }
    // block-level edits
    let ops = ((template.len() as f64 * rate) / 256.0).ceil() as usize;
    for _ in 0..ops {
        if v.len() < 1024 {
            break;
        }
        match rng.below(4) {
            0 => {
                // insert junk drawn from the family's own source (dead code)
                let at = rng.below(v.len());
                let n = rng.range(16, 256);
                let junk = chain.emit(rng, n);
                v.splice(at..at, junk);
            }
            1 => {
                let len = rng.range(16, 256);
                let at = rng.below(v.len() - len);
                v.drain(at..at + len);
            }
            2 => {
                let len = rng.range(32, 256);
                let from = rng.below(v.len() - len);
                let block = v[from..from + len].to_vec();
                let at = rng.below(v.len());
                v.splice(at..at, block);
            }
            _ => {
                // swap two blocks (function reordering)
                let len = rng.range(64, 512);
                let a = rng.below(v.len() - 2 * len);
                let b = rng.range(a + len, v.len() - len);
                let (left, right) = v.split_at_mut(b);
                left[a..a + len].swap_with_slice(&mut right[..len]);
            }
        }
    }
    v
}

fn assemble(rng: &mut SplitMix64, body: Vec<u8>, runtime: &[u8]) -> Vec<u8> {
    let pre = rng.range(0, 1024);
    let mut out = rng.bytes(pre); // packed-stub-like noise
    let at = rng.below(body.len() + 1);
    out.extend_from_slice(&body[..at]);
    out.extend_from_slice(runtime);
    out.extend_from_slice(&body[at..]);
    let post = rng.range(0, 1024);
    out.extend(rng.bytes(post));
    out
}

/// Generates the corpus. Labels are `family_00`, `family_01`, … and `benign`.
pub fn generate(cfg: &SynthConfig) -> Vec<Sample> {
    let mut rng = SplitMix64::new(cfg.seed);
    let runtime = Chain::new(&mut rng).emit(&mut rng, 4096);
    let mut out = Vec::new();
    for f in 0..cfg.families {
        let mut frng = rng.fork();
        let chain = Chain::new(&mut frng);
        let template = chain.emit(&mut frng, cfg.template_len);
        for _ in 0..cfg.per_family {
            let body = mutate(&mut frng, &template, cfg.mutation_rate, &chain);
            let bytes = assemble(&mut frng, body, &runtime);
            out.push(Sample {
                label: format!("family_{f:02}"),
                id: format!("{:016x}", frng.next_u64()),
                bytes,
            });
        }
    }
    // Vendors: a code source plus a pool of shared library modules.
    let vendors: Vec<(Chain, Vec<Vec<u8>>, SplitMix64)> = (0..cfg.benign_vendors.max(1))
        .map(|_| {
            let mut vrng = rng.fork();
            let chain = Chain::new(&mut vrng);
            let modules = (0..8)
                .map(|_| chain.emit(&mut vrng, cfg.template_len / 6))
                .collect();
            (chain, modules, vrng)
        })
        .collect();
    for b in 0..cfg.benign {
        let (chain, modules, _) = &vendors[b % vendors.len()];
        let mut brng = rng.fork();
        let mut picked: Vec<usize> = (0..modules.len()).collect();
        for i in (1..picked.len()).rev() {
            picked.swap(i, brng.below(i + 1));
        }
        let take = brng.range(3, 5);
        let mut body = Vec::new();
        for &m in &picked[..take] {
            body.extend_from_slice(&modules[m]);
            let own = brng.range(cfg.template_len / 16, cfg.template_len / 4);
            body.extend(chain.emit(&mut brng, own));
        }
        let bytes = assemble(&mut brng, body, &runtime);
        out.push(Sample {
            label: "benign".into(),
            id: format!("{:016x}", brng.next_u64()),
            bytes,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_sized() {
        let cfg = SynthConfig {
            families: 2,
            per_family: 3,
            benign: 2,
            template_len: 4096,
            ..Default::default()
        };
        let a = generate(&cfg);
        let b = generate(&cfg);
        assert_eq!(a.len(), 8);
        assert!(
            a.iter()
                .zip(&b)
                .all(|(x, y)| x.bytes == y.bytes && x.id == y.id)
        );
        assert!(a.iter().all(|s| s.bytes.len() > 4096));
    }
}
