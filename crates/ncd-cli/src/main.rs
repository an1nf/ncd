//! Corpus layout: `ROOT/<label>/<sample>`; the file stem is the sample id.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use ncd_core::extract::{self, Format};
use ncd_core::metrics::{self, Report};
use ncd_core::synth::{self, SynthConfig};
use ncd_core::{Codec, DistMatrix, Method, View, knn, ncd, npy};
use rayon::prelude::*;
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "ncd",
    version,
    about = "Normalized Compression Distance for malware-family and network-anomaly classification"
)]
struct Cli {
    /// Worker threads [default: all cores]
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Opts {
    /// zstd[:1-22], xz[:0-9], bzip2[:1-9], deflate[:0-9]
    #[arg(long, short, default_value = "zstd:19")]
    codec: Codec,
    /// concat, concat-sym, conditional (zstd only)
    #[arg(long, short, default_value = "concat")]
    method: Method,
    /// code, no-overlay, whole
    #[arg(long, short, default_value = "code")]
    view: View,
    /// Truncate each object after extraction
    #[arg(long)]
    max_bytes: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// NCD of two files
    Dist {
        a: PathBuf,
        b: PathBuf,
        #[command(flatten)]
        opts: Opts,
    },
    /// Pairwise matrix as .npy plus index CSV
    Matrix {
        dir: PathBuf,
        /// Output .npy; also writes <out>.index.csv
        #[arg(long, short)]
        out: PathBuf,
        #[command(flatten)]
        opts: Opts,
    },
    /// k-NN evaluation, leave-one-out or hold-out
    Eval {
        dir: PathBuf,
        #[arg(long, short, default_value_t = 1)]
        k: usize,
        /// Stratified hold-out fraction; 0 = leave-one-out
        #[arg(long, default_value_t = 0.0)]
        test_frac: f64,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        /// Metrics and predictions as JSON
        #[arg(long)]
        json: Option<PathBuf>,
        #[command(flatten)]
        opts: Opts,
    },
    /// Mean NCD to the k nearest references
    Score {
        /// Reference corpus, e.g. benign only
        #[arg(long)]
        refs: PathBuf,
        /// Query corpus; labels other than --normal-label are positives for ROC AUC
        #[arg(long)]
        queries: PathBuf,
        #[arg(long, short, default_value_t = 5)]
        k: usize,
        #[arg(long, default_value = "benign")]
        normal_label: String,
        /// Output CSV: id,label,score
        #[arg(long, short)]
        out: PathBuf,
        #[command(flatten)]
        opts: Opts,
    },
    /// Compare codecs: throughput, NCD(x,x), window overflows, 1-NN accuracy
    Bench {
        dir: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "zstd:3,zstd:19,xz:9,bzip2:9,deflate:9"
        )]
        codecs: Vec<Codec>,
        #[arg(long, short, default_value = "concat")]
        method: Method,
        #[arg(long, short, default_value = "code")]
        view: View,
        #[arg(long)]
        max_bytes: Option<usize>,
    },
    /// Write the synthetic corpus
    Synth {
        out: PathBuf,
        #[arg(long, default_value_t = 6)]
        families: usize,
        #[arg(long, default_value_t = 20)]
        per_family: usize,
        #[arg(long, default_value_t = 30)]
        benign: usize,
        #[arg(long, default_value_t = 16384)]
        template_len: usize,
        #[arg(long, default_value_t = 0.08)]
        mutation_rate: f64,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
    },
}

struct Corpus {
    ids: Vec<String>,
    labels: Vec<u32>,
    names: Vec<String>,
    objs: Vec<Vec<u8>>,
}

impl Corpus {
    fn load(root: &Path, opts: &Opts) -> Result<Self> {
        if !root.is_dir() {
            bail!("{} is not a directory", root.display());
        }
        let mut entries: Vec<(PathBuf, String)> = Vec::new();
        for e in walkdir::WalkDir::new(root)
            .min_depth(1)
            .max_depth(2)
            .sort_by_file_name()
        {
            let e = e?;
            if !e.file_type().is_file() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let label = if e.depth() == 2 {
                e.path()
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|s| s.to_string_lossy().into_owned())
            } else {
                None
            };
            entries.push((e.into_path(), label.unwrap_or_else(|| "unlabeled".into())));
        }
        if entries.is_empty() {
            bail!("no samples found under {}", root.display());
        }
        let loaded: Vec<extract::Extracted> = entries
            .par_iter()
            .map(|(p, _)| {
                extract::load(p, opts.view, opts.max_bytes).with_context(|| p.display().to_string())
            })
            .collect::<Result<_>>()?;

        let fell_back = loaded.iter().filter(|e| e.fell_back).count();
        let mut formats: BTreeMap<&str, usize> = BTreeMap::new();
        for e in &loaded {
            *formats
                .entry(match e.format {
                    Format::Pe => "PE",
                    Format::Elf => "ELF",
                    Format::Unknown => "other",
                })
                .or_default() += 1;
        }
        let mut names: Vec<String> = entries.iter().map(|(_, l)| l.clone()).collect();
        names.sort();
        names.dedup();
        let index: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u32))
            .collect();
        let labels = entries.iter().map(|(_, l)| index[l.as_str()]).collect();
        let ids = entries
            .iter()
            .map(|(p, _)| {
                p.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        let objs: Vec<Vec<u8>> = loaded.into_iter().map(|e| e.bytes).collect();
        let total: usize = objs.iter().map(Vec::len).sum();
        eprintln!(
            "loaded {} samples, {} classes, {:.1} MiB ({} view; formats {:?}; {} fell back to whole file)",
            objs.len(),
            names.len(),
            total as f64 / (1 << 20) as f64,
            opts.view.name(),
            formats,
            fell_back
        );
        Ok(Self {
            ids,
            labels,
            names,
            objs,
        })
    }

    fn sizes(&self) -> Vec<usize> {
        self.objs.iter().map(Vec::len).collect()
    }

    fn subset(&self, idx: &[usize]) -> Vec<&[u8]> {
        idx.iter().map(|&i| self.objs[i].as_slice()).collect()
    }
}

fn warn_window(codec: Codec, a: &[usize], b: &[usize], same: bool) {
    let over = ncd::window_overflows(codec.window(), a, b, same);
    if over > 0 {
        eprintln!(
            "warning: {over} pair(s) exceed the {} KiB window of {codec}; their NCD saturates towards 1 \
             (use zstd/xz, --view code or --max-bytes)",
            codec.window() >> 10
        );
    }
}

fn print_report(r: &Report) {
    println!("accuracy     {:.4}", r.accuracy);
    println!("macro-F1     {:.4}", r.macro_f1);
    println!("weighted-F1  {:.4}", r.weighted_f1);
    println!(
        "\n{:<24} {:>9} {:>9} {:>9} {:>8}",
        "class", "precision", "recall", "f1", "support"
    );
    for c in &r.per_class {
        println!(
            "{:<24} {:>9.4} {:>9.4} {:>9.4} {:>8}",
            c.label, c.precision, c.recall, c.f1, c.support
        );
    }
}

fn report_json(r: &Report) -> serde_json::Value {
    json!({
        "accuracy": r.accuracy,
        "macro_f1": r.macro_f1,
        "weighted_f1": r.weighted_f1,
        "per_class": r.per_class.iter().map(|c| json!({
            "label": c.label, "precision": c.precision, "recall": c.recall, "f1": c.f1, "support": c.support
        })).collect::<Vec<_>>(),
        "confusion": r.confusion,
    })
}

/// Seeded Fisher-Yates per class.
fn stratified_split(labels: &[u32], test_frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut by_class: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_class.entry(l).or_default().push(i);
    }
    let mut s = seed ^ 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (_, mut idx) in by_class {
        for i in (1..idx.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let n_test = if idx.len() < 2 {
            0
        } else {
            ((idx.len() as f64 * test_frac).round() as usize).clamp(1, idx.len() - 1)
        };
        test.extend_from_slice(&idx[..n_test]);
        train.extend_from_slice(&idx[n_test..]);
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }
    match cli.cmd {
        Cmd::Dist { a, b, opts } => {
            let xa = extract::load(&a, opts.view, opts.max_bytes)?;
            let xb = extract::load(&b, opts.view, opts.max_bytes)?;
            warn_window(opts.codec, &[xa.bytes.len()], &[xb.bytes.len()], false);
            let d = ncd::ncd_with(opts.codec, opts.method, &xa.bytes, &xb.bytes)?;
            println!("{d:.6}");
        }

        Cmd::Matrix { dir, out, opts } => {
            let c = Corpus::load(&dir, &opts)?;
            warn_window(opts.codec, &c.sizes(), &c.sizes(), true);
            let t = Instant::now();
            let m = ncd::pairwise(opts.codec, &c.objs, opts.method)?;
            eprintln!("{}×{} matrix in {:.2?}", m.rows(), m.cols(), t.elapsed());
            npy::write_f32(
                BufWriter::new(fs::File::create(&out)?),
                m.rows(),
                m.cols(),
                m.as_slice(),
            )?;
            let index = out.with_extension("index.csv");
            let mut w = BufWriter::new(fs::File::create(&index)?);
            writeln!(w, "id,label")?;
            for (id, &l) in c.ids.iter().zip(&c.labels) {
                writeln!(w, "{id},{}", c.names[l as usize])?;
            }
            w.flush()?;
            eprintln!("wrote {} and {}", out.display(), index.display());
        }

        Cmd::Eval {
            dir,
            k,
            test_frac,
            seed,
            json,
            opts,
        } => {
            let c = Corpus::load(&dir, &opts)?;
            let t = Instant::now();
            let holdout = test_frac > 0.0;
            let (eval_idx, y_pred, protocol) = if holdout {
                let (train, test) = stratified_split(&c.labels, test_frac, seed);
                if train.is_empty() || test.is_empty() {
                    bail!("empty train or test set");
                }
                let sizes = c.sizes();
                let (ts, tr): (Vec<usize>, Vec<usize>) = (
                    test.iter().map(|&i| sizes[i]).collect(),
                    train.iter().map(|&i| sizes[i]).collect(),
                );
                warn_window(opts.codec, &ts, &tr, false);
                let cross =
                    ncd::cross(opts.codec, &c.subset(&test), &c.subset(&train), opts.method)?;
                let train_labels: Vec<u32> = train.iter().map(|&i| c.labels[i]).collect();
                let protocol = format!("hold-out ({} train / {} test)", train.len(), test.len());
                (test, knn::predict(&cross, &train_labels, k), protocol)
            } else {
                warn_window(opts.codec, &c.sizes(), &c.sizes(), true);
                let m = ncd::pairwise(opts.codec, &c.objs, opts.method)?;
                (
                    (0..c.objs.len()).collect(),
                    knn::loo_predict(&m, &c.labels, k),
                    "leave-one-out".to_string(),
                )
            };
            let y_true: Vec<u32> = eval_idx.iter().map(|&i| c.labels[i]).collect();
            let r = metrics::report(&y_true, &y_pred, &c.names);
            println!(
                "NCD {}-NN | codec {} | method {} | view {} | {protocol} | {:.2?}\n",
                k,
                opts.codec,
                opts.method.name(),
                opts.view.name(),
                t.elapsed()
            );
            print_report(&r);
            if let Some(path) = json {
                let preds: Vec<_> = eval_idx
                    .iter()
                    .zip(&y_pred)
                    .map(|(&i, &p)| {
                        json!({
                            "id": c.ids[i],
                            "true": c.names[c.labels[i] as usize],
                            "pred": c.names.get(p as usize).cloned().unwrap_or_default(),
                        })
                    })
                    .collect();
                let doc = json!({
                    "codec": opts.codec.to_string(), "method": opts.method.name(), "view": opts.view.name(),
                    "k": k, "protocol": protocol, "labels": c.names,
                    "metrics": report_json(&r), "predictions": preds,
                });
                let mut w = BufWriter::new(fs::File::create(&path)?);
                serde_json::to_writer_pretty(&mut w, &doc)?;
                w.flush()?;
                eprintln!("wrote {}", path.display());
            }
        }

        Cmd::Score {
            refs,
            queries,
            k,
            normal_label,
            out,
            opts,
        } => {
            let r = Corpus::load(&refs, &opts)?;
            let q = Corpus::load(&queries, &opts)?;
            warn_window(opts.codec, &q.sizes(), &r.sizes(), false);
            let t = Instant::now();
            let cross = ncd::cross(opts.codec, &q.objs, &r.objs, opts.method)?;
            let scores = knn::anomaly_scores(&cross, k);
            eprintln!(
                "{}×{} cross matrix in {:.2?}",
                cross.rows(),
                cross.cols(),
                t.elapsed()
            );
            let mut w = BufWriter::new(fs::File::create(&out)?);
            writeln!(w, "id,label,score")?;
            for ((id, &l), s) in q.ids.iter().zip(&q.labels).zip(&scores) {
                writeln!(w, "{id},{},{s}", q.names[l as usize])?;
            }
            w.flush()?;
            let positive: Vec<bool> = q
                .labels
                .iter()
                .map(|&l| q.names[l as usize] != normal_label)
                .collect();
            match metrics::roc_auc(&scores, &positive) {
                Some(auc) => println!("ROC AUC (positive = not `{normal_label}`): {auc:.4}"),
                None => println!(
                    "ROC AUC unavailable (queries need both `{normal_label}` and other labels)"
                ),
            }
            eprintln!("wrote {}", out.display());
        }

        Cmd::Bench {
            dir,
            codecs,
            method,
            view,
            max_bytes,
        } => {
            let opts = Opts {
                codec: Codec::default(),
                method,
                view,
                max_bytes,
            };
            let c = Corpus::load(&dir, &opts)?;
            let sizes = c.sizes();
            let total: usize = sizes.iter().sum();
            let n = c.objs.len();
            let labelled = c.names.len() > 1;
            println!(
                "| codec | C(x) MiB/s | mean C(x)/|x| | mean NCD(x,x) | pairs over window | matrix time | 1-NN LOO acc | macro-F1 |"
            );
            println!("|---|---:|---:|---:|---:|---:|---:|---:|");
            for codec in codecs {
                if method == Method::Conditional && !codec.supports_conditional() {
                    println!(
                        "| {codec} | - | - | - | - | n/a ({}) | - | - |",
                        method.name()
                    );
                    continue;
                }
                let t = Instant::now();
                let lens = ncd::compressed_lens(codec, &c.objs)?;
                let secs = t.elapsed().as_secs_f64();
                let ratio = lens
                    .iter()
                    .zip(&sizes)
                    .map(|(&l, &s)| l as f64 / s.max(1) as f64)
                    .sum::<f64>()
                    / n as f64;
                let self_d: Vec<f64> = c
                    .objs
                    .par_iter()
                    .map(|x| ncd::ncd_with(codec, method, x, x).unwrap_or(f64::NAN))
                    .collect();
                let self_mean = self_d.iter().sum::<f64>() / n as f64;
                let over = ncd::window_overflows(codec.window(), &sizes, &sizes, true);
                let t = Instant::now();
                let m: DistMatrix = ncd::pairwise(codec, &c.objs, method)?;
                let mt = t.elapsed();
                let (acc, f1) = if labelled {
                    let r =
                        metrics::report(&c.labels, &knn::loo_predict(&m, &c.labels, 1), &c.names);
                    (format!("{:.4}", r.accuracy), format!("{:.4}", r.macro_f1))
                } else {
                    ("-".into(), "-".into())
                };
                println!(
                    "| {codec} | {:.1} | {ratio:.3} | {self_mean:.4} | {over} / {} | {mt:.2?} | {acc} | {f1} |",
                    total as f64 / (1 << 20) as f64 / secs,
                    n * (n - 1) / 2
                );
            }
        }

        Cmd::Synth {
            out,
            families,
            per_family,
            benign,
            template_len,
            mutation_rate,
            seed,
        } => {
            let cfg = SynthConfig {
                families,
                per_family,
                benign,
                template_len,
                mutation_rate,
                seed,
            };
            let samples = synth::generate(&cfg);
            for s in &samples {
                let dir = out.join(&s.label);
                fs::create_dir_all(&dir)?;
                fs::write(dir.join(format!("{}.bin", s.id)), &s.bytes)?;
            }
            eprintln!(
                "wrote {} synthetic samples to {}",
                samples.len(),
                out.display()
            );
        }
    }
    Ok(())
}
