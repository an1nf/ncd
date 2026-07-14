use ncd_core::metrics::report;
use ncd_core::synth::{SynthConfig, generate};
use ncd_core::{Codec, Method, knn, ncd};

fn corpus() -> (Vec<Vec<u8>>, Vec<u32>, Vec<String>) {
    let cfg = SynthConfig {
        families: 4,
        per_family: 8,
        benign: 10,
        template_len: 8 << 10,
        ..Default::default()
    };
    let samples = generate(&cfg);
    let mut names: Vec<String> = samples.iter().map(|s| s.label.clone()).collect();
    names.sort();
    names.dedup();
    let labels = samples
        .iter()
        .map(|s| names.iter().position(|n| *n == s.label).unwrap() as u32)
        .collect();
    (
        samples.into_iter().map(|s| s.bytes).collect(),
        labels,
        names,
    )
}

#[test]
fn loo_knn_classifies_families() {
    let (objs, labels, names) = corpus();
    for (codec, method) in [
        (Codec::Zstd { level: 19 }, Method::Concat),
        (Codec::Xz { level: 6 }, Method::Concat),
    ] {
        let m = ncd::pairwise(codec, &objs, method).unwrap();
        let pred = knn::loo_predict(&m, &labels, 1);
        let r = report(&labels, &pred, &names);
        let fam_ok = labels
            .iter()
            .zip(&pred)
            .filter(|(t, _)| names[**t as usize] != "benign")
            .all(|(t, p)| t == p);
        assert!(
            fam_ok,
            "{codec}/{}: families misclassified, acc={}",
            method.name(),
            r.accuracy
        );
    }
}
