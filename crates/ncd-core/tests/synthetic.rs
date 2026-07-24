use ncd_core::metrics::{report, roc_auc};
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
        (Codec::Zstd { level: 19 }, Method::Conditional),
        (Codec::Xz { level: 6 }, Method::Concat),
    ] {
        let m = ncd::pairwise(codec, &objs, method).unwrap();
        let pred = knn::loo_predict(&m, &labels, 1);
        let r = report(&labels, &pred, &names);
        assert!(
            r.accuracy > 0.9,
            "{codec}/{}: acc={}",
            method.name(),
            r.accuracy
        );
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

#[test]
fn anomaly_score_separates_malware_from_benign() {
    let (objs, labels, names) = corpus();
    let benign = names.iter().position(|n| n == "benign").unwrap() as u32;
    // first half of benign covers every vendor
    let ben_idx: Vec<usize> = (0..objs.len()).filter(|&i| labels[i] == benign).collect();
    let (refs, held) = ben_idx.split_at(ben_idx.len() / 2);
    let queries: Vec<usize> = (0..objs.len()).filter(|i| !refs.contains(i)).collect();
    let r: Vec<&[u8]> = refs.iter().map(|&i| objs[i].as_slice()).collect();
    let q: Vec<&[u8]> = queries.iter().map(|&i| objs[i].as_slice()).collect();
    let cross = ncd::cross(Codec::Zstd { level: 19 }, &q, &r, Method::Concat).unwrap();
    let scores = knn::anomaly_scores(&cross, 3);
    let positive: Vec<bool> = queries.iter().map(|&i| labels[i] != benign).collect();
    let auc = roc_auc(&scores, &positive).unwrap();
    assert!(auc > 0.9, "AUC {auc}, held-out benign {}", held.len());
}
