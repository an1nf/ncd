#[derive(Clone, Debug)]
pub struct ClassMetrics {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: u64,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub accuracy: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub per_class: Vec<ClassMetrics>,
    /// `confusion[true][pred]`
    pub confusion: Vec<Vec<u64>>,
}

/// Predictions outside `0..names.len()` count as errors.
pub fn report(y_true: &[u32], y_pred: &[u32], names: &[String]) -> Report {
    assert_eq!(y_true.len(), y_pred.len());
    let k = names.len();
    let mut conf = vec![vec![0u64; k]; k];
    let mut correct = 0u64;
    let mut pred_count = vec![0u64; k];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        if t == p {
            correct += 1;
        }
        if (p as usize) < k {
            conf[t as usize][p as usize] += 1;
            pred_count[p as usize] += 1;
        }
    }
    let n = y_true.len().max(1) as f64;
    let mut per_class = Vec::with_capacity(k);
    for c in 0..k {
        let tp = conf[c][c] as f64;
        let support: u64 = y_true.iter().filter(|&&t| t as usize == c).count() as u64;
        let precision = if pred_count[c] == 0 {
            0.0
        } else {
            tp / pred_count[c] as f64
        };
        let recall = if support == 0 {
            0.0
        } else {
            tp / support as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        per_class.push(ClassMetrics {
            label: names[c].clone(),
            precision,
            recall,
            f1,
            support,
        });
    }
    let present: Vec<&ClassMetrics> = per_class.iter().filter(|c| c.support > 0).collect();
    let macro_f1 = present.iter().map(|c| c.f1).sum::<f64>() / present.len().max(1) as f64;
    let weighted_f1 = present.iter().map(|c| c.f1 * c.support as f64).sum::<f64>() / n;
    Report {
        accuracy: correct as f64 / n,
        macro_f1,
        weighted_f1,
        per_class,
        confusion: conf,
    }
}

/// Mann-Whitney U with tied ranks. `None` if a class is empty or a score is NaN.
pub fn roc_auc(scores: &[f32], positive: &[bool]) -> Option<f64> {
    assert_eq!(scores.len(), positive.len());
    if scores.iter().any(|s| s.is_nan()) {
        return None;
    }
    let n_pos = positive.iter().filter(|&&p| p).count();
    let n_neg = positive.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum_pos = 0f64;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &t in &idx[i..=j] {
            if positive[t] {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    let u = rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0;
    Some(u / (n_pos as f64 * n_neg as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_mixed() {
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let r = report(&[0, 0, 1, 1], &[0, 0, 1, 1], &names);
        assert_eq!(r.accuracy, 1.0);
        assert_eq!(r.macro_f1, 1.0);
        let r = report(&[0, 0, 1, 1], &[0, 1, 1, 1], &names);
        assert_eq!(r.accuracy, 0.75);
        assert_eq!(r.confusion, vec![vec![1, 1], vec![0, 2]]);
        assert!((r.per_class[1].precision - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn auc_values() {
        assert_eq!(
            roc_auc(&[0.1, 0.2, 0.8, 0.9], &[false, false, true, true]),
            Some(1.0)
        );
        assert_eq!(
            roc_auc(&[0.9, 0.8, 0.2, 0.1], &[false, false, true, true]),
            Some(0.0)
        );
        assert_eq!(
            roc_auc(&[0.5, 0.5, 0.5, 0.5], &[false, true, false, true]),
            Some(0.5)
        );
        assert_eq!(roc_auc(&[0.1, 0.2], &[true, true]), None);
    }
}
