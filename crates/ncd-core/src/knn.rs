//! k-NN classification and anomaly scores.

use rayon::prelude::*;

use crate::ncd::DistMatrix;

/// `k` smallest indices, sorted, NaN last.
pub fn nearest(row: &[f32], k: usize, exclude: Option<usize>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).filter(|&j| Some(j) != exclude).collect();
    let k = k.min(idx.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &usize, b: &usize| row[*a].total_cmp(&row[*b]).then(a.cmp(b));
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp);
    idx
}

/// Majority vote; ties go to the class with the nearest member.
pub fn vote(row: &[f32], labels: &[u32], k: usize, exclude: Option<usize>) -> Option<u32> {
    let nn = nearest(row, k, exclude);
    // (label, count, first rank)
    let mut tally: Vec<(u32, usize, usize)> = Vec::new();
    for (rank, &j) in nn.iter().enumerate() {
        let l = labels[j];
        match tally.iter_mut().find(|t| t.0 == l) {
            Some(t) => t.1 += 1,
            None => tally.push((l, 1, rank)),
        }
    }
    tally
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)))
        .map(|t| t.0)
}

pub fn loo_predict(m: &DistMatrix, labels: &[u32], k: usize) -> Vec<u32> {
    assert_eq!(m.rows(), m.cols(), "pairwise matrix must be square");
    assert_eq!(m.rows(), labels.len());
    (0..m.rows())
        .into_par_iter()
        .map(|i| vote(m.row(i), labels, k, Some(i)).unwrap_or(u32::MAX))
        .collect()
}

pub fn predict(cross: &DistMatrix, train_labels: &[u32], k: usize) -> Vec<u32> {
    assert_eq!(cross.cols(), train_labels.len());
    (0..cross.rows())
        .into_par_iter()
        .map(|i| vote(cross.row(i), train_labels, k, None).unwrap_or(u32::MAX))
        .collect()
}

/// Mean distance to the `k` nearest references.
pub fn anomaly_scores(cross: &DistMatrix, k: usize) -> Vec<f32> {
    (0..cross.rows())
        .into_par_iter()
        .map(|i| {
            let row = cross.row(i);
            let nn = nearest(row, k, None);
            if nn.is_empty() {
                return f32::NAN;
            }
            nn.iter().map(|&j| row[j]).sum::<f32>() / nn.len() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_excludes_and_sorts() {
        let row = [0.5, 0.1, 0.9, 0.0, 0.3];
        assert_eq!(nearest(&row, 3, Some(3)), vec![1, 4, 0]);
        assert_eq!(nearest(&row, 10, None).len(), 5);
        assert!(nearest(&[], 3, None).is_empty());
    }

    #[test]
    fn tie_break_by_nearest() {
        let row = [0.2, 0.1, 0.9];
        let labels = [3, 7, 3];
        assert_eq!(vote(&row, &labels, 2, None), Some(7));
        assert_eq!(vote(&row, &labels, 3, None), Some(3));
    }

    #[test]
    fn loo_on_blocks() {
        let n = 6;
        let mut d = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = if (i < 3) == (j < 3) { 0.1 } else { 0.9 };
            }
        }
        let m = DistMatrix::new(n, n, d).unwrap();
        let labels = [0, 0, 0, 1, 1, 1];
        assert_eq!(loo_predict(&m, &labels, 1), labels.to_vec());
        let s = anomaly_scores(&m, 2);
        assert!((s[0] - 0.1).abs() < 1e-6);
    }
}
