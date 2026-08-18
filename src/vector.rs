//! Distance kernels for normalized and arbitrary vectors.

use crate::{Error, Result};

/// Absolute deviation from unit L2 norm still treated as pre-normalized.
///
/// Matches the FiQA fixture preparer: a vector is accepted when
/// `| ||v|| - 1 | <= UNIT_NORM_TOLERANCE`.
pub const UNIT_NORM_TOLERANCE: f32 = 0.01;

/// Returns `Ok(())` when every coordinate is finite and `||v||` is within
/// [`UNIT_NORM_TOLERANCE`] of `1`.
///
/// [`crate::HnswIndex::insert`] and search call this when
/// [`crate::Config::check_vectors`] is set. Debug builds also
/// `debug_assert` the same contract when the flag is off.
pub fn check_normalized_vector(vector: &[f32]) -> Result<()> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(Error::InvalidVector("coordinates must be finite"));
    }
    let norm = dot(vector, vector).sqrt();
    if !norm.is_finite() || (norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
        return Err(Error::InvalidVector("norm must be within 0.01 of one"));
    }
    Ok(())
}

/// Debug-asserts the pre-normalized contract; returns [`Error::InvalidVector`]
/// when `enforce` is set.
pub(crate) fn validate_input_vector(vector: &[f32], enforce: bool) -> Result<()> {
    if enforce {
        return check_normalized_vector(vector);
    }
    debug_assert!(
        check_normalized_vector(vector).is_ok(),
        "HNSW insert/search vectors must be finite and unit-normalized (||v|| within {UNIT_NORM_TOLERANCE} of 1); set Config::check_vectors to return Error"
    );
    Ok(())
}

/// Returns the dot product of equal-length vectors.
///
/// AVX2 and NEON kernels use eight partial sums. Every other target uses the
/// safe scalar fallback.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product dimensions must match");

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked at runtime. The implementation
        // performs unaligned loads only while eight elements remain.
        return unsafe { dot_avx2(a, b) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Advanced SIMD is part of the baseline AArch64 architecture.
        // The implementation performs loads only while eight elements remain.
        return unsafe { dot_neon(a, b) };
    }

    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut accumulator = [0.0_f32; 8];
    let chunks = a.len() / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        for lane in 0..8 {
            accumulator[lane] += a[offset + lane] * b[offset + lane];
        }
    }
    let mut sum = accumulator.into_iter().sum::<f32>();
    for index in chunks * 8..a.len() {
        sum += a[index] * b[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut accumulator = _mm256_setzero_ps();
    let mut index = 0;
    while index + 8 <= a.len() {
        // SAFETY: the loop condition guarantees both pointers have eight
        // readable f32 values. Unaligned loads accept arbitrary alignment.
        let left = unsafe { _mm256_loadu_ps(a.as_ptr().add(index)) };
        // SAFETY: same bounds argument as the left-hand load.
        let right = unsafe { _mm256_loadu_ps(b.as_ptr().add(index)) };
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(left, right));
        index += 8;
    }
    let mut lanes = [0.0_f32; 8];
    // SAFETY: lanes contains eight writable f32 values; unaligned stores accept
    // its natural alignment.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    let mut sum = lanes.into_iter().sum::<f32>();
    while index < a.len() {
        sum += a[index] * b[index];
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let mut low = vdupq_n_f32(0.0);
    let mut high = vdupq_n_f32(0.0);
    let mut index = 0;
    while index + 8 <= a.len() {
        // SAFETY: the loop condition guarantees four readable elements at each
        // pointer, and NEON loads do not require alignment.
        let left_low = unsafe { vld1q_f32(a.as_ptr().add(index)) };
        // SAFETY: same bounds argument as left_low.
        let right_low = unsafe { vld1q_f32(b.as_ptr().add(index)) };
        // SAFETY: the loop condition guarantees another four readable values.
        let left_high = unsafe { vld1q_f32(a.as_ptr().add(index + 4)) };
        // SAFETY: same bounds argument as left_high.
        let right_high = unsafe { vld1q_f32(b.as_ptr().add(index + 4)) };
        low = vaddq_f32(low, vmulq_f32(left_low, right_low));
        high = vaddq_f32(high, vmulq_f32(left_high, right_high));
        index += 8;
    }

    let mut lanes = [0.0_f32; 8];
    // SAFETY: each destination points at four writable f32 values.
    unsafe { vst1q_f32(lanes.as_mut_ptr(), low) };
    // SAFETY: lanes[4..] contains four writable f32 values.
    unsafe { vst1q_f32(lanes.as_mut_ptr().add(4), high) };
    let mut sum = lanes.into_iter().sum::<f32>();
    while index < a.len() {
        sum += a[index] * b[index];
        index += 1;
    }
    sum
}

/// Cosine distance for vectors already normalized to unit length.
///
/// This is `1 - dot(a, b)` and does not inspect finiteness or norm.
/// Insert and search `debug_assert` that contract, and return
/// [`Error::InvalidVector`] when [`crate::Config::check_vectors`] is set.
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

/// Cosine similarity for arbitrary vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine dimensions must match");
    let dot_product = dot(a, b);
    let a_squared = dot(a, a);
    let b_squared = dot(b, b);
    if dot_product == 0.0 || a_squared == 0.0 || b_squared == 0.0 {
        return 0.0;
    }
    dot_product / (a_squared.sqrt() * b_squared.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn dot_product_is_correct() {
        assert!((dot(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0, 4.0, 5.0]) - 40.0).abs() < 0.001);
    }

    #[test]
    fn cosine_distance_for_unit_vectors() {
        let a = [1.0, 0.0, 0.0, 0.0];
        assert!(cosine_distance(&a, &a).abs() < 0.001);
        assert!((cosine_distance(&a, &[0.0, 1.0, 0.0, 0.0]) - 1.0).abs() < 0.001);
        assert!((cosine_distance(&a, &[-1.0, 0.0, 0.0, 0.0]) - 2.0).abs() < 0.001);
    }

    #[test]
    fn general_cosine_handles_zero_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.001);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn check_normalized_vector_rejects_nan_inf_and_off_unit() {
        assert!(check_normalized_vector(&[1.0, 0.0]).is_ok());
        assert!(check_normalized_vector(&[1.009, 0.0]).is_ok());
        assert!(matches!(
            check_normalized_vector(&[f32::NAN, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            check_normalized_vector(&[f32::INFINITY, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            check_normalized_vector(&[f32::NEG_INFINITY, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            check_normalized_vector(&[2.0, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            check_normalized_vector(&[0.0, 0.0]),
            Err(Error::InvalidVector(_))
        ));
        assert!(matches!(
            check_normalized_vector(&[1.011, 0.0]),
            Err(Error::InvalidVector(_))
        ));
    }

    #[test]
    fn dispatched_kernel_matches_scalar_with_a_tail() {
        let left = (0..19).map(|value| value as f32 / 19.0).collect::<Vec<_>>();
        let right = (0..19)
            .map(|value| (19 - value) as f32 / 19.0)
            .collect::<Vec<_>>();
        assert!((dot(&left, &right) - dot_scalar(&left, &right)).abs() < 0.000_01);
    }
}
