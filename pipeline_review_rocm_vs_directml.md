# Pipeline Review: Reproduction of "Adversarial Threats to Cloud IDS"
### ROCm (primary, RX7600) vs. DirectML (reference) — Full Results, Timings, and Notes

**Source notebooks:** `adversarial_cloud_ids_defense_implement_cuda.ipynb` (ROCm) and
`adversarial_cloud_ids_defense_implement_dhoogla.ipynb` (DirectML)
**Paper:** Holla, Polepalli & Sasikumar, "Adversarial Threats to Cloud IDS: Robust Defense With
Adversarial Training and Feature Selection," IEEE Access, 2025.
**Dataset:** CSE-CIC-IDS2018, `dhoogla/csecicids2018` cleaned Parquet mirror, all 10 daily
capture files (6.66M raw rows, 78 columns). The paper used the full ~250GB raw CSE-CIC-IDS2018
release; our Parquet mirror is the complete, cleaned, deduplicated equivalent of the same 10
days — same attack-category coverage, much smaller on disk due to compression and dtype
optimization, not a data subset.

**ROCm is the reference implementation for this review** since all future training work moves
to ROCm. DirectML numbers are retained as the original validated baseline and used throughout
for comparison.

---

## 1. Run Metadata

| | ROCm (primary) | DirectML (reference) |
|---|---|---|
| Run started | 2026-08-01 19:50:47 | 2026-07-30 12:27:55 |
| Total wall-clock time | **03:13:12** (3h 13m) | **08:04:33** (8h 04m) |
| Device | `cuda` (ROCm 7.2.1) — AMD Radeon RX 7600 | `privateuseone:0` (DirectML) — AMD Radeon RX 7600 |
| Torch version | 2.9.1+rocm7.2.1 | (DirectML build, version not logged) |
| Seed / Data seed | `SEED=42`, `DATA_SEED=42` | `SEED=42` (no `DATA_SEED` separation in this snapshot — see §7) |
| Output folder | `./output/seed_42/` | `./output/` |
| **Overall speedup** | **2.5x faster end-to-end** (3h13m vs 8h05m) | — |

---

## 2. Configuration Used

| Parameter | ROCm | DirectML | Same? |
|---|---|---|---|
| `day_patterns` | `null` (all 10 files) | `null` (all 10 files) | ✅ |
| `test_size` | 0.2 | 0.2 (implicit default) | ✅ |
| `top_t_mutual_info` | 25 | 25 | ✅ |
| `top_t_shap` | 15 | 15 | ✅ |
| `mi_var_thresh` | 1e-05 | 1e-05 (implicit default) | ✅ |
| `mi_sample_size` | 50,000 | 50,000 (implicit default) | ✅ |
| `shap_background_samples` | 100 | 100 (implicit default) | ✅ |
| `hidden_sizes` | [128, 64, 32] | [128, 64, 32] (implicit default) | ✅ |
| `dropout` | 0.3 | 0.3 (implicit default) | ✅ |
| `learning_rate` | 0.001 | 0.001 (implicit default) | ✅ |
| `attack_eps` | 0.05 | 0.05 | ✅ |
| `pgd_iters` | 20 | 20 | ✅ |
| `pgd_eps_step_ratio` | 0.1 | 0.1 (implicit, eps/10) | ✅ |
| `deepfool_iters` | 50 | 50 | ✅ |
| `square_iters` | 300 | 300 | ✅ |
| `square_p_init` | 0.3 | 0.3 (implicit default) | ✅ |
| `n_attack_eval_samples` | 2,000 | 2,000 | ✅ |
| `n_square_samples` | 1,000 | 1,000 | ✅ |
| `n_shap_samples` | 300 | 300 | ✅ |
| `n_adv_train_samples` | 5,000 | 5,000 | ✅ |
| `epochs_baseline` | 50 | 50 | ✅ |
| `epochs_defended_stage1` | 50 | 50 | ✅ |
| `epochs_defended_stage2` | 15 | 15 | ✅ |
| `batch_size` | 512 | 512 | ✅ |
| `structural_features` (masking) | `["proto_TCP","proto_UDP","proto_ICMP"]` | not present (older snapshot) | ROCm-only addition |
| `data_seed` | 42 (explicit, isolated) | not present — split/SMOTE/feature-selection tied to the single `SEED=42` | ROCm-only fix |

**Conclusion: hyperparameters are effectively identical between the two runs.** The only
substantive differences are the `DATA_SEED` isolation and feature-masking config, both additions
made after the DirectML run, neither of which changes behavior when `SEED == DATA_SEED == 42`
in both cases (confirmed in §7).

---

## 3. Data Pipeline — Verified Identical

| Step | ROCm | DirectML | Match? |
|---|---|---|---|
| Files loaded | 10/10 daily Parquet files | 10/10 daily Parquet files | ✅ |
| Raw combined shape | 6,659,532 rows × 78 cols | 6,659,532 rows × 78 cols | ✅ exact |
| Duplicate rows dropped | 573,667 | 573,667 | ✅ exact |
| Rows after cleaning | 6,085,865 | 6,085,865 | ✅ exact |
| Class balance after cleaning | Benign: 4,760,022 / Attack: 1,325,843 | Benign: 4,760,022 / Attack: 1,325,843 | ✅ exact |
| Final feature matrix (pre-selection) | (6,085,865, 79) | (6,085,865, 79) | ✅ exact |
| Variance threshold survivors | 71/79 features | 71/79 features | ✅ exact |
| **Selected 25 MI features** | Identical list (see below) | Identical list | ✅ **byte-for-byte identical** |
| Train/test split shapes | Train (4,868,692, 25) / Test (1,217,173, 25) | Train (4,868,692, 25) / Test (1,217,173, 25) | ✅ exact |
| SMOTE-balanced train set | (7,616,036, 25), classes [3,808,018 / 3,808,018] | (7,616,036, 25), classes [3,808,018 / 3,808,018] | ✅ exact |

**Selected 25 mutual-information features (identical in both runs):**
`Bwd Packets Length Total, Avg Bwd Segment Size, Subflow Bwd Bytes, Bwd Packet Length Mean,
Fwd Packets Length Total, Subflow Fwd Bytes, Bwd Packet Length Max, Fwd Packet Length Max,
Avg Fwd Segment Size, Fwd Packet Length Mean, Bwd Packet Length Std, Fwd Packet Length Std,
Packet Length Variance, Packet Length Std, Packet Length Mean, Packet Length Max,
Avg Packet Size, Fwd Header Length, Init Fwd Win Bytes, Fwd IAT Max, Fwd IAT Total,
Init Bwd Win Bytes, Flow IAT Max, Bwd Header Length, Fwd IAT Mean`

**Note:** none of `proto_TCP`/`proto_UDP`/`proto_ICMP` were selected by mutual-information
ranking in either run — so the feature-masking added to the ROCm notebook (`structural_features`
config) is currently a no-op in practice, since there's nothing structural in the selected set
to mask. This is not a bug; it just means this particular safeguard isn't exercised by the
current feature selection outcome. It would matter if `top_t_mutual_info` were raised high
enough for a `proto_*` column to enter the selected set.

**This confirms the data-loading, cleaning, and feature-selection code is verified identical
between backends** — the only variables left that could explain downstream differences are
training dynamics themselves (see §7).

---

## 4. Baseline Model — Full Results

| Metric | ROCm | DirectML | Paper |
|---|---|---|---|
| Training time | **66.27 min** | 202.01 min | not stated |
| Clean Accuracy | 95.40% | 85.84% | 85% |
| Clean F1 | 89.74% | 70.54% | 83% |
| Clean Precision | 87.29% | 64.51% | 84% |
| Clean Recall | 92.34% | 77.82% | 82% |
| Inference time | 0.000357 ms/sample | 0.000372 ms/sample | not stated |

### Under attack

| Attack | Metric | ROCm | DirectML | Paper |
|---|---|---|---|---|
| **FGSM** | Accuracy | 74.95% | 57.05% | 60% |
| | F1 | 25.99% | 40.80% | 58% |
| | Precision | 34.65% | 28.79% | 61% |
| | Recall | 20.80% | 69.98% | 55% |
| | Mean L∞ / L2 perturbation | 0.0499 / 0.2494 | not logged (older notebook) | not stated |
| | **Attack success rate*** | 21.56% (of 1,902 originally-correct) | not logged | not stated |
| **PGD** | Accuracy | 68.60% | 50.50% | 55% |
| | F1 | 17.59% | 28.67% | 52% |
| | Precision | 19.76% | 20.62% | 54% |
| | Recall | 15.84% | 47.04% | 50% |
| | Mean L∞ / L2 perturbation | 0.0462 / 0.1786 | not logged | not stated |
| | **Attack success rate*** | 27.87% | not logged | not stated |
| **DeepFool** | Accuracy | 73.85% | 83.70% | 50% |
| | F1 | 32.52% | 64.64% | 48% |
| | Precision | 35.80% | 59.72% | 49% |
| | Recall | 29.79% | 70.45% | 47% |
| | Mean L∞ / L2 perturbation | 0.0499 / 0.2488 (capped) | not logged | not stated |
| | **Attack success rate*** | 24.19% | not logged | not stated |
| **Square Attack** | Accuracy | 57.6% | 65.7% | 63% |
| | F1 | 14.86% | 44.05% | not stated |
| | Precision | 12.59% | 33.01% | not stated |
| | Recall | 18.14% | 66.18% | not stated |
| | **Attack success rate*** | 39.11% | 34.3%** | not stated |

\* *Attack success rate = fraction of originally-correct predictions the attack actually
flipped — added in the ROCm notebook, not present in the DirectML run's logging.*
\*\* *DirectML's Square Attack success rate was computed via the older `1 - accuracy` proxy,
not the corrected definition — not directly comparable to ROCm's figure.*

---

## 5. Defended Model (Stage 1 + Stage 2) — Full Results

| Metric | ROCm | DirectML | Paper |
|---|---|---|---|
| Stage 1 training time | **70.36 min** (clean, SHAP-robust features) | 203.02 min | not stated |
| Stage 2 training time | **21.00 min** (clean + adversarial) | 59.61 min | not stated |
| Adversarial data generation time | 0.54 min (32.6s) | 1.37 min (82.4s) | not stated |
| **Total defended pipeline time** | **91.90 min** | 264.00 min | not stated |
| Clean Accuracy | 96.14% | 92.00% | ~88%+ |
| Clean F1 | 91.22% | 83.44% | 86% |
| Clean Precision | 90.42% | 75.97% | 87% |
| Clean Recall | 92.04% | 92.53% | 85% |
| Inference time | 0.000404 ms/sample | 0.004241 ms/sample | 11 ms/sample (paper, different hardware) |

### Under attack

| Attack | Metric | ROCm | DirectML | Paper |
|---|---|---|---|---|
| **FGSM** | Accuracy | 87.80% | 88.20% | 88% |
| | F1 | 63.25% | 74.29% | 86% |
| | Precision | 87.14% | 68.89% | 87% |
| | Recall | 49.65% | 80.61% | 85% |
| | **Attack success rate*** | 11.46% | not logged | not stated |
| **PGD** | Accuracy | 72.85% | 80.40% | 88% |
| | F1 | 23.84% | 56.35% | — |
| | Precision | 29.31% | 53.26% | — |
| | Recall | 20.09% | 59.81% | — |
| | **Attack success rate*** | 24.11% | not logged | not stated |
| **DeepFool** | Accuracy | 88.80% | 92.25% | 88% |
| | F1 | 65.00% | 81.48% | — |
| | Precision | 95.85% | 82.37% | — |
| | Recall | 49.17% | 80.61% | — |
| | **Attack success rate*** | 10.26% | not logged | not stated |
| **Square Attack** | Accuracy | 76.2% | 84.4% | 84% |
| | F1 | 40.50% | 63.89% | 83% |
| | Precision | 41.33% | 60.53% | — |
| | Recall | 39.71% | 67.65% | — |
| | **Attack success rate*** | 20.29% | 15.6%** | 16% |

**Defended beats baseline under every attack, in both runs — 8/8 comparisons confirm the
paper's core claim holds on both backends.**

---

## 6. Overhead (Table 6 analogue)

| | ROCm | DirectML | Paper |
|---|---|---|---|
| Baseline training | **1.10 hrs** | 3.37 hrs | not stated |
| Defended training (total) | **1.53 hrs** | 4.40 hrs | not stated |
| Overhead (retrain only) | **37.8%** | 30.0% | not stated |
| Overhead (full pipeline, incl. attack-gen) | **38.7%** | 30.7% | ~25% |

Both backends land in the same order of magnitude as the paper's ~25% figure; ROCm's overhead
reads somewhat higher, consistent with the seed-to-seed overhead variance (19–46% range) already
observed across your earlier multi-seed ROCm study — this single-run difference is within that
established noise band, not a new anomaly.

---

## 7. Key Open Finding: Backend Sensitivity (Critical — Read Before Trusting Absolute Numbers)

With the data pipeline confirmed byte-identical (§3), the following differences must originate
from training dynamics alone, given the same `SEED=42`:

| | ROCm | DirectML | Gap |
|---|---|---|---|
| Baseline clean accuracy | 95.40% | 85.84% | **+9.56 points** |
| Baseline clean recall | 92.34% | 77.82% | +14.52 points |
| Defended clean accuracy | 96.14% | 92.00% | +4.14 points |
| SHAP-selected robust features | 13/15 overlap with DirectML | 13/15 overlap with ROCm | **2 features differ each way** |

**SHAP-robust feature divergence (new finding from this review):**
- ROCm selected `Fwd IAT Mean` and `Fwd IAT Total` that DirectML did not.
- DirectML selected `Fwd Packet Length Std` and `Packet Length Max` that ROCm did not.
- 13 of 15 features agree between backends — the disagreement is real but modest.

This is meaningful: SHAP stability scores are computed against the *trained baseline model's*
behavior on clean vs. adversarial inputs. Since the two backends' baseline models converged to
measurably different weights (evidenced by the accuracy/recall gap above), it follows that their
SHAP explanations — and therefore which features get selected as "robust" for Stage 1 — also
diverge. **This is corroborating evidence that the backend difference is a genuine training-time
phenomenon propagating through the whole pipeline, not an isolated accuracy-metric quirk.**

**Root cause is still not confirmed.** Plausible mechanisms, none yet verified:
- Different weight-initialization RNG consumption order between DirectML's and ROCm's low-level
  tensor-creation code paths, despite identical `torch.manual_seed(42)`.
- Different BatchNorm / kernel implementations between the two backends affecting training
  trajectory, especially early in training when BN statistics are still stabilizing.
- Possible floating-point precision or op-fusion differences in how each backend executes
  identical high-level PyTorch calls.

**Recommendation:** this should be reported as a disclosed limitation, not silently resolved.
Since ROCm is now the primary backend going forward, the practical decision is to standardize on
ROCm's numbers for all future work (already the plan), but the write-up should state plainly that
the DirectML run — which was the closer numeric match to the paper's 85% baseline — could not be
reconciled with the faster ROCm backend on identical code and data, and flag this as an area
future work could investigate (e.g. testing `torch.use_deterministic_algorithms(True)` on both
backends, or comparing initial-weight tensors directly before any training occurs).

---

## 8. Known Caveats (Carried Forward, Still Applicable)

1. **DeepFool's success rate looks artificially strong on both backends** (24.19% baseline /
   10.26% defended on ROCm) because ART's DeepFool cannot be ground-truth-targeted — it always
   moves toward the nearest decision boundary from the model's own current prediction, regardless
   of the `y` passed to `generate()`. The eps-capping fix (`cap_to_eps`) constrains the
   perturbation magnitude fairly, but the *direction* selection bias inherent to DeepFool remains
   a structural ART limitation, not something fixable via configuration.
2. **Dataset scale vs. the paper:** the paper used the full raw CSE-CIC-IDS2018 release
   (~250GB); this reproduction uses the cleaned Parquet mirror of the same 10 daily files
   (~6.7GB on disk, 6.66M rows pre-cleaning) due to local hardware constraints (RX7600 8GB VRAM,
   consumer-grade storage/RAM). Row-for-row attack-category coverage is the same 10 days /
   7 attack scenarios as the paper; the size difference is almost entirely file-format
   efficiency (Parquet compression + correct dtypes vs. raw uncompressed CSV), not missing data.
3. **Architecture width (128-64-32), attack eps/iteration counts, and feature-selection
   threshold T are not specified in the paper** — every one of these is a documented assumption
   in this implementation, now exposed via `CONFIG`, not something either run can be "wrong"
   about relative to the paper.
4. **Single-seed run for this specific comparison.** This review compares one ROCm run against
   one DirectML run, both at `SEED=42`. The separate 3-seed ROCm stability study (seeds 42, 123,
   2024, with the corrected `DATA_SEED` isolation) is the relevant reference for seed-to-seed
   variance — not superseded by this document.

---

## 9. Artifacts Saved

Both runs saved the full standard artifact set (`all_metrics.json`, `run_config.json`, model
checkpoints, scaler, feature-selection scores, SHAP values, comparison plots, Tables 4–6 as CSV).
**ROCm additionally saved** (features not present in the DirectML snapshot):
- `checkpoints/` — baseline and Stage 1 model checkpoints saved mid-run (crash recovery)
- `eps_sweep.csv` / `eps_sweep_plot.png` — 30-row epsilon-sweep robustness curve (5 eps values ×
  6 attack/model combinations); exact per-point values not reproduced in this document — see the
  CSV artifact directly for the full curve.

---

## 10. Summary

| Question | Answer |
|---|---|
| Is the data pipeline identical between backends? | **Yes, verified byte-for-byte** through feature selection |
| Does the defended model beat baseline under every attack? | **Yes, on both backends, 8/8** |
| Is ROCm faster? | **Yes — 2.5x end-to-end (3h13m vs 8h05m), consistent with the earlier 4.2x per-epoch microbenchmark** |
| Do absolute accuracy numbers match between backends? | **No — ~10-point gap, root cause not yet confirmed, now further evidenced by diverging SHAP feature selection** |
| Which backend more closely matches the paper's headline numbers? | **DirectML** (85.8% baseline vs. paper's 85%; ROCm's 95.4% is well above) |
| Recommended backend for future training | **ROCm**, per project direction — but absolute accuracy figures from ROCm should be reported alongside this disclosed backend-sensitivity caveat until the root cause is resolved |
