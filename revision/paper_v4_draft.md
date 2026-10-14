# Explainable Fracture Detection with a Fine-Tuned ResNet50 and Grad-CAM: Leak-Free Re-Evaluation, Calibration and Quantitative Explanation Validation

**Author:** Yug G. Lakhani, Department of Computer Engineering, Gyanmanjari Innovative University (yglakhani@gmiu.edu.in, ORCID 0009-0004-2930-4953)

> **DRAFT v4 — status: METHODS REVISED, EXPERIMENTS NOT YET RUN.**
> Every value that depends on the retrained model is marked `[PENDING: …]`
> and must be filled only from files under `results/` produced by the
> commands in `revision/RUNBOOK.md`. The v3 validation figures (94.07%
> accuracy etc.) must **not** be carried into the Results of this version:
> they were measured on a set with 90.8% train–validation source overlap.
> Text that is final (problem, audit of v3, protocol) is written as final.

---

## Abstract
[Rewrite after experiments. Structure: (1) v3 prototype and its leakage audit
(90.8% of validation file-name stems also in training); (2) this revision:
source-grouped, perceptual-hash-deduplicated train/validation/test split,
verified annotation class IDs with an independent spot-check of
`[PENDING: n]` labels/boxes by `[PENDING: reviewer role]`, documented
per-source datasets; (3) results on the held-out test set and on
`[PENDING: external dataset name]`: accuracy `[PENDING]`, sensitivity at the
pre-declared operating point `[PENDING]`, AUC `[PENDING]`, ECE before/after
temperature scaling `[PENDING]`; (4) quantitative Grad-CAM validation:
pointing-game accuracy `[PENDING]`, energy-in-box vs chance `[PENDING]`;
(5) conclusion bounded by what the external result supports.]

**Index Terms—** Fracture detection, ResNet50, transfer learning, Grad-CAM, explainable AI, data leakage, calibration, radiographs, web deployment.

## I. Introduction
[Carry over from v3, with the final paragraph replaced by:]
This revision re-evaluates the v3 prototype under a leak-free protocol. The
contributions are: (1) a source-grouped, hash-deduplicated split with a
held-out test set that is evaluated once; (2) verified labels and bounding
boxes, with a documented human spot-check; (3) calibrated probabilities and
an operating threshold selected on validation data for a pre-declared
target sensitivity of 0.95; (4) quantitative validation of Grad-CAM against
annotated boxes, together with source-confounding tests; (5) per-source and
external-dataset evaluation. The web application (FastAPI/React, Grad-CAM,
manual-review flag) is unchanged from v3.

## II. Related Work
[Carry over from v3 unchanged, adding calibration (Guo et al.) and
saliency-sanity-check (Adebayo et al.) citations already in the v3 list.]

## III. Research Gap and Questions
[Carry over; RQ1–RQ3 now answerable: RQ1 = test/external performance under
the grouped split, RQ3 = which safeguards are now met (Table VI is replaced
by the "status after revision" table in §VI).]

## IV. Method

### A. Datasets and provenance
All source datasets are recorded in the dataset registry (name, URL,
licence, download date, anatomical regions, image counts, annotation
format, patient-ID availability): `[PENDING: registry table generated from
datasets/dataset_registry.csv]`. Every pooled image is traced to one source
via the data inventory (`datasets/data_inventory.csv`). Images are not
redistributed; licences are as recorded in the registry.

### B. Annotation verification
Annotation files are YOLO-format text files. Class IDs were verified
against each source's own documentation (`datasets/class_mapping.json`);
IDs outside the mapping are reported as issues, never silently treated as
fracture (the v3 rule). `scripts/verify_annotations.py` produced:
`[PENDING: n files, n valid boxes, n issue rows by type, class distribution]`.
A stratified random sample of `[PENDING: n]` annotated images was
independently reviewed by `[PENDING: name/role, e.g. consultant
radiologist]`; label agreement was `[PENDING: %]`, boxes judged correct in
`[PENDING: %]`, and `[PENDING: corrections applied / exclusion rule]`.

### C. Leak-free split
Images were grouped by union-find over (i) MD5, (ii) 64-bit perceptual
hash within Hamming distance 4, and (iii) file-name stem before the
Roboflow `.rf.` suffix. Groups — never individual images — were split
70/15/15 (seed 42). Result: `[PENDING: n images → n groups; per-partition
and per-source counts table from results/split_summary.json]`; the integrity
check confirms no group spans two partitions.

### D–F. Preprocessing, model, training
[As v3 §IV-B–D, updated: batch size `[PENDING]`, seeds 0/1/2, checkpoint =
best validation F1, saved checkpoint + per-image prediction CSVs. State
hardware and wall-time from the run JSONs.]

### G. Calibration and thresholding
A temperature fitted on validation logits minimised NLL
(ECE `[PENDING: before → after]`). The operating threshold
`[PENDING: value]` is the lowest calibrated threshold reaching sensitivity
≥ 0.95 on validation (specificity then `[PENDING]`), frozen before any
test/external evaluation. The 0.80 manual-review flag of v3 is replaced by
this threshold, or retained only as `[PENDING: justify or remove]`.

### H. Evaluation protocol
Test set evaluated once per run; external dataset
(`[PENDING: name, registry row]`) never used for training or selection.
Metrics: accuracy, sensitivity, specificity, PPV, NPV, F1, AUC, ECE, Brier —
overall and per source, with group-cluster bootstrap 95% CIs.
Grad-CAM (unchanged `layer4` setup) is validated on test fracture images
with verified boxes: pointing game, CAM energy inside box vs box-area
chance level, IoU at 0.5-of-max, and a weight-randomisation sanity check
(Spearman CAM correlation `[PENDING]`). Source-confounding: per-source
sensitivity spread, leave-one-source-out estimates, source predictability
from embeddings (CV accuracy vs majority chance `[PENDING]`), and
border/text-marker ablation with McNemar's test (`[PENDING]`).

## V. Results
### A. Training behaviour — `[PENDING: history from run JSONs]`
### B. Held-out test performance — `[PENDING: Table from results/summary_test.csv + calibration_threshold.json, mean ± SD over seeds, with CIs]`
### C. External evaluation — `[PENDING: same metrics on external set; state the internal→external gap honestly]`
### D. Calibration — `[PENDING: T, ECE/Brier before-after, reliability figure]`
### E. Grad-CAM validation — `[PENDING: gradcam_summary.json incl. per-source]`
### F. Source-confounding — `[PENDING: source_confounding.json]`

## VI. Discussion
[Rewrite around the *new* numbers only: compare test (not validation)
figures with the pooled literature estimates, interpret the external gap
against the 0.95 → 0.86 AUC drop reported in the literature, report where
the target sensitivity was/was not met on test/external, and what the
Grad-CAM metrics do and do not license. Replace v3 Table VI with:]

| Safeguard | v3 | This revision |
|---|---|---|
| Patient/source-level split | Not applied | Grouped by MD5+pHash+stem; `[PENDING: verify]` |
| Independent test set | None | Yes, evaluated once |
| External validation | None | `[PENDING: dataset + result]` |
| Verified labels/boxes | No | `[PENDING: audit + reviewer outcome]` |
| Provenance/licences | Not recorded | Registry: `[PENDING]` |
| Calibration / target-sensitivity threshold | No | Temperature + frozen threshold |
| Quantitative Grad-CAM validation | No | Pointing game / energy / IoU / randomisation |
| Repeated runs / seeds | Single run | Seeds 0–2, mean ± SD |

## VII. Limitations
[Keep only limitations that remain true after the rerun, e.g.: patient IDs
unavailable so grouping is by source image, not patient; single external
dataset; reviewer was/was not a clinician; binary task; per-region results
only where anatomy is documented. Delete limitations the revision fixed —
do not keep claiming a flaw that the new protocol removed, and do not claim
a fix that the `[PENDING]` items have not yet evidenced.]

## VIII. Conclusion and Future Work
[One paragraph bounded by the external result; next steps that remain:
patient-level grouping where IDs exist, reader study, latency measurement,
prospective evaluation.]

## Data and Code Availability
[As v3, updated: code + split manifests + prediction CSVs + audit outputs
in the repository; images not redistributed (licences in the registry).]

## References
[v3 list; add dataset citations from the completed registry.]
