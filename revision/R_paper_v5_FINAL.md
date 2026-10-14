# Explainable Fracture Detection with ResNet50 and Grad-CAM: A Leak-Free Re-Evaluation on FracAtlas with Calibration, External Testing and Quantitative Explanation Validation

**Yug G. Lakhani**
Department of Computer Engineering, Gyanmanjari Innovative University, Bhavnagar, Gujarat, India
yglakhani@gmiu.edu.in · ORCID: 0009-0004-2930-4953

---

## Abstract

A previous version of this work reported 94.07% accuracy for a ResNet50 fracture classifier, but an audit found that 90.8% of its validation images had augmented copies in training, invalidating the estimate. This revision therefore discards that result and re-runs the study end-to-end under a leak-free protocol on a documented dataset. We use FracAtlas (4,083 radiographs, 717 fracture; CC BY 4.0), whose annotations were created by two expert radiologists and verified by an orthopedist. Images were grouped by MD5, perceptual hash (Hamming ≤ 4) and file-name stem — 4,083 images collapsed to 3,799 groups — and groups were split 70/15/15 into train/validation/test with no group crossing partitions. An annotation audit verified all 922 YOLO boxes (class 0 = fracture) and a 30-image author spot-check. A ResNet50 trained from random initialisation (ImageNet weights were not used) with class-weighted loss reached, on the untouched test set (n = 615), accuracy 75.6%, sensitivity 48.3%, specificity 82.2% and AUC 0.737 at the default 0.5 threshold. Temperature scaling (T = 0.66) reduced validation ECE from 0.178 to 0.147; a threshold chosen on validation for a pre-declared target sensitivity of 0.95 (thr = 0.073) achieved sensitivity 1.00 on test but specificity of only 0.10, showing that thresholding cannot rescue weak discrimination. On an external Kaggle-derived cohort (n = 142, hash-decontaminated) AUC fell to 0.418, while a second fracture-only external cohort (n = 2,384) gave sensitivity 0.56 at 0.5. Quantitatively, Grad-CAM did not localise fractures: pointing-game accuracy 0/120, mean energy inside the expert box 0.8% versus a 1.0% chance level, mean IoU 0.009; maps did change under weight randomisation (Spearman −0.27). Source identity was predictable from model embeddings (93.3% vs 81.2% chance). The corrected protocol thus reverses the original conclusion: with leakage removed, this model, dataset size and training regime do not support a clinical performance claim. All code, manifests, predictions and audit artefacts are released.

**Index Terms—** Fracture detection, ResNet50, Grad-CAM, data leakage, calibration, external validation, explainable AI, FracAtlas.

---

## I. Introduction

Missed fractures are a recognised source of diagnostic error on radiographs [1], [2], and deep-learning reviews report pooled sensitivities/specificities of about 91–94%/91–93% [1], [6], [7], alongside limited external validation, inconsistent reporting, and rarely-validated explanations. Our earlier prototype (fine-tuned ResNet50, Grad-CAM, FastAPI/React application) reported 94.07% validation accuracy. Auditing that evaluation, we found it had been performed on the same set used for checkpoint selection, and that 1,268 of 1,396 validation file-name stems (90.8%) also occurred in training, consistent with augmented copies of single radiographs crossing a random per-image split. That estimate was therefore withdrawn.

This paper is the corrected study. It asks: (RQ1) what performance does the same architecture achieve when train/validation/test are separated at source-image level and the test set is touched once? (RQ2) do calibrated probabilities and a threshold selected for a pre-declared target sensitivity behave acceptably on test and external data? (RQ3) does Grad-CAM actually localise expert-annotated fractures, and is source identity confounded with the learned representation? Contributions: (i) a leak-free grouped re-evaluation with a full audit trail; (ii) calibration and target-sensitivity thresholding with frozen operating point; (iii) two external evaluations; (iv) quantitative Grad-CAM validation against expert boxes, including a randomisation sanity check; (v) source-confounding analyses. The answers are largely negative, and are reported as such: that is the point of performing the corrections.

## II. Related Work

Kuo et al. [1] (42 studies) reported pooled sensitivity/specificity of 92%/91% internally and 91%/91% externally; Jung et al. [6] reported 94%/92% for radiographs; Hansen et al. [7] 92%/93% for wrist fractures with wide intervals. MURA [9] labels abnormality rather than fracture; FracAtlas [10] provides 4,083 radiographs with classification, localisation and segmentation annotations from expert annotators. Transfer learning is standard [4], [11], [12], and reported accuracies vary widely across protocols [13]–[15]. Grad-CAM [8] has been used mostly qualitatively in fracture work [12]; on chest radiographs saliency methods localised far worse than humans [18]. External performance commonly drops relative to internal estimates (e.g. AUC 0.95 → 0.86) [19]; dataset pooling and duplicates are a documented leakage source [26]; source shortcuts are a documented failure mode [27], [28]; and saliency maps require sanity checks [31] and calibration requires explicit assessment [29].

## III. Methods

### A. Dataset and provenance

The primary dataset is **FracAtlas** [10]: 4,083 musculoskeletal radiographs (717 fracture, 3,366 non-fracture), with COCO, PASCAL VOC, VGG and YOLO annotations, licensed CC BY 4.0 (Figshare doi:10.6084/m9.figshare.22363012; downloaded 14 Oct 2026 via a GitHub mirror). Per its publication, annotations were produced by two expert radiologists and verified/merged by an orthopedist. Metadata (anatomical region flags hand/leg/hip/shoulder/mixed, view, fracture count) come from the released `dataset.csv`. Every dataset used is recorded in a registry (name, URL, licence, date, anatomy, counts, annotation format) released with the code; the original undocumented Kaggle pool of the earlier prototype was abandoned rather than re-described. Two external cohorts were used, neither for training nor selection: **External-1**, a Kaggle bone-fracture collection distributed via GitHub (94 fractured, 48 healthy images; licence not stated by its mirror; used for evaluation only and not redistributed); and **External-2**, the "Simple/Comminuted vs Non-fracture" collection (GitHub), of which only the fracture folders were usable (Section III-D).

### B. Annotation verification

All 4,084 YOLO files were parsed automatically. Every box line was required to carry a class ID present in the verified mapping {0: fracture} taken from the dataset documentation, with normalised in-bounds coordinates. Result: 717 files with valid boxes, 922 boxes, all class 0, 3,366 empty files (non-fracture), and one stray non-annotation file (`labels.txt`); one `dataset.csv`-versus-folder label disagreement (IMG0003298.jpg) was found, with the folder label retained and recorded. From the stratified random review sample (seed 42), the author visually inspected 30 boxed radiographs: in all 30, the box overlaid bone at a plausible fracture or fixation site. This author check is not a clinical diagnosis; the clinical authority of the labels rests on the dataset's expert annotation and orthopedist verification [10].

### C. Leak-free grouped split

Images were grouped by union–find over (i) identical MD5, (ii) 64-bit perceptual hash within Hamming distance 4, and (iii) file-name stem. The 4,083 images formed 3,799 groups (15 MD5, 81 exact-pHash and 188 near-pHash merges beyond stems). Groups were split 70/15/15 (seed 42) with an assertion that no group spans partitions (Table I).

**TABLE I. Grouped split of FracAtlas**

| Split | Fracture | No fracture | Total | Groups |
|---|---:|---:|---:|---:|
| Train | 476 | 2,379 | 2,855 | 2,659 |
| Validation | 121 | 492 | 613 | 570 |
| Test | 120 | 495 | 615 | 570 |

### D. External-set decontamination

Every external image was screened against all 4,083 FracAtlas images by MD5 and perceptual hash (≤ 4 bits). External-1: no overlaps (94/48 retained). External-2: its entire Non-fracture folder (3,366 images) proved MD5-identical to FracAtlas images and was excluded; its fracture folders (1,211 simple + 1,173 comminuted) contained no overlap and were retained as a fracture-only cohort (sensitivity is the only defined metric there).

### E. Model and training

ResNet50 [20] with a two-unit head (fracture positive), 224×224 inputs, ImageNet normalisation, training augmentation of horizontal flip and rotation ≤ 10°. Because pretrained-weight downloads were unavailable in the compute environment, the network was trained **from random initialisation** — a deliberate, stated deviation from common practice [4], [11] that likely depresses performance and is treated as a limitation. Adam (lr 1e-4), class-weighted cross-entropy (weights 3.00/0.60 from train counts), batch 8, five epochs on CPU (52 min), seed 0. The checkpoint with the highest validation F1 (epoch 4) was selected; the test set was then evaluated once. A second fine-tuning stage at higher learning rate degraded validation F1 and was rejected on validation evidence alone. Predictions (per-image probabilities) were saved for validation, test and external sets.

### F. Calibration and thresholding

A temperature fitted on validation logits minimised negative log-likelihood. On calibrated validation probabilities, the operating threshold was the one reaching the pre-declared target sensitivity ≥ 0.95 with maximal specificity. Temperature and threshold were frozen before any test/external use. We report ECE (15 bins), Brier score, and group-cluster bootstrap 95% CIs (500 resamples).

### G. Grad-CAM validation and source confounding

Grad-CAM [8] at `layer4` was generated for all 120 test fracture images (159 expert boxes from the PASCAL VOC files). Metrics: pointing game, CAM energy inside the box versus the box-area chance level, IoU at 0.5-of-max, and Spearman correlation with maps from a weight-randomised copy (layer4 + head re-initialised) [31]. Source confounding was assessed by per-region performance, source predictability from penultimate embeddings (PCA-64 + cross-validated logistic regression against majority-class chance), and the external evaluations.

## IV. Results

### A. Held-out test performance (RQ1)

Training selected epoch 4 (validation F1 0.472, AUC 0.703). On the untouched test set at the default threshold (Table II): accuracy 75.6%, sensitivity 48.3% (58/120 fractures found), specificity 82.2%, precision 39.7%, F1 0.436, AUC 0.737 (bootstrap 95% CI at the calibrated operating point for AUC: 0.687–0.786).

**TABLE II. Test-set confusion matrix (threshold 0.5)**

| True \ Predicted | Fracture | No fracture |
|---|---:|---:|
| Fracture (120) | 58 | 62 |
| No fracture (495) | 88 | 407 |

### B. Calibration and target-sensitivity threshold (RQ2)

Temperature T = 0.66 reduced validation ECE from 0.178 to 0.147 (Brier 0.166) — calibration remained poor. Meeting the pre-declared sensitivity target on validation required a threshold of 0.073, at which validation specificity was only 0.112. Frozen and applied to test, that threshold produced sensitivity 1.000 but specificity 0.103 and accuracy 27.8% (95% CI 24.3–31.3%). In other words, with AUC ≈ 0.74 there is no operating point that is both sensitive and specific; thresholding exposes, rather than fixes, weak discrimination.

### C. External evaluation (RQ2)

External-1 (n = 142) at 0.5: accuracy 29.6%, sensitivity 3.2%, specificity 81.3%, AUC 0.418 — below chance ranking, i.e. a substantial domain shift between FracAtlas and this Kaggle collection (different cropping, presentation and labelling conventions). At the frozen sensitive threshold, external sensitivity rose only to 56.4% (specificity 37.5%, ECE 0.569). External-2 (2,384 fracture images): sensitivity 55.8% at 0.5 and 99.6% at the frozen threshold. Internal (0.737) and external (0.418) AUCs bracket the degradation reported in the literature [19] and exceed it; no generalisation claim survives.

### D. Grad-CAM validation (RQ3)

Across the 120 boxed test fractures: pointing-game accuracy **0.0%**, mean CAM energy inside the expert box **0.8%** against a **1.0%** box-area chance level, mean IoU **0.009**. The explanations therefore do not localise fractures in this model — they are, at best, diffuse region highlights, consistent with the cautionary benchmark of Saporta et al. [18]. The sanity check passed in form only: randomised-weight maps correlated −0.27 (Spearman) with the originals, so the maps do depend on the learned weights — they simply do not point at the fracture.

### E. Source confounding (RQ3)

At threshold 0.5, per-region test sensitivity varied widely (hand 68.6%, leg 20.8%, hip and shoulder 0% on very small fracture counts of 21 and 11 images respectively — interpretable only as instability warnings). Source dataset (FracAtlas vs External-1) was predictable from the model's embeddings with 93.3% cross-validated accuracy against an 81.2% majority-class chance level, confirming that the representation encodes acquisition/source characteristics — a concrete shortcut risk of the kind described in [27], [28], and a plausible contributor to the external collapse.

## V. Discussion

Removing leakage reverses the headline of the earlier prototype: 94.07% (leaky development estimate) becomes AUC 0.737 on a grouped held-out test, sensitivity below 50% at the default threshold, near-chance external ranking, and Grad-CAM localisation at or below chance. Three lessons follow. First, the dominant factor was evaluation hygiene, not architecture: the same ResNet50 family spans 94% and 74% depending only on split integrity, which is why grouped splitting and hash decontamination (Sections III-C/D) should precede any modelling claim on pooled public data. Second, calibration and threshold selection are diagnostic tools as much as remedies: ECE stayed high (0.147) and the sensitive operating point cost 90% of specificity, correctly signalling that the underlying ranking is weak. Third, an explanation method that passes a randomisation check can still be clinically meaningless; quantitative box-level validation (Section III-G) is the minimum standard before heat maps are shown to users as "where the model looks".

The weak absolute performance has identifiable causes, stated rather than hidden: training from random initialisation instead of ImageNet pretraining, a single seed, only 476 fracture training images, batch size 8 with trainable BatchNorm, and a CPU-scale five-epoch budget. Each is fixable; none licenses re-interpreting the present numbers upward. The web application of the earlier work (FastAPI + React + Grad-CAM display) remains a functioning prototype, but with these results it must not present its output as decision support.

## VI. Limitations

(1) A single run and seed; run-to-run variability is unmeasured (multi-seed aggregation tooling is released, but time constraints limited this study to one seed). (2) Random rather than ImageNet initialisation likely lowers performance relative to standard practice. (3) FracAtlas provides no patient identifiers, so grouping is by source image, not patient; images of one patient could still cross partitions. (4) External-1 is small (n = 142) with an unstated licence and documentation in its mirror, and External-2 is fracture-only; both are reported with those caveats and neither was redistributed. (5) The annotation spot-check was performed by the author, not an independent clinician; the clinical grounding relies on FracAtlas's expert annotation process. (6) Per-region fracture counts for hip and shoulder are too small for stable estimates. (7) Calibration, threshold and Grad-CAM metrics inherit the model's weakness; they characterise this model and must be re-measured for any retrained version.

## VII. Conclusion

Under a leak-free, source-grouped protocol on FracAtlas, a from-scratch ResNet50 reaches AUC 0.737 on held-out test data, cannot meet a 0.95 sensitivity target without collapsing specificity, fails to generalise to external cohorts (AUC 0.418), and produces Grad-CAM maps that do not localise expert-annotated fractures. Every claim in this paper is accompanied by the artefact that generated it. The practical contribution is the verified pipeline — grouped perceptual-hash splitting, annotation auditing, provenance registration, frozen calibration/thresholding, external testing and quantitative explanation validation — together with a clear demonstration of why the corrected numbers, not the leaky 94%, are the ones that can be believed. Repeating this protocol with ImageNet-pretrained weights, multiple seeds and a larger fracture sample is the necessary next step before any performance, let alone clinical, claim.

## Data and Code Availability

Code (split, audit, training, calibration, Grad-CAM and confounding tools and this study's driver configuration), the dataset registry, split manifests, per-image predictions, calibration and Grad-CAM summaries, and audit outputs are available at https://github.com/Unusualyug/fracture_detection_. FracAtlas is available from its publishers under CC BY 4.0 (doi:10.6084/m9.figshare.22363012); external-cohort images are not redistributed. The trained checkpoint was produced in a CPU environment without pretrained weights, as described. This system is a research prototype and is not intended for clinical use.

## References

[1] R. Y. L. Kuo et al., "Artificial intelligence in fracture detection: a systematic review and meta-analysis," *Radiology*, vol. 304, no. 1, pp. 50–62, 2022, doi: 10.1148/radiol.211785.
[2] A. Guermazi et al., "Improving radiographic fracture recognition performance and efficiency using artificial intelligence," *Radiology*, vol. 302, no. 3, pp. 627–636, 2022, doi: 10.1148/radiol.210937.
[3] G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017, doi: 10.1016/j.media.2017.07.005.
[4] J. Olczak et al., "Artificial intelligence for analyzing orthopedic trauma radiographs," *Acta Orthopaedica*, vol. 88, no. 6, pp. 581–586, 2017, doi: 10.1080/17453674.2017.1344459.
[5] P. H. S. Kalmet et al., "Deep learning in fracture detection: a narrative review," *Acta Orthopaedica*, vol. 91, no. 2, pp. 215–220, 2020, doi: 10.1080/17453674.2019.1711323.
[6] J. Jung, J. Dai, B. Liu, and Q. Wu, "Artificial intelligence in fracture detection with different image modalities and data types," *PLOS Digital Health*, vol. 3, no. 1, e0000438, 2024, doi: 10.1371/journal.pdig.0000438.
[7] V. Hansen et al., "Deep learning performance compared to healthcare experts in detecting wrist fractures from radiographs," *European Journal of Radiology*, vol. 174, Art. no. 111399, 2024, doi: 10.1016/j.ejrad.2024.111399.
[8] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.
[9] P. Rajpurkar et al., "MURA: Large dataset for abnormality detection in musculoskeletal radiographs," arXiv:1712.06957, 2017.
[10] I. Abedeen et al., "FracAtlas: A dataset for fracture classification, localization and segmentation of musculoskeletal radiographs," *Scientific Data*, vol. 10, Art. no. 521, 2023, doi: 10.1038/s41597-023-02432-4.
[11] D. H. Kim and T. MacKinnon, "Artificial intelligence in fracture detection: transfer learning from deep convolutional neural networks," *Clinical Radiology*, vol. 73, no. 5, pp. 439–445, 2018, doi: 10.1016/j.crad.2017.11.015.
[12] H. Wang et al., "Harnessing ResNet50 and SENet for enhanced ankle fracture identification," *BMC Musculoskeletal Disorders*, vol. 25, Art. no. 250, 2024, doi: 10.1186/s12891-024-07355-8.
[13] A. Tahir et al., "Enhancing diagnosis: ensemble deep-learning model for fracture detection using X-ray images," *Clinical Radiology*, vol. 79, no. 11, pp. e1394–e1402, 2024, doi: 10.1016/j.crad.2024.08.006.
[14] T. Aldhyani et al., "Diagnosis and detection of bone fracture in radiographic images using deep learning approaches," *Frontiers in Medicine*, vol. 11, Art. no. 1506686, 2025, doi: 10.3389/fmed.2024.1506686.
[15] A. Alam et al., "Novel transfer learning based bone fracture detection using radiographic images," *BMC Medical Imaging*, vol. 25, Art. no. 5, 2025, doi: 10.1186/s12880-024-01546-4.
[16] M. He, X. Wang, and Y. Zhao, "A calibrated deep learning ensemble for abnormality detection in musculoskeletal radiographs," *Scientific Reports*, vol. 11, Art. no. 9097, 2021, doi: 10.1038/s41598-021-88578-w.
[17] Y. L. Thian et al., "Convolutional neural networks for automated fracture detection and localization on wrist radiographs," *Radiology: Artificial Intelligence*, vol. 1, no. 1, Art. no. e180001, 2019, doi: 10.1148/ryai.2019180001.
[18] A. Saporta et al., "Benchmarking saliency methods for chest X-ray interpretation," *Nature Machine Intelligence*, vol. 4, pp. 867–878, 2022.
[19] J. Olczak et al., "External validation of an artificial intelligence multi-label deep learning model capable of ankle fracture classification," *BMC Musculoskeletal Disorders*, vol. 25, Art. no. 788, 2024, doi: 10.1186/s12891-024-07884-2.
[20] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.
[21] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248–255.
[22] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.
[23] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in NeurIPS*, vol. 32, 2019.
[24] S. Ioffe and C. Szegedy, "Batch normalization," in *Proc. ICML*, 2015, pp. 448–456.
[25] Y. Wu and K. He, "Group normalization," in *Proc. ECCV*, 2018, pp. 3–19.
[26] M. Roberts et al., "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans," *Nature Machine Intelligence*, vol. 3, pp. 199–217, 2021.
[27] J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs," *PLOS Medicine*, vol. 15, no. 11, Art. no. e1002683, 2018.
[28] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nature Machine Intelligence*, vol. 3, pp. 610–619, 2021.
[29] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017, pp. 1321–1330.
[30] J. Zhang et al., "Top-down neural attention by excitation backprop," *Int. J. Computer Vision*, vol. 126, pp. 1084–1102, 2018.
[31] J. Adebayo et al., "Sanity checks for saliency maps," in *Advances in NeurIPS*, vol. 31, 2018.
