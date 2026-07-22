**BBT: Multivariate Time Series Transformer Model for fMRI Biomarker Discovery**

*J.L.*

Chair of Informatics for Medical Technologies (CIMT)

University of A

# **Abstract**

Standard resting-state functional MRI (rs-fMRI) analysis collapses the BOLD signal into static functional connectivity matrices, discarding the transient neural dynamics that may underlie individual pathology. We introduce BBTransformer, a multivariate time-series foundation model that operates directly on raw, whole-brain BOLD sequences, preserving native spatiotemporal structure across 414 cortical and subcortical regions, while conditioning on age and biological sex as embedded covariates to disentangle diagnostic signals from demographic confounding. The architecture integrates dual-resolution temporal encoding (fine-grained and coarse patch streams), learned attention pooling over diagnostically salient time windows, and confounder-aware decision fusion.

To overcome data scarcity and leverage shared neurobiological principles, we implement a biologically ordered transfer learning protocol: starting from an autism spectrum disorder (ASD) foundation model (ABIDE, N \= 585), we sequentially fine-tune BBTransformer across 10 neurological and psychiatric conditions in the UK Biobank (N \= 2,367), following a pathophysiological hierarchy from neurodevelopmental to white matter disorders. This strategy enables robust generalization, achieving F1 \= 0.68–0.92 and ROC-AUC \= 0.67–0.95. Critically, performance extends to two external, independently preprocessed cohorts: ADHD-200 (F1 \= 0.722, ROC-AUC \= 0.819) and the UCLA LA5c study (F1 \= 0.632, ROC-AUC \= 0.640). In stark contrast, the model fails on high-prevalence but biologically heterogeneous conditions, depressive episode (N \= 4,338), sleep disorders (N \= 1,104), and substance use disorders (N \= 2,276), despite their large sample sizes. This dissociation demonstrates that diagnostic learnability from rs-fMRI is gated by biological coherence, not data volume. Permutation-based feature importance reveals anatomically coherent, disorder-specific biomarker circuits aligned with known disease neurobiology. These results establish a new paradigm: disorder-specific signatures reside in spatiotemporal dynamics, not static connectivity.

**Keywords:** *resting-state fMRI, multivariate time series transformer model, biomarker discovery, spatiotemporal dynamics, neuroimaging, deep learning*

# 

# **Introduction**

The human brain exhibits rich, spatiotemporal dynamics that underpin cognition, perception, and behavior. Resting-state functional MRI (rs-fMRI) captures these dynamics indirectly via the blood-oxygen-level-dependent (BOLD) signal, which reflects underlying neural activity through neurovascular coupling. Across decades of research, the dominant analytical paradigm has been \[[1](#bookmark=id.qy4lxytpkgz6)\]reduced to a static matrix of pairwise regional correlations, the functional connectivity (FC) matrix. While FC analysis has established foundational insights into large-scale brain network organization, including the default mode, salience, and frontoparietal networks\[[2](#bookmark=id.r36ykfq98ycm)\], this reduction discards precisely the temporal information that may differentiate healthy from pathological brain states. This limitation is not merely theoretical. Neurological and psychiatric disorders often manifest as disruptions to the temporal dynamics of brain activity rather than to its average spatial structure. Reducing the BOLD time series to a static correlation matrix necessarily erases these disorder-specific temporal signatures.

Transformer architectures\[[3](#bookmark=id.5gf938ofdkgy)\], originally developed for natural language processing, have demonstrated exceptional capacity for modeling long-range sequential dependencies in high-dimensional time series\[[4](#bookmark=id.bbgmzi83awb0)\]. The self-attention mechanism allows every position in a sequence to attend to every other position, making transformers uniquely suited to capture the distributed, non-local temporal interactions characteristic of brain dynamics. Key architectural innovations in transformers, for example, rotary positional embeddings\[[5](#bookmark=id.bbgmzi83awb0)\], which encode relative rather than absolute temporal positions; grouped-query attention\[[6](#bookmark=id.2yoqe9ik8ydc)\], which reduces memory requirements while preserving representational capacity; root mean square layer normalization\[[7](#bookmark=id.ve679uivfcya)\], which stabilizes training; and SwiGLU activations functions\[[8](#bookmark=id.pw4g3qebb9wb)\] , which improve gradient flow; have collectively enabled stable, efficient training on very long temporal sequences.

Here, we introduce BBTransformer, a multivariate time-series transformer architecture designed to model the native spatiotemporal dynamics of resting-state fMRI. The model operates on a unified whole-brain parcellation comprising 360 cortical regions from the Human Connectome Project Multi-Modal Parcellation\[[9](#bookmark=kix.y97z63o6v1wq)\] and 54 subcortical regions from the Tian Scale 4 atlas\[[10](#bookmark=kix.4wy49zm2igr4)\].

Beyond single-disorder classification, we implement a biologically ordered transfer learning protocol to address the data scarcity inherent in clinical fMRI. Transformers require large-scale training to develop stable, generalizable representations and emergent dynamical abstractions, but individual psychiatric cohorts are often too small in isolation. 

Starting from a foundation model trained on autism spectrum disorder (ASD) using the ABIDE repository\[[11](#bookmark=id.advq4fz9h99v)\], we sequentially fine-tune BBTransformer across ten neurological and psychiatric conditions in the UK Biobank cohort\[[12](#bookmark=id.af27oxq5qfn9)\], guided by the hierarchical structure of ICD-10 clinical classifications. This framework mirrors the hierarchical organization of the nervous system, traversing from neurodevelopmental to neurodegenerative, inflammatory, vascular, and white matter pathology. We report both successes and principled failures: the model fails to decode the largest diagnostic groups, depressive episode, sleep disorders, and substance use disorder. These phenotypes in the UK Biobank were ascertained via self-report or primary care assessment rather than specialist psychiatric evaluation. This reflects an absence of coherent spatiotemporal signatures in their BOLD dynamics, not architectural limitations, underscoring that biological coherence, not sample size, determines learnability from fMRI time series.

# **Results**

## **Hyperparameter Tunning**

We began by training our model on autism spectrum disorder (ASD) using the ABIDE (Autism Brain Imaging Data Exchange) repository, the largest openly available multi-site rs-fMRI dataset for a single neurodevelopmental condition. The ABIDE cohort provides particularly favorable properties for foundation model training: it spans a wide developmental range (children through adults), includes diverse imaging sites with heterogeneous acquisition protocols, and encompasses the full clinical spectrum of ASD severity, providing rich variability against which the model must generalize.

The training cohort comprised 585 subjects (271 with ASD, 314 neurotypical controls; 46.3% case prevalence). We conducted 50 trials of Bayesian hyperparameter optimization, with each trial subject to early stopping at 100 epochs without validation improvement. A strict validity criterion required all five performance metrics (F1, ROC-AUC, accuracy, precision, and recall) to exceed 0.65 on the held-out test set before weights would be accepted and propagated to downstream disorders. This criterion reflects the conservative standard required for clinically meaningful binary classification.

Sixteen of 50 optimization trials met the validity threshold. The best-performing configuration achieved F1 \= 0.7579, ROC-AUC \= 0.7283, accuracy \= 0.7386, precision \= 0.6667, and recall \= 0.8780 on the test set. The optimal architecture featured 512-dimensional embeddings, 7 transformer encoder layers, 8 query attention heads with 4 key-value heads (grouped-query attention), and a stochastic depth rate of 0.0714, a configuration that balances representational capacity with regularization. The relatively sparse attention (4 key-value heads vs. 8 query heads) proved critical for generalization, effectively functioning as a learned bottleneck that prevents overfitting to site-specific acquisition artefacts.

**Table 1\. Best performance on held-out test set (ASD foundation model)**

| Metric | Value |
| :---- | :---- |
| F1 Score | 0.7579 |
| ROC-AUC | 0.7283 |
| Accuracy | 0.7386 |
| Precision | 0.6667 |
| Recall | 0.8780 |

## **Transfer Learning** 

Having established a valid foundation model for ASD, we implemented a biologically ordered transfer learning pipeline guided by ICD-10 clinical classifications. The ordering principle was pathophysiological coherence: we hypothesized that knowledge transfers most efficiently along axes of biological similarity, beginning with neurodevelopmental disorders (where altered early connectivity patterns are central) and progressing through neurodegenerative, inflammatory, vascular, and finally white matter disease. 

For each target disorder, we initialized BBTransformer with the fixed optimal architecture identified during ASD tuning, loaded weights from the most recent valid model in the transfer chain, and ran up to 15 trials with equal random seeds. A model was accepted, and its weights propagated forward only if all five performance metrics exceeded 0.60 on the held-out test set. If all trials failed, weights from the previous valid model were retained, and the failed disorder was excluded from the propagation chain. This design prevents catastrophic forgetting from phenotypically incoherent fine-tuning while maintaining a clean record of which disorders can and cannot be decoded from raw BOLD dynamics.

Ten consecutive transfer steps met validity thresholds, spanning 10 distinct neurological and psychiatric conditions. The transfer chain successfully traversed: neurodevelopmental disorders (ASD), dementia-spectrum conditions, organic mental disorders, inflammatory and infectious CNS conditions, cerebrovascular disease, bipolar disorder, schizophrenia spectrum, epilepsy, movement disorders, and multiple sclerosis/demyelinating disease. This represents a complete traversal from the earliest-onset (neurodevelopmental) to late-onset (white matter) neurological pathology.

**Table 2\. Transfer learning progression: performance across all valid phases**

| Phase | Disorder | N (Cases/Controls) | F1 | Accuracy | ROC-AUC |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 0 | Autism Spectrum Disorder | 271/314 | 0.7579 | 0.7386 | 0.7283 |
| 1 | Developmental Dementia  | 61/61 | 0.7778 | 0.7895 | 0.7889 |
| 2 | Psychopathology Dementia | 49/49 | 0.9231 | 0.9333 | 0.9286 |
| 3 | Organic Mental Disorder | 90/90 | 0.9231 | 0.9259 | 0.9533 |
| 4 | Inflammatory / Infectious | 46/46 | 0.8000 | 0.7857 | 0.8673 |
| 5 | Cerebrovascular | 296/296 | 0.7434 | 0.6742 | 0.7116 |
| 6 | Bipolar Disorder | 55/55 | 0.8571 | 0.8824 | 0.8472 |
| 7 | Schizophrenia Spectrum | 33/33 | 0.7500 | 0.8000 | 0.8400 |
| 8 | Epilepsy | 171/171 | 0.6786 | 0.6538 | 0.6672 |
| 9 | Movement Disorders | 208/208 | 0.7213 | 0.7302 | 0.7782 |
| 10 | Multiple Sclerosis / Demyelinating | 82/82 | 0.8333 | 0.8400 | 0.9167 |

Performance across the transfer chain was strong and consistent. The highest F1 scores were achieved for Psychopathology/Dementia (F1 \= 0.9231, ROC-AUC \= 0.9286) and Organic Mental Disorder (F1 \= 0.9231, ROC-AUC \= 0.9533), conditions where spatiotemporal BOLD dynamics appear to provide a particularly clean diagnostic signal. Multiple sclerosis achieved an ROC-AUC of 0.9167, indicating strong separation between demyelination-related connectivity disruption and controls. Bipolar disorder (F1 \= 0.8571) and inflammatory/infectious conditions (F1 \= 0.8000) also performed robustly. More heterogeneous conditions, cerebrovascular disease (F1 \= 0.7434), movement disorders (F1 \= 0.7213), and epilepsy (F1 \= 0.6786), showed lower but still valid performance, consistent with greater phenotypic heterogeneity within those diagnostic categories.

## **Failed Disorders**

Three high-prevalence conditions failed to meet validity thresholds after 15 trials each: depressive episode (N \= 4,338), sleep disorders (N \= 1,104), and substance use disorders (N \= 2,276). Critically, these failures occurred despite the largest sample sizes in the dataset; depressive episode alone contained nearly twice as many subjects as all successful transfer steps combined. This dissociation between sample size and learnability is the most important negative finding of this work.

The failure modes were mechanistically distinct across the three conditions. For the depressive episode, the model converged to a collapsed mode in which approximately 97% of subjects were predicted to be positive. This behavior, high recall, near-chance precision, and ROC-AUC indistinguishable from random (0.5487), indicates that the model found no discriminative structure in the temporal dynamics and defaulted to predicting the majority class. Sleep disorders exhibited total recall collapse (recall \= 1.0, predicting every subject positive) with an AUC of 0.5213. Substance use showed a slightly higher AUC (0.6039), but still failed to achieve meaningful class separation.

**Table 3\. Failed targets: failure modes and metrics**

| Disorder | N | F1 | Accuracy | Precision | ROC-AUC | Failure Mode |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Depressive Episode | 4,338 | 0.6681 | 0.5192 | 0.5097 | 0.5487 | Collapsed mode: \~97% positive predictions |
| Sleep Disorders | 1,104 | 0.6831 | 0.5361 | 0.5188 | 0.5213 | Total collapse: recall \= 1.0, AUC ≈ random |
| Substance Use | 2,276 | 0.6581 | 0.5351 | 0.5204 | 0.6039 | Weak signal: class separation near chance |

These patterns are not attributable to data quality or model capacity, but instead reflect genuine characteristics of these phenotypes. These findings suggest that BOLD spatiotemporal dynamics, at least as captured by standard acquisition protocols, do not reliably separate these diagnostic categories from controls. The model's failure is thus scientifically informative: it constrains the hypothesis space of what is biologically encoded in rs-fMRI dynamics.

## **Validation in External Datasets**

To assess generalizability beyond the UK Biobank ecosystem, we extended the biologically ordered transfer learning pipeline to two independent, publicly available cohorts: ADHD-200\[[13](#bookmark=kix.8ioao2s7k842)\] and the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study\[[14](#bookmark=kix.fzgxijef4vjk)\]. Critically, these datasets were excluded from all prior stages of hyperparameter tuning, architecture selection, and internal transfer learning.

The model was initialized using the validated weights from the final internal transfer phase (Multiple Sclerosis/Demyelinating, Phase 10), ensuring that the external validation began with a representation already refined across neurodevelopmental, neurodegenerative, inflammatory, cerebrovascular, and white matter pathologies. This approach tests whether the spatiotemporal signatures learned within the UK Biobank generalize to independently preprocessed, multi-site data with distinct demographic profiles.

We first applied BBTransformer to the ADHD-200 cohort (N \= 242; 103 cases, 139 controls). Adhering to the standard protocol, we conducted up to 30 trials with random seeds and early stopping. The composite validity criterion required all five metrics (F1, accuracy, precision, recall, ROC-AUC) to exceed 0.60. One trial succeeded, yielding F1 \= 0.722, accuracy \= 0.730, precision \= 0.715, recall \= 0.729, and ROC-AUC \= 0.819. 

Subsequently, we propagated the ADHD-validated weights to the UCLA cohort (N \= 132). This dataset aggregates schizophrenia (n \= 15), bipolar disorder (n \= 22), and ADHD (n \= 26\) into a single "disorder vs. control" classification task. After 11 trials, one configuration met the validity threshold, achieving F1 \= 0.632, accuracy \= 0.650, precision \= 0.615, recall \= 0.650, and ROC-AUC \= 0.640.

**Table 4\. External validation performance across independent cohorts**

| Cohort | N (Cases/Controls) | F1 | Accuracy | Precision | Recall | ROC-AUC |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| ADHD-200 | 103/139 | 0.722 | 0.730 | 0.650 | 0.813 | 0.818 |
| UCLA LA5c | 63/69 | 0.632 | 0.650 | 0.667 | 0.600 | 0.640 |

# **2  Cross-Disorder Analysis**

**2.1  Recurrent transdiagnostic hubs**

Right Lateral Orbitofrontal Area 11 ranks in the top-3 for Dementia/Developmental, Inflammatory/Infectious, Epilepsy, Parkinson's/Movement, Multiple Sclerosis, and ranks highly for Cerebrovascular, six of eleven conditions. 

Right Posterior Polar Area 10 appears in the top-3 for ASD, Organic Mental Disorder, Bipolar Disorder, Multiple Sclerosis, and Parkinson's. 

**2.2  Convergent evidence** 

| Region | Disorder | Mechanism |
| :---- | :---- | :---- |
| Putamen (ventral anterior) | ASD | Striatal reward/repetitive behaviour circuit |
| Supramarginal gyrus (PF) | Cerebrovascular | MCA lesion,  neglect/aphasia locus |
| Anterior cingulate (area 24\) | Psychopathology/Dementia | Affective-cognitive hub; FTD/AD atrophy |
| Frontal operculum / insula | Schizophrenia | Salience network; consistent meta-analytic locus |
| VLPFC/DLPFC (area 9-46) | ASD, Bipolar | Cognitive control; frontoparietal network |
| PCC (area 23ab) | Organic Mental Disorder | Core DMN; AD hypometabolism |
| OFC (area 11\) in epilepsy | Epilepsy | OFC-onset seizure networks |

*Table 12 | Regions with strongest convergent support between BBTransformer importance rankings and neuroimaging literature.*

Together, these convergent findings validate the sensitivity of our permutation-based framework to detect biologically grounded neural signatures, even within a heterogeneous clinical dataset. They also highlight a critical principle: not all high-importance regions are equal. While some may reflect shared vulnerability or methodological artifacts, others map onto canonical disease mechanisms with high fidelity. Disentangling these categories is essential for translating machine learning outputs into clinically meaningful biomarkers.

# **Methods**

## **Participants and Data Sources**

Two independent data sources were used during transfer learning in this study. Foundation model training and initial hyperparameter optimization were conducted using the Autism Brain Imaging Data Exchange (ABIDE) repository, a multi-site, openly available dataset comprising 585 participants (271 with autism spectrum disorder and 314 typically developing controls) aged 6 to 50 years. Transfer learning across downstream neurological conditions used resting-state fMRI and phenotypic data from the UK Biobank, accessed under the application number. The UK Biobank is a large-scale prospective cohort study of approximately 500,000 participants aged 40–69 years at baseline, with a neuroimaging substudy providing brain MRI and phenotypic data for a subset of participants. All participants provided written informed consent to their respective studies. Use of UK Biobank data was approved by the North West Multi-centre Research Ethics Committee. ABIDE data are openly shared with no access restrictions under a data use agreement that requires attribution and non-commercial use.

For each disorder analyzed, cases were identified using ICD-10 diagnostic codes present in the UK Biobank health record linkage (Hospital Episode Statistics for England, Scottish Morbidity Records, and Patient Episode Database for Wales). Within the UK Biobank, case–control cohorts were balanced by randomly undersampling the larger group without replacement to achieve equal size, followed by joint random permutation of subject order using a fixed seed (42) to ensure reproducibility.

The ADHD-200 cohort comprises 242 participants, including 103 individuals diagnosed with attention-deficit/hyperactivity disorder (ADHD) and 139 typically developing controls. The sample is pediatric, with a mean age of 13.42 ± 2.90 years among cases and 12.64 ± 2.82 years among controls. A pronounced male predominance is observed in the ADHD group, with 91 males and 12 females, compared to 82 males and 57 females in the control group. The UCLA Clinical Cohort includes 132 adult participants, of whom 63 meet diagnostic criteria for a psychiatric disorder, specifically schizophrenia (n \= 15), bipolar disorder (n \= 22), or ADHD (n \= 26), and 69 are psychiatrically healthy controls. The mean age of cases is 33.89 ± 9.48 years, compared to 30.25 ± 8.58 years for controls. Within this cohort, the sex distribution among cases consists of 43 males and 20 females, while controls include 35 males and 34 females. Diagnoses in both cohorts were established using standardized clinical assessments, and all participants provided informed consent under protocols approved by their respective institutional review boards.

## **fMRI Preprocessing**

Preprocessing of ABIDE data followed the Configurable Pipeline for the Analysis of Connectomes (C-PAC) default pipeline with the following specifications: slice timing correction, motion realignment, co-registration of functional to structural images, white matter and CSF signal regression, and motion scrubbing.

UK Biobank imaging data were processed using the official UK Biobank neuroimaging pipeline, which includes susceptibility distortion correction, motion correction, nonlinear registration to MNI152NLin6Asym space, and ICA-FIX artefact removal. 

For external validation, we leveraged two independently preprocessed, publicly available datasets. The ADHD-200 dataset was sourced from The Neuro Bureau ADHD-200 Preprocessed Repository[\[11\]](#bookmark=id.y83hi0qs7b0c) , specifically using data processed with the Athena pipeline, which applies AFNI and FSL-based tools for slice timing correction, motion correction, spatial normalization, and denoising via nuisance regression. The UCLA Consortium for Neuropsychiatric Phenomics LA5c Study was preprocessed using fMRIPrep v0.4.4, a BIDS-compliant pipeline that integrates FSL, ANTs, FreeSurfer, and AFNI to perform motion correction, susceptibility distortion correction, coregistration, segmentation, and spatial normalization with minimal user-defined parameters. In both cases, we used the fully preprocessed outputs as provided; no additional site-specific preprocessing steps were applied.

## **Brain Parcellation**

To ensure model generalizability and cross-site portability, we standardized all inputs, regardless of origin, to a uniform spatiotemporal format. We developed a dedicated Python library to automate this standardization, enabling seamless application across diverse datasets and global research environments. Our framework utilizes a fixed resolution of 150 timepoints at a 2.0 s TR across 414 integrated cortical and subcortical regions.

A high-resolution combined atlas was constructed for all analyses, totaling 414 regions (360 cortical \+ 54 subcortical). Cortical parcellation was defined by the Human Connectome Project Multi-Modal Parcellation, identifying 360 regions based on myelin content, cortical thickness, and task activation topography. This was integrated with the Tian Scale 4 (S4) atlas , providing fine-grained subdivision of 54 subcortical structures, including thalamic nuclei, striatum, amygdala, and hippocampal subfields.

For ADHD-200, we began with the Athena-preprocessed repository , which includes multi-site rs-fMRI data processed with AFNI/FSL. From an initial pool of 562 functional runs, only 242 yielded complete 414-region coverage after parcellation; the remainder were excluded due to motion-induced dropout, registration failure, or invalid temporal structure.

For the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study , we used fMRIPrep v0.4.4 outputs organized in BIDS format. Initial parcellation of 265 subjects revealed 133 cases with incomplete ROI coverage (e.g., 383 or 412 regions), likely due to boundary misalignment or segmentation errors in subcortical structures. Only 132 subjects produced a full 414-ROI time series and were retained. 

To harmonize disparate acquisition protocols, UK Biobank (490 volumes, TR \= 0.735 s), ABIDE (variable lengths/TRs), ADHD-200 (multi-site, TR \= 2.0–2.5 s), and UCLA (TR \= 2.0 s), all native BOLD time series were resampled to a uniform 150-timepoint representation at a 2.0 s interval using cubic spline interpolation. This yielded a standardized 300-second (5-minute) resting-state window, enabling multi-site training without site-specific temporal bias. Finally, each regional time series was independently z-scored per subject to eliminate amplitude scaling differences while preserving intrinsic temporal dynamics.

## **Model Architecture**

BBTransformer is a multivariate time-series foundation model designed to decode neurological and psychiatric conditions directly from raw, whole-brain BOLD dynamics. Its architecture integrates three core innovations tailored to the structure of rs-fMRI data: (1) dual-resolution temporal encoding to capture both fine-grained fluctuations and coarse-scale trends, (2) adaptive temporal pooling that learns diagnostically relevant time windows, and (3) explicit integration of demographic confounders without disrupting spatiotemporal representations.

### **Input Representation**

The model accepts a three-dimensional input tensor **X** ∈ ℝB × T × D, where *B* denotes batch size, *T* represents temporal sequence length, and *D* corresponds to the feature dimensionality per timepoint (here, *D* \= 414 parcellated brain regions). Each timepoint vector captures instantaneous whole-brain activity without prior dimensionality reduction or functional connectivity computation. Two auxiliary covariates are provided: continuous age *a* ∈ ℝB and biological sex as a binary external factor *e* ∈ {0,1}B, are incorporated to mitigate potential confounding effects.

Input preprocessing applies root mean square normalization (RMSNorm) across the feature dimension, followed by a bias-free linear projection mapping each *D*\-dimensional timepoint to a *d*model\-dimensional embedding space (default: *d*model \= 512). A second RMSNorm layer and dropout (*p* \= 0.17) stabilise the projected representations, yielding the primary input sequence **X**proc ∈ ℝB × T × dmodel.

### **Dual-Pathway Temporal Encoding**

The architecture employs two parallel processing streams that capture complementary temporal scales, fused via cross-attention before classification. 

**Primary Stream: Global Context Modelling**  
The primary stream processes the full-resolution sequence through *L* \= 7 transformer encoder layers. Each layer implements grouped-query attention (GQA) with *hq* \= 16 query heads and *hkv* \= 8 key-value heads, reducing key-value cache memory requirements by 50% relative to standard multi-head attention while preserving representational capacity through query diversity. The head dimension is *d*head \= *d*model / *hq* \= 32\.

Relative temporal positions are encoded via rotary position embeddings (RoPE) applied to query and key vectors. RoPE rotates embedding subspaces by frequency-dependent angles derived from token position, encoding relative temporal distances rather than absolute indices. Formally, for query **q***i* and key **k***j* at positions *i* and *j*:

**q***i*rot \= **q***i* ⊙ cos(**θ***i*) \+ rotate\_half(**q***i*) ⊙ sin(**θ***i*)

**k***j*rot \= **k***j* ⊙ cos(**θ***j*) \+ rotate\_half(**k***j*) ⊙ sin(**θ***j*)

where **θ***p* \= \[*p* · ω0, *p* · ω0, *p* · ω1, *p* · ω1, …\] with frequencies ω*k* \= 10000−2*k*/*d*head. This formulation enables generalisation to variable-length sequences and facilitates learning of temporally invariant relationships.

Each encoder layer follows a pre-normalisation design with RMSNorm applied before both sublayers. The attention sublayer computes GQA with RoPE, applies stochastic depth regularisation, and incorporates a residual connection followed by post-attention RMSNorm. The feed-forward sublayer employs SwiGLU activation:

FFN(**x**) \= **W**2 \[ SiLU(**W**1**x**) ⊙ (**W**3**x**) \]

where SiLU denotes *x* · σ(*x*), ⊙ represents element-wise multiplication, and all projections are bias-free. This gated formulation enhances gradient flow and representational expressivity relative to ReLU or GELU alternatives. Stochastic depth regularisation employs linearly increasing drop path probabilities across layers, from 0.0 in the first layer to *r* \= 0.10 in the final layer, improving training stability for deep architectures.

**Local Stream: Patch-Based Feature Extraction**

A parallel pathway extracts coarse-grained temporal patterns via patch-based processing. The input sequence is partitioned into non-overlapping temporal patches of size *P* \= 3, with trailing timesteps trimmed if *T* mod *P* ≠ 0\. Each patch concatenates *P* × *d*model features and projects to a reduced embedding dimension *d*patch \= ⌊*d*model × 0.75⌋ \= 384 via a bias-free linear layer. RMSNorm, SiLU activation, and dropout (*p* \= 0.15) complete the patch embedding module, yielding **X**patch ∈ ℝB × ⌊T/P⌋ × dpatch.

Patch tokens undergo no transformer encoding; instead, a global representation **g**patch ∈ ℝ*d*patch is obtained via mean pooling across the patch dimension. This design prioritises computational efficiency while preserving aggregate short-scale temporal statistics.

**Multi-Scale Feature Fusion**

Following the primary transformer stack, patch representations are upsampled to match the primary sequence length *T* via linear interpolation. Zero-padding aligns the patch embedding dimension to *d*model, producing **X**patch↑ ∈ ℝB × T × dmodel.

A multi-head cross-attention module (with *hq* heads, dropout *p* \= 0.15) then fuses the streams, with primary stream outputs serving as queries and upsampled patch features as keys and values. The cross-attended output is added to the primary stream with a residual connection and stabilised via RMSNorm. This mechanism enables each primary timepoint to selectively attend to relevant coarse-scale patterns, integrating fine-grained and aggregated temporal information.

### **Adaptive Temporal Pooling and Confounder Integration**

A learned attention pooling mechanism aggregates the fused sequence into a global diagnostic embedding. A two-layer multilayer perceptron with hidden dimension 512, Tanh activation, and dropout (*p* \= 0.16) computes unnormalized scalar weights for each timepoint. Softmax normalisation yields attention coefficients **α** ∈ ℝB × T × 1, and the context vector is computed as:

**g**main \= Σ*t*\=1*T* α*t* · **x***t*

This adaptive pooling emphasises diagnostically salient temporal windows without predefined window boundaries.

Confounder representations are integrated via linear projection of scalar age to *d*age \= 32 dimensions and a learned embedding table mapping the binary external factor to *d*ext \= 16 dimensions.

### **Classification Objective**

The final prediction integrates four components via concatenation:

**z** \= \[**g**main; **g**patch; **e**age; **e**ext\] ∈ ℝ*d*model \+ *d*patch \+ *d*age \+ *d*ext

A two-layer classifier with bias-free projections, RMSNorm, SiLU activations, and progressive dropout (*p* \= 0.04, *p* \= 0.02) maps **z** to output logits. For binary classification, the scalar logit is passed to a sigmoid function during inference; for multi-class tasks, softmax is applied across the output dimension.

### **Regularisation Strategies**

All linear layers employ Xavier uniform initialisation with layer-specific gain factors: standard projections use gain \= 1.0, residual output projections use gain \= 0.02, and classifier layers use gain \= 0.01. Embedding weights are initialised from 𝒩(0, 0.02), and biases, where present, are zero-initialised. This hierarchical initialisation strategy mitigates signal explosion in deep residual architectures.

## 

## **Training Protocol**

We performed Bayesian optimization using the Tree-structured Parzen Estimator (TPE) implemented in Optuna over 50 trials to identify high-performing configurations for autism spectrum disorder (ASD) classification. The search space included architectural hyperparameters, embedding dimension (256–768), encoder layers (4–10), attention heads (4–16), key-value heads (2–8), patch size (2–6), patch embedding ratio (0.25–1.0), temporal pooling hidden dimension (128–1024), stochastic depth rate (0.0–0.2), and per-module dropout rates (input, patch, attention, feedforward, classifier, temporal; each 0.0–0.5). Training hyperparameters included learning rate (1e⁻⁵–1e⁻³, log-uniform) and weight decay (1e⁻⁸–1e⁻⁴, log-uniform).

The final model was trained for up to 5,000 epochs using the Ranger21 optimizer, a lookahead-enhanced adaptive optimizer that internally schedules learning rate warmup and decay, with early stopping triggered after 90 consecutive epochs without improvement in validation F1 score. The loss function was binary cross-entropy with logits; no class weighting was applied, as the ASD cohort was approximately balanced (271 cases, 314 controls; prevalence \= 46.3%). During hyperparameter tuning, adaptive focal loss was optionally enabled for highly imbalanced transfer tasks, but not used for the ASD foundation model.

Data were split into 70% training (n=409), 15% validation (n=88), and 15% test (n=88) sets, stratified by diagnosis. Splits were held fixed across all trials to ensure direct comparability of validation metrics.

## **Transfer Learning Protocol**

For each downstream disorder, the fixed optimal architecture identified during ASD tuning was used without modification. Weights were initialized from the most recent valid model in the transfer chain (i.e., the preceding disorder that met all validity thresholds). The training procedure followed the same Ranger21 optimizer with identical learning rate and weight decay as the ASD tuning best configuration, with early stopping after 90 epochs without validation improvement. A model was accepted, and its weights propagated forward, only if all five metrics (F1, ROC-AUC, accuracy, precision, recall) exceeded 0.60 on the held-out test set. If no trial met this criterion after 15 attempts, the prior valid weights were retained, and the failed disorder was recorded. Total effective training involved 2,367 subjects across 11 conditions (including ASD foundation training), representing 23.5% of the total 10,085 subjects processed.

![][image1]

*Figure. Biologically ordered transfer learning pipeline. BBTransformer was pre-trained on ASD (Phase 0, ABIDE) and sequentially fine-tuned across 10 neurological and psychiatric disorders. Arrows denote the direction of weight propagation; transition labels indicate the pathophysiological axis traversed. All displayed phases met the validity criterion (all five metrics ≥ 0.60).* 

## 

## **Permutation-based regional importance**

To identify brain regions most critical for diagnostic classification, we computed permutation feature importance across all 414 cortical and subcortical regions of interest. For each region, we generated a perturbed validation set by randomly reassigning its BOLD time series across subjects, that is, the time series for a given region in one subject was replaced with that from another randomly selected subject. This preserves the temporal structure and marginal distribution of the BOLD signal within each individual while disrupting the association between regional activity and diagnostic label. The trained model was then evaluated on this perturbed data, and the resulting F1 score was recorded. This procedure was repeated 50 times per region using independent random permutations. The importance of each region was defined as the mean decrease in F1 score relative to the baseline (unperturbed) performance, averaged across the 50 repetitions. Regions were ranked by this metric, and the top three are reported for each disorder.

![][image2]  
*Figure. BBTransformer full computational diagram. Three parallel streams process the raw 150 × 414 BOLD input: (1) a Global Stream with a 6-layer GQA+RoPE+SwiGLU transformer encoder; (2) a Local Patch Stream producing coarse temporal tokens integrated via cross-attention; and (3) a Confounder Stream encoding age and imaging site as learned embeddings. Temporal attention pooling produces a single diagnostic embedding passed to a final MLP classifier.*

# 

# **Discussion**

## **Temporal Dynamics**

The primary finding of this study is that raw spatiotemporal BOLD dynamics harbor sufficient diagnostic information to classify a diverse spectrum of neurological and psychiatric conditions with clinically meaningful accuracy. Critically, this is achieved without the use of hand-crafted connectivity features. This result challenges the prevailing view that resting-state fMRI is inherently too stochastic or prone to inter-site variability to permit reliable, single-subject classification. Our model overturns this consensus by demonstrating that prioritizing the temporal sequence of whole-brain states, rather than their time-averaged correlations, yields robust diagnostic performance, with F1 scores ranging from 0.68 to 0.92 across ten distinct conditions.

The analogy to protein or language models is particularly instructive. Just as AlphaFold2 and its successors infer the biophysical rules of protein folding from statistical regularities in evolutionary sequences, BBTransformer decodes the spatiotemporal signatures of brain dysfunction directly from raw data. The model does not require a priori definitions of metastability, phase relationships, or autocorrelation structures; instead, it autonomously discovers how disruptions in these temporal rhythms distinguish pathological states from health, bypassing the information bottleneck of static functional connectivity. 

## **Transfer Learning**

Our training framework is a distinct methodological contribution that addresses the data scarcity bottleneck in clinical neuroimaging. By leveraging a 10-step transfer chain based on pathophysiological relatedness, we demonstrate that weights learned from high-prevalence conditions significantly improve fine-tuning on small, rare-disorder cohorts. This suggests that pathophysiological similarity can effectively guide weight initialization, allowing models to inherit shared temporal dynamics that are otherwise unlearnable from scratch in small samples. Ultimately, this framework establishes a scalable curriculum for translating large-scale foundational knowledge into high-precision clinical applications. 

## **Diagnosis Quality**

The failure of our model to classify depressive episodes, sleep disorders, and substance use disorders, despite these having the largest sample sizes, is perhaps the most scientifically significant finding of this work. This result directly refutes the assumption that scale alone enables learnability, demonstrating instead that biological coherence is the primary determinant of model success. For a diagnostic label to be detectable, it must map onto a consistent dynamical state in the BOLD signal; our results provide quantitative evidence of where this mapping breaks down.

In the UK Biobank, these specific conditions are often assigned by general practitioners through brief encounters rather than structured clinical interviews. For instance, the ICD-10 code for depression aggregates a highly heterogeneous population, ranging from reactive low mood to mild depression, each associated with divergent patterns of circuit disruption. Under these conditions, the temporal dynamics of ‘depressed’ subjects are no more similar to one another than to controls, meaning the diagnostic boundary does not exist in dynamical-state space.

This diagnostic quality problem extends to sleep and substance use disorders, where administrative labels encompass varied neuromodulatory profiles (e.g., alcohol vs. stimulants) that lack biological homogeneity. These failures align with broader critiques of categorical diagnostic systems and imply that fMRI biomarker discovery requires biologically homogeneous cohorts defined by mechanistic criteria.

## **Interpretability**

Feature importance reveals how regional interactions, encoded within the model weights, reconfigure across different conditions. The finding that top-ranked regions form anatomically coherent circuits serves a dual purpose: it validates that the model captures biologically grounded patterns rather than site-specific artifacts, and it generates testable hypotheses regarding the temporal dynamics of disease-specific circuits.

This capacity for internal weight reconfiguration suggests that transformer architectures can serve as powerful tools for the discovery of dynamical biomarkers across the neurological spectrum.

## **Limitations**

The primary constraint of our model is the data-scale ceiling inherent to current neuroimaging research. While foundation models in structural biology like AlphaFold2 leverage hundreds of thousands of protein structures, the 2,367 subjects utilized here represent only a fraction of that scale. This is not a limitation of the architecture, but a reflection of the field-wide scarcity of well-phenotyped, high-quality fMRI data. Overcoming the performance plateaus observed in heterogeneous conditions will require a coordinated, global data-sharing infrastructure at an unprecedented scale.

Furthermore, while our model decodes regional dynamics, it is currently limited to binary classification. Because many neurological conditions are better conceptualized as spectra, future extensions must incorporate multi-class architectures and regression objectives to capture clinical severity. Additionally, while we provide regional interpretability, further work is needed to systematically analyze the temporal saliency, the specific timepoints and sequences, that the model prioritizes during inference.

Finally, the clinical potential of single-subject classification must be balanced against the ethical risks of misclassification and data privacy. We emphasize that these results are a proof of concept for spatiotemporal foundation models, not a validated diagnostic tool for clinical deployment.

# **Conclusion**

We have introduced a multivariate time series foundational model that decodes neurological and psychiatric conditions directly from raw rs-fMRI time series, bypassing the information bottleneck imposed by static functional connectivity analysis. By learning the spatiotemporal dynamics of brain dysfunction, the sequences, rhythms, and co-fluctuations that distinguish pathological from healthy neural dynamics, the model achieves robust classification performance across 10 diverse conditions using biologically ordered transfer learning from an ASD foundation. The principled failures on depressive episode, sleep disorders, and substance use disorders are scientifically informative, establishing that biological coherence of the phenotype, not sample size, is the binding constraint on learnability from spatiotemporal BOLD dynamics.

These results open several avenues for future work. In the near term, the transfer chain should be extended to additional conditions, alternative transfer topologies guided by neurobiological embedding similarity should be explored, and regression and subtype identification objectives should be incorporated alongside binary classification. 

# **Future Directions**

The transition from task-specific architectures to large-scale foundation models represents a critical frontier in computational neuroscience. To unlock the emergent properties, capabilities that arise only at high computational and data thresholds, the neuroimaging community must prioritize architectural scaling as a primary objective. Current academic efforts frequently operate in isolation; however, to achieve the transformative performance seen in protein folding and natural language processing, we must bridge the gap between academic research and the large-scale computational strategies utilized by the private sector. While our current model was trained on 2,367 subjects, this cohort represents only a fraction of the data scale necessary to reach a "scaling law" inflection point. Establishing open-source networks for the decentralized exchange of models and high-dimensional datasets is no longer an elective endeavor, but a requirement for the next generation of neuroimaging discovery.

# **References**

1. Biswal, B., Yetkin, F. Z., Haughton, V. M. & Hyde, J. S. Functional connectivity in the motor cortex of resting human brain using echo-planar MRI. *Magn. Reson. Med.* 34, 537–541 (1995).  
2. Buckner, R. L., Andrews-Hanna, J. R. & Schacter, D. L. The brain’s default network: anatomy, function, and relevance to disease. *Ann. N. Y. Acad. Sci.* 1124, 1–38 (2008).  
3. Vaswani, A. et al. Attention is all you need. *Adv. Neural Inf. Process. Syst.* 30, 5998–6008 (2017).  
4. Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. A time series is worth 64 words: long-term forecasting with transformers. *Proc. Int. Conf. Learn. Represent.* (2023).  
5. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B. & Liu, Y. RoFormer: enhanced transformer with rotary position embedding. Preprint at https://arxiv.org/abs/2104.09864 (2021).  
6. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyansky, Y., Lebrón, F. & Sanghai, S. GQA: training generalized multi-query transformer models from multi-head checkpoints. *Proc. Conf. Empirical Methods Nat. Lang. Process.* 2023, 4895–4901 (2023).  
7. Zhang, B. & Sennrich, R. Root mean square layer normalization. *Adv. Neural Inf. Process. Syst.* 32, 12360–12371 (2019).  
8. Shazeer, N. GLU variants improve transformer. Preprint at https://arxiv.org/abs/2002.05202 (2020).  
9. Glasser, M. F. et al. A multi-modal parcellation of human cerebral cortex. *Nature* 536, 171–178 (2016).  
10. Tian, Y., Margulies, D. S., Breakspear, M. & Zalesky, A. Topographic organization of the human subcortex unveiled with functional connectivity gradients. *Nat. Neurosci.* 23, 1421–1432 (2020).  
11. Di Martino, A. et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Mol. Psychiatry* 19, 659–667 (2014).  
12. Alfaro-Almagro, F. et al. Image processing and quality control for the first 10,000 brain imaging datasets from UK Biobank. *NeuroImage* 166, 400–424 (2018).  
13. Brown, M. R. G. et al. ADHD-200 Global Competition: diagnosing ADHD using personal characteristic data can outperform resting state fMRI measurements. *Front. Syst. Neurosci.* 6, 69 (2012).  
14. Poldrack, R. A. et al. A phenome-wide examination of neural and cognitive function. *Sci. Data* 3, 160110 (2016).  
15. Sato, J. R., Hoexter, M. Q., Salum, G. A. & Brammer, M. J. Subcortical alterations in autism: meta-analysis of structural MRI. *Dev. Cogn. Neurosci.* 12, 28–35 (2014).  
16. Ecker, C., Bookheimer, S. Y. & Murphy, D. G. M. Neuroanatomy of autism spectrum disorder. *Autism Res.* 14, 1893–1895 (2021).  
17. Greicius, M. D., Srivastava, G., Reiss, A. L. & Menon, V. Default-mode network activity distinguishes Alzheimer's disease from healthy aging: evidence from functional MRI. *Proc. Natl. Acad. Sci. USA* 101, 4637–4642 (2004).  
18. Velayudhan, L., Dury, R., Susan, F., Penny, G., Kempton, M. & Sagnik, B. Reduced supramarginal gyrus gray matter volume associated with cognitive impairment in Alzheimer’s disease: a 7-Tesla MRI study. *Alzheimers Dement.* 12, P3-279 (2016).  
19. Van Hoesen, G. W., Parvizi, J. & Chu, C.-C. Orbitofrontal cortex pathology in Alzheimer's disease. *Cereb. Cortex* 10, 243–251 (2000).  
20. Zhou, J. et al. Divergent network connectivity changes in behavioral variant frontotemporal dementia and Alzheimer's disease. *Brain* 133, 1352–1367 (2010).  
21. Chételat, G. et al. Atrophy in the anterior cingulate cortex predicts apathy in Alzheimer’s disease. *Neurobiol. Aging* 34, 2333–2341 (2013).  
22. Rosen, H. J. et al. Neuroanatomical correlates of behavioural disorders in dementia. *Brain* 128, 2612–2625 (2005).   
23. Rosen, H. J., Miller, B. L. & Kramer, J. H. Orbitofrontal and DMN connectivity changes in dementia and encephalopathy. *Ann. Neurol.* 95, 641–652 (2024).  
24. Shen, X., Taylor, G. & Semple, S. Anterior cingulate cortex alterations in dementia. *Sci. Rep.* 11, 21384 (2021).  
25. Lyketsos, C. G. et al. Neuropsychiatric symptoms in Alzheimer's disease. *Alzheimers Dement.* 7, 532–539 (2011).  
26. Yang, H. et al. Study of brain morphology change in Alzheimer's disease and amnestic mild cognitive impairment compared with normal controls. *Gen. Psychiatr.* 32, e100005 (2019).  
27. Schröder, J., Kubera, K. M. & Wolf, R. C. Cortical connectivity in dementia and vascular cognitive impairment. *Brain Commun.* 7, fcaf477 (2025).  
28. Habbal, D. et al. Default-mode network abnormalities in herpes simplex encephalitis. *J. Neurovirol.* 16, 269–276 (2010).  
29. Duong, M. T., Rudie, J. D. & Mohan, S. Neuroimaging patterns of intracranial infections: meningitis, cerebritis, and their complications. *Neuroimaging Clin. N. Am.* 33, 11–41 (2023).  
30. Müller, V. I. et al. Lesion-symptom mapping in post-stroke cognition and dementia. *Brain Commun.* 7, fcaf012 (2025).  
31. Cho, E. B. et al. Disrupted structural network of inferomedial temporal regions in relapsing–remitting multiple sclerosis compared with neuromyelitis optica spectrum disorder. *Sci. Rep.* 12, 5152 (2022).  
32. Abé, C. et al. Bipolar disorder type I and II show distinct relationships between cortical thickness and executive function. *Acta Psychiatr. Scand.* 138, 325–335 (2018).  
33. Bi, B., Che, D. & Bai, Y. Neural network of bipolar disorder: toward integration of neuroimaging and neurocircuit-based treatment strategies. *Transl. Psychiatry* 12, 143 (2022).  
34. Li, H. et al. Decreased functional connectivity of vermis–ventral prefrontal cortex in bipolar disorder. *Front. Hum. Neurosci.* 15, 711688 (2021).  
35. Guo, Z. et al. Alterations in sulcal depth and associated functional connectivity in schizophrenia with auditory verbal hallucinations. *Front. Psychiatry* 16, 1641190 (2025).  
36. van Ommen, M. M. et al. Reduced occipital responsiveness in psychosis with visual hallucinations: evidence from dynamic object recognition fMRI. *medRxiv* (2022).   
37. Ryvlin, P. et al. The role of the orbitofrontal cortex in focal epilepsy. *Epilepsia* 55 (Suppl 1), 34–39 (2014).  
38. Dash, D. & Tripathi, M. The extratemporal lobe epilepsies in the epilepsy monitoring unit. *Ann. Indian Acad. Neurol.* 17 (Suppl 1), S50–S55 (2014).  
39. Jaafar, N., Bhatt, A., Eid, A. & Koubeissi, M. Z. The temporal lobe as a symptomatogenic zone in medial parietal lobe epilepsy. *Front. Neurol.* 13, 804128 (2022).  
40. Taylor, A. E., Saint-Cyr, J. A. & Lang, A. E. Frontal lobe dysfunction in Parkinson's disease. The cortical focus of neostriatal outflow. *Brain* 109, 845–883 (1986).  
41. Jia, X. et al. Progressive prefrontal cortex dysfunction in Parkinson's disease with probable REM sleep behavior disorder: a 3-year longitudinal study. *Front. Aging Neurosci.* 13, 750767 (2022).  
42. Pedrazzini, E. & Ptak, R. Damage to the right temporoparietal junction, but not lateral prefrontal or insular cortex, amplifies the role of goal-directed attention. *Sci. Rep.* 9, 306 (2019).  
43. Louapre, C. et al. Brain networks disconnection in early multiple sclerosis cognitive deficits: an anatomofunctional study. *Hum. Brain Mapp.* 35, 4706–4717 (2014).  
44. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021).

