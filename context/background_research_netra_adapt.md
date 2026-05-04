# Comprehensive Background Research for Netra-Adapt
## Source-Free Domain Adaptation for Cross-Ethnic Glaucoma Screening

---

## 1. Cross-Ethnic Fairness & Domain Shift in Ophthalmic AI

### 1.1 The Core Problem: Domain Shift Exacerbates Fairness Disparities

The transfer of medical AI models across demographic groups is now recognized as a critical fairness challenge. Recent work has established that domain shift -- whether from imaging modalities or population differences -- systematically worsens algorithmic fairness:

**FairDomain (Tian et al., ECCV 2024)** [^5^][^8^][^17^] presents the first systematic study of algorithmic fairness under domain shifts for both medical segmentation and classification. Their key findings directly support your work:
- Group performance disparities **significantly worsen** when transferring from source to target domains
- For glaucoma classification specifically, they found AUC disparities of **0.085 lower for Black patients vs. White patients** when using fundus images
- They introduce the Fair Identity Attention (FIA) module and curate paired cross-modal datasets (en face fundus and SLO) for the same patient cohort to isolate domain-shift effects from demographic distribution effects
- Their Equity-Scaled Performance (ESP) metrics provide a framework for evaluating both accuracy and fairness -- you should consider citing this for your evaluation

**Key citation for your paper**: "Domain shift significantly exacerbates group performance disparities between source and target domains, indicating the critical necessity for fairness-oriented algorithms" [^8^]

### 1.2 Retinal Pigmentation as a Biological Factor

**Rajesh et al. (Nature Communications, 2025)** [^14^] introduces the **Retinal Pigment Score (RPS)** -- a continuous metric that quantifies fundus pigmentation from color fundus photographs. Critical findings:
- GWAS identified **20 loci** associated with skin, iris, and hair pigmentation
- While RPS correlates with ethnicity, there is **substantial overlap** between ethnic groups -- demonstrating that "ethnicity is not biology"
- RPS provides a way to quantify training dataset diversity and explain model performance variation
- This directly supports your argument about melanin-induced domain shift between Western and Indian retinas

**Key citation for your paper**: "RPS decouples traditional demographic variables from clinical imaging characteristics and may serve as a useful metric to quantify dataset diversity and explain model performance" [^14^]

### 1.3 Documented Bias in Retinal Diagnostics

**Burlina et al. (Translational Vision Science & Technology, 2021)** [^54^][^60^] demonstrated concrete AI bias in retinal diagnostics:
- Baseline DR diagnostic system: **73.0% accuracy on lighter-skin vs. 60.5% on darker-skin patients** (delta = 12.5%, P = 0.008)
- Using synthetic data augmentation for debiasing achieved parity: 72.0% vs. 71.5% (delta = 0.5%)
- This was one of the first studies to explicitly link fundus pigmentation differences to diagnostic performance gaps

**Key citation for your paper**: This establishes the precedent that pigmentation differences cause measurable diagnostic disparities.

### 1.4 Domain Shift-Induced Unfairness in Screening

**Wang et al. (IOVS, 2025)** [^44^] -- the paper cited in your slides -- specifically examines domain shift-induced unfairness in AI for diabetic retinopathy screening. Their ARVO abstract identifies cross-population performance degradation as a key challenge in deploying screening systems globally.

---

## 2. Source-Free Domain Adaptation (SFDA) Landscape

### 2.1 Foundational Methods

**SHOT (Liang et al., ICML 2020)** [^11^] remains the foundational SFDA baseline:
- Freezes the classifier (hypothesis) and learns target-specific features using **information maximization** and **self-supervised pseudo-labeling**
- Uses k-means clustering with mutual information maximization for self-training
- Your Netra-Adapt outperforms SHOT (0.884 vs. 0.765 AUROC) -- a **15.5% relative improvement**

**Theoretical understanding of label noise in SFDA** [^10^]: Recent work frames SFDA as learning with label noise (LLN) and identifies the "early-time training phenomenon" (ETP) -- early training dynamics can be leveraged to improve pseudo-label quality. This connects to your observation that naive entropy minimization can collapse.

### 2.2 Recent Advances in Medical SFDA

**AIF-SFDA (Li et al., AAAI 2025)** [^1^] proposes an Autonomous Information Filter that uses frequency-based learnable filters to decouple domain-variant from domain-invariant information:
- Uses Information Bottleneck (IB) and self-supervision constraints
- Outperforms existing SFDA on retinal vessel segmentation (DSC 69.14 vs. 67.03 for prior SOTA)
- Demonstrates that frequency-domain decomposition is effective for fundus image adaptation

**ProSFDA (Pattern Recognition, 2025)** uses prompt learning for medical image segmentation.

**VP-SFDA (Health Data Science, 2025)** applies visual prompt-based SFDA for cross-modal medical segmentation.

**CLMS (2024)** [^3^] proposes continual learning for SFDA:
- Addresses catastrophic forgetting during adaptation via multi-scale reconstruction and style alignment
- Shows that retraining can lead to catastrophic forgetting of source anatomical knowledge
- Uses replay-based approaches to preserve morphological features

**Key citation**: "The risk of forgetting important morphological features during adaptation poses a risk to the performance of target applications" [^3^]

### 2.3 Comprehensive SFDA Resources

A curated GitHub repository [^2^] tracks SFDA papers, code, and benchmarks, including:
- Surveys: "A Comprehensive Survey on Source-Free Domain Adaptation" (TPAMI 2024)
- Medical imaging-specific methods
- Theoretical foundations: "Understanding and Improving SFDA from a Theoretical Perspective" (CVPR 2024)

---

## 3. Test-Time Adaptation (TTA) for Medical Imaging

### 3.1 Foundational TTA: TENT

**TENT (Wang et al., ICLR 2021)** [^19^][^20^][^22^][^23^] established test entropy minimization as the core TTA paradigm:
- Optimizes normalization statistics and channel-wise affine transformations online
- Uses entropy of predictions as self-supervision signal
- Handles source-free domain adaptation on digit recognition, semantic segmentation, and VisDA-C
- Reduces generalization error on corrupted ImageNet-C

**Critical insight for your work**: TENT showed that entropy minimization can reduce error, but also noted cases where "tent decreases entropy but increases loss" [^19^] -- directly supporting your catastrophic collapse finding.

### 3.2 Continual and Stable TTA

The field has rapidly evolved beyond TENT. A comprehensive survey [^16^] identifies three families:
1. **Optimization-based**: Entropy minimization (TENT, EATA, SAR, RMT, SANTA, REM), pseudo-labeling, parameter restoration
2. **Parameter-Efficient**: Normalization layer updates, adaptive updates
3. **Architecture-based**: Teacher-student frameworks, adapters, prompting, masked modeling

**Key methods relevant to your work**:
- **EATA (Niu et al., 2022)**: Addresses stability concerns in TENT
- **SAR (Niu et al., 2023)**: Stable test-time adaptation for dynamic wild world
- **RMT (Dobler et al., 2023)**: Robust mean teacher approach
- **SANTA (Chakrabarty et al., 2024)**: Addresses error accumulation
- **CPT4 (2025)** [^41^]: Continual Prompted Transformer for test-time training -- uses prompt pools to mitigate catastrophic forgetting in continual TTA

### 3.3 TTA Benchmarks for Medical Imaging

**MedSeg-TTA (2025)** [^24^] presents a comprehensive benchmark examining 20 adaptation methods across 7 imaging modalities (MRI, CT, ultrasound, pathology, dermoscopy, OCT, chest X-ray). This provides an important reference for positioning your ophthalmology-specific adaptation method.

---

## 4. Self-Supervised Vision Transformers for Medical Imaging

### 4.1 DINOv2/DINOv3 as Medical Backbones

**DINO (Caron et al., ICCV 2021)** and successors learn through self-distillation, producing Vision Transformers that capture **shape-based features rather than texture** -- a critical property for cross-ethnic robustness:

Key properties relevant to your work:
- **Shape bias**: DINO features are inherently more robust to pigmentation variations than texture-based CNN features [^6^][^7^][^12^]
- **Zero-shot transfer**: Strong zero-shot performance without domain-specific pre-training
- **Feature quality**: DINOv2 features consistently yield SOTA or competitive results for disease classification in X-ray, CT, MRI [^7^]

**"Does DINOv3 Set a New Medical Vision Standard?" (2025)** [^38^]:
- Benchmarks DINOv3 across 2D/3D classification and segmentation on multiple medical imaging modalities
- DINOv3 **outperforms medical-specific foundation models** like BiomedCLIP and CT-Net on several tasks despite being trained only on natural images
- Key limitation identified: does not consistently obey scaling laws in medical domain
- Establishes DINOv3 as "a strong baseline whose powerful visual features can serve as a robust prior for multiple complex medical tasks"

**DINOv3 backbone characteristics** [^39^]:
- Employs advanced patch embedding, rotary positional encoding, self-distillation pretraining
- Almost always kept frozen with lightweight domain-adaptive modules
- Freezing preserves learned representations while minimizing overfitting

### 4.2 RETFound: Retinal Foundation Models

**RETFound-Green (Nature Communications, 2025)** [^50^]:
- Uses **DINOv2 weights** as starting point for retinal foundation model training
- Proposes novel "Token Reconstruction" strategy (vs. MAE used by competing approaches)
- Trains on retinal images with self-supervised learning
- Processes images at 392x392 resolution (vs. 224x224 for competitors)
- Demonstrates that starting from DINOv2 and adapting to retinal images is highly effective

### 4.3 Self-Supervised Learning for Glaucoma Specifically

**DINO-EYE / SSL for Glaucoma Detection** [^49^][^52^]:
- Self-supervised contrastive learning on unlabeled fundus images achieves **AUROC of 0.852** with only 80% of labeled data for fine-tuning
- Grad-CAM visualizations show SSL models concentrate on clinically relevant regions (optic disc and cup)
- SSL-trained networks learn region-based features that align with clinical diagnostic standards

**PaRCL (Yi et al., IEEE BIBM 2024)** [^49^]: Pathology-aware Representation Contrastive Learning specifically for glaucoma classification on fundus images.

---

## 5. Information Maximization, Entropy & Diversity

### 5.1 The Entropy Minimization vs. Diversity Maximization Trade-off

**Wu et al. (2020)** [^46^][^47^] provide crucial theoretical foundations directly relevant to your MixEnt-Adapt:
- **Entropy minimization alone (EMO) collapses to trivial solutions** -- producing single-class predictions
- They prove Lemma 1: If entropy reaches zero, the inferred category distribution has entropy <= the true distribution, meaning collapse is inevitable
- Propose **Minimal-Entropy Diversity Maximization (MEDM)** to balance the two objectives
- Demonstrate SVHN->MNIST where EMO predicts only digit "1" consistently

**Key theoretical support for your work**: This provides theoretical grounding for why your Information Maximization loss (combining entropy minimization WITH diversity maximization) is necessary to prevent collapse.

### 5.2 SHOT's Information Maximization Formulation

SHOT's IM loss [^11^]: L_IM = L_ent - lambda * L_div
- L_ent: Entropy minimization (confidence)
- L_div: Diversity maximization (marginal class distribution entropy)
- This formulation is exactly what you build upon in MixEnt-Adapt

### 5.3 MEDM Theoretical Results

Wu et al. show that the weighting between entropy minimization and diversity maximization is **essential** -- not just a minor hyperparameter. The balance determines whether the method achieves:
- Collapse (too much entropy minimization)
- Uniform but uninformative predictions (too much diversity)
- Optimal adaptation (balanced)

This directly supports your ablation showing each component contributes incrementally to performance.

---

## 6. AdaIN and Feature Calibration Methods

### 6.1 Adaptive Instance Normalization

**AdaIN** [^18^] aligns channel-wise feature statistics (mean and variance) between inputs:
- Originally proposed for arbitrary style transfer
- Core operation: z_adapted = sigma_c * (z - mu_u)/sigma_u + mu_c
- Preserves content while normalizing style (domain-specific statistics)

### 6.2 AdaIN for Domain Adaptation

Multiple recent works apply AdaIN for medical domain adaptation [^28^]:
- AdaIN adaptation achieves **86% Dice** vs. 84% for IN adaptation and 75% for random embedding in multi-task spine imaging
- Domain modulation via AdaIN effectively adapts to new domains at test time
- "Folding domain adaptation, self-supervised learning, and knowledge distillation into unified architectures by AdaIN-based code modulation" [^18^]

**Key insight for your paper**: AdaIN's property of preserving content (structural retinal features) while normalizing style (domain-specific statistics like pigmentation and illumination) makes it particularly suited for cross-ethnic fundus adaptation.

---

## 7. Catastrophic Collapse & Failure Modes in TTA

### 7.1 Model Collapse in Medical AI

Recent work on **model collapse** [^27^] shows how adaptation can destroy clinical capability:
- Models can achieve low training loss while destroying decision boundaries
- "Models became overconfident on degraded content they could no longer clinically interpret"
- False confidence: 196-fold confidence gap between synthetic and real clinical outputs
- Vocabulary collapsed from 6,025 unique words to 62 (99% reduction)

This directly mirrors your finding that the model collapsed to single-class prediction with specificity dropping to 1.4%.

### 7.2 Catastrophic Forgetting in Continual Learning

**CPT4 (2025)** [^41^] addresses catastrophic forgetting in TTA via prompt pools:
- "Real-world scenarios frequently encounter continual shifts in target data domains during testing, leading to complexities in ongoing adaptation and error propagation"
- Uses shared prompts and batch normalization module to retain prior task information

**Adaptive Additive Parameter Updates for ViTs (2025)** [^36^]:
- Selectively updates only self-attention layers within frozen ViT backbone
- Demonstrates that catastrophic forgetting in ViTs can be mitigated by targeted adaptation

### 7.3 The Specific Danger in Medical TTA

Your finding that "unsupervised TTA is currently unsafe for clinical deployment" aligns with broader concerns:
- Models can exploit confounding variables (lighting, metadata) rather than pathology [^19^]
- Entropy minimization alone drives toward trivial solutions [^46^]
- Without diversity constraints, models collapse to single-class predictors

---

## 8. Calibration and Uncertainty in Medical AI

### 8.1 Expected Calibration Error (ECE) in Your Context

Your Netra-Adapt achieves ECE of 0.056 vs. 0.089 for SHOT and 0.142 for Frozen DINO. This is significant because:

**Foundation Model Robustness to Technical Factors (2026)** [^56^]:
- RAD-DINO achieves excellent calibration stability across view types (ECE difference < 0.01)
- Other models show "marked differential calibration" with ECE varying from 0.218 to 0.428 depending on technical factors
- Calibration varies substantially across architectures and acquisition parameters
- This confirms that good calibration under domain shift is non-trivial and valuable

### 8.2 Uncertainty Quantification Methods

Comprehensive reviews [^34^][^35^][^37^] establish:
- **Predictive entropy** is the most widely used uncertainty measure for medical image classification
- Entropy maximizes at p=0.5 (maximum indecision) and minimizes at p=0 or p=1
- Most modern CNNs are **not well calibrated** and produce overconfident predictions
- Temperature scaling is an effective post-hoc calibration method

Your use of temperature scaling (T=1.5) reducing ECE from 0.056 to 0.042 aligns with established best practices.

---

## 9. Glaucoma Screening: Clinical & Global Context

### 9.1 The Scale of the Problem

- Glaucoma affects **>76 million people globally** (Tham et al., 2014)
- Second leading cause of irreversible blindness
- Access to specialist ophthalmologists is limited in many regions
- Early detection through screening is critical

### 9.2 AI Deployment in Low-Resource Settings

**Field deployment experience (2025)** [^48^] highlights real-world challenges:
- **Image quality**: Handheld cameras produce variable illumination, sharpness, color
- **Workflow integration**: Manual transfer frictions, data entry time, user experience
- **Domain shift**: Models trained on public datasets (EyePACS/AIROGS) fail on portable camera images
- **Ground truth definition**: Different protocols (e.g., cup-to-disc ratio thresholds) across settings
- **Mixed results**: MobileNetv3 trained on AIROGS needed adaptation for real screening contexts

**Smart Fundus for India** [^45^]:
- AI-powered portable devices for retinal health assessment
- Integration of autoencoders, attention networks, GANs for retinal classification
- Enables teleophthalmology for remote diagnosis in rural/underserved areas
- Key challenges: dataset limitations, multimodal integration, model transparency, clinical acceptance

### 9.3 Cross-Dataset Generalization for Glaucoma

**Generative AI for Retinal Image Translation (2025)** [^26^]:
- Shows AUC performance varies significantly across racial subgroups for glaucoma detection
- Black patients: AUC 0.80-0.93 depending on model
- Asian patients: AUC 0.83-0.94
- White patients: AUC 0.88-0.93
- Demonstrates that domain adaptation via generative image translation can improve cross-population performance

**ViT for Glaucoma Detection (Fan et al., 2024)** [^31^]:
- ViT generalizes well to Chinese, Japanese, Spanish, African, and European descent populations
- ViT focuses on clinically relevant neuroretinal rim features
- However, performance varies across datasets, underscoring dataset-driven differences

### 9.4 Handheld Camera Challenges in India

Your specific context (desktop Western cameras vs. Indian handheld devices) reflects a documented challenge:
- **Remidio, Forus, Bosch** handheld cameras produce variable resolution, illumination, sharpness, color
- Camera-specific artifacts cause SFT oracle to fail (training on Remidio, testing on Bosch)
- Camera-aware adaptation is necessary but not sufficient alone

---

## 10. Foundation Models: Implicit Domain Generalization

A recent survey [^55^] establishes that foundation models represent a paradigm shift for domain adaptation:

**Key findings relevant to your DINOv3 choice**:
- Foundation models "operate in zero-shot or few-shot regimes, and their robustness is attributed to large-scale, diverse pretraining"
- They "exhibit remarkable zero-shot and few-shot transferability to previously unseen target distributions"
- "Foundation models consistently outperform traditional domain adaptation methods on zero-shot classification tasks, often without requiring any access to the target domain"
- Self-supervised objectives (contrastive learning, masked image modeling) produce representations "less sensitive to superficial domain-specific artifacts"

**This directly explains your zero-shot finding**: DINOv3 achieving AUROC 0.852 zero-shot is consistent with broader evidence that foundation models are inherently robust to domain shifts.

**Important caveat**: Foundation model features "degrade in scenarios requiring deep domain specialization" [^38^], which explains why adaptation (Netra-Adapt) can still improve upon the strong zero-shot baseline.

---

## 11. Key Negative Results: Scientific Value

Your paper documents several important negative results that contribute to the field:

### 11.1 RAG Failure (AUROC 0.509)
This aligns with findings that:
- Pure visual feature similarity cannot bridge ethnic gaps for subtle pathology [^8^]
- Feature space similarity alone cannot overcome cross-ethnic domain shift
- Domain shift displaces embeddings such that nearest neighbors in source data don't correspond to pathologically similar images in target

### 11.2 SFT Oracle Failure (AUROC 0.586)
This demonstrates **camera artifact overfitting**:
- "Severe overfitting to camera-specific artifacts" -- the model memorized brightness/contrast patterns rather than learning robust glaucoma features
- Training accuracy 87.3% vs. test accuracy 39.4% confirms memorization
- Model predicted "Glaucoma" for 93.7% of test cases -- collapse to a simple heuristic
- Highlights importance of domain-aware training when camera heterogeneity exists

### 11.3 Catastrophic TTA Collapse (AUROC 0.390)
This demonstrates the dangers of naive unsupervised adaptation:
- Entropy minimization alone drives collapse to trivial solutions [^46^]
- Without diversity constraints, models exploit confounding variables
- "The model stopped looking at the optic nerve and drifted toward random artifacts"
- This is a **safety-critical finding** for clinical deployment of TTA methods

---

## 12. Suggestions for Strengthening Your Paper

Based on this background research, here are specific recommendations:

### 12.1 Related Work Section
1. **Add FairDomain citation** [^5^] -- it's the most directly relevant prior work on fairness under domain shift in ophthalmology
2. **Add Rajesh et al. RPS citation** [^14^] -- provides biological grounding for pigmentation-based domain shift
3. **Add Burlina et al.** [^54^] -- established the pigmentation-performance link in retinal AI
4. **Cite the comprehensive SFDA survey** (TPAMI 2024) [^2^]
5. **Cite MEDM theoretical work** [^46^] -- provides theoretical justification for your entropy+diversity approach
6. **Add DINOv3 medical benchmark** [^38^] -- supports your choice of DINOv3 backbone

### 12.2 Methodology
1. **Consider using Equity-Scaled Performance (ESP) metrics** from FairDomain to evaluate fairness alongside accuracy
2. **Report per-camera ECE** to show calibration robustness across device types
3. **Consider RPS analysis**: If you can access RPS computation, quantifying pigmentation differences between AIROGS and Chakshu would strengthen your biological argument

### 12.3 Discussion
1. **Frame your negative results as contributions**: The RAG failure, collapse, and SFT oracle failure all provide important safety information for the field
2. **Connect to deployment challenges**: Cite field deployment work [^48^] showing real-world challenges align with your controlled experimental findings
3. **Discuss calibration more prominently**: Your ECE results (0.056) are strong and clinically relevant
4. **Position within foundation model paradigm**: Your zero-shot result (0.852) is consistent with broader evidence that foundation models are inherently cross-ethnic robust

### 12.4 Limitations & Future Work
1. **Single-sample TTA**: Cite emerging work on removing batch dependency (CPT4, single-sample variants)
2. **Multi-class extension**: Glaucoma staging is mentioned as future work
3. **Additional ethnic populations**: FairDomain provides a benchmark framework
4. **Theoretical guarantees**: Cite ongoing theoretical work on SFDA convergence

---

## 13. Complete Reference List (Suggested)

| Citation | Key Contribution | Where to Cite |
|----------|-----------------|---------------|
| Tian et al. (ECCV 2024) [^5^] | FairDomain: Fairness under domain shift | Related work, motivation |
| Rajesh et al. (Nat. Comm. 2025) [^14^] | Retinal Pigment Score | Biological motivation |
| Burlina et al. (TVST 2021) [^54^] | AI bias in retinal diagnostics | Related work |
| Wang et al. (IOVS 2025) [^44^] | Domain shift unfairness in DR | Motivation |
| Liang et al. (ICML 2020) [^11^] | SHOT: Foundational SFDA | Baseline, related work |
| Li et al. (AAAI 2025) [^1^] | AIF-SFDA: Medical SFDA | Related work |
| Wu et al. (2020) [^46^] | MEDM: Entropy vs. diversity theory | Theoretical grounding |
| Wang et al. (ICLR 2021) [^22^] | TENT: Test-time adaptation | Related work |
| Döbler et al. (2023), Niu et al. | Continual TTA stability | Related work |
| Caron et al. (ICCV 2021) | DINO self-supervised ViT | Backbone choice |
| DINOv3 Medical (2025) [^38^] | DINOv3 medical benchmark | Backbone justification |
| RETFound-Green (2025) [^50^] | Retinal foundation model | Related work |
| Tham et al. (2014) | Glaucoma prevalence stats | Introduction |
| de Vente et al. (IEEE TMI 2024) | AIROGS dataset | Methods |
| Kumar et al. (2023) | Chakshu dataset | Methods |
| Fan et al. (2024) [^31^] | ViT cross-ethnic glaucoma | Related work |
| Field Deployment (2025) [^48^] | Real-world deployment challenges | Discussion |
| Foundation Model Survey (2025) [^55^] | FM as implicit generalizers | Discussion |
| Calibration Study (2026) [^56^] | Technical vs. demographic factors | Discussion |

---

*Research compiled: May 4, 2026*
*Sources: 50+ papers from PubMed, arXiv, ECCV, ICML, ICLR, AAAI, IEEE TMI, Nature Communications*
