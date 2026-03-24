# ArtExtract Task 1 - Convolutional-Recurrent Architecture for Painting Classification

## Overview

The notebook implements a multi-task deep learning model for classifying paintings by **Artist** (23 classes), **Genre** (10 classes), and **Style** (27 classes) simultaneously. The model is built on a hybrid convolutional-recurrent architecture using the WikiArt/ArtGAN dataset.

---

## 1. Dataset and Data Strategy

The WikiArt dataset is distributed across six task-specific CSV files separate train and validation splits for each of the three tasks. A key challenge is that the three tasks cover different subsets of the full image collection:

- Style labels exist for **80,097** images (broadest coverage)
- Genre labels exist for **64,994** images
- Artist labels exist for only **19,052** images (restricted to the 23 target artists)

Rather than discarding images lacking labels for all three tasks, a unified master dataframe was constructed by merging all six CSVs by filename. Missing labels are assigned `-1` for the respective task. During training, `CrossEntropyLoss(ignore_index=-1)` automatically excludes unlabeled samples, allowing each classification head to learn from available labels. This approach, termed **partial-label multi-task learning**, maximizes data utilization for each head without requiring artificial label imputation.

---

## 2. Architectural Strategy

### Use of Two Backbones: ConvNeXt-Small and HRNet-W18:

Painting classification differs fundamentally from natural image classification, requiring two complementary perspectives:

**Macro scale - global texture, color, and composition:**
These features include overall color palettes, compositional balance, and structural forms characteristic of artistic styles. Such features require broad receptive fields and semantic understanding. **ConvNeXt-Small** provides superior performance through 7 \* 7 depthwise convolutions, enabling larger receptive fields compared to ResNet or EfficientNet architectures. This approach captures long-range texture statistics across the image while downsampling to an 8×8 feature map, encoding global characteristics at the cost of spatial detail.

**Micro scale - fine details, brushwork, and edge quality:**
These features distinguish artists within the same style, such as Van Gogh's thick impasto strokes versus Monet's delicate dabs, or Rembrandt's tonal transitions versus Frans Hals' expressive brushwork. Standard downsampling destroys such information. **HRNet-W18** maintains high-resolution branches throughout the network, preserving spatial details. Its multi-resolution fusion ensures fine brushstrokes remain represented in the final feature map.

The dual-backbone architecture captures complementary information: ConvNeXt encodes global visual characteristics while HRNet preserves technical execution details. Feature maps are projected to equal channel dimensions, concatenated, normalized via BatchNorm, and passed to the sequence model.

### Use of BiLSTM over the Spatial Tokens:

After the fusion block produces a feature map , it is flattened into a sequence of **64 spatial tokens**, each of dimension 512. These tokens are passed through a **Bidirectional LSTM**.

Paintings exhibit **spatial dependencies** across the canvas - stylistic properties in one region relate to properties in others. An artist's technique manifests consistently across the entire surface.Using BiLSTM models, these cross-spatial dependencies can be captured by processing feature token sequences and constructing global context that simple pooling operations cannot achieve.

Bidirectionality is essential because the spatial token sequence lacks a natural reading direction. Unlike sequential data (text, audio), spatial painting features are equally valid in both reading directions. A unidirectional LSTM would introduce arbitrary bias. The bidirectional architecture ensures each token incorporates contextual information from all surrounding tokens, producing a mean-pooled embedding representing the complete painting.

A single BiLSTM layer (hidden size 256) was selected over stacked layers. Additional layers would learn dependencies among already-contextualized representations across only 64 tokens, introducing parameter and computational overhead without significant accuracy improvement at this dataset scale.

**Pretrained weights:** ImageNet (ConvNeXt-Small), ImageNet (HRNet-W18)
**New layers trained from scratch:** fusion block, BiLSTM, classification heads

---

## 3. Training Strategy

### Handling Class Imbalance

The WikiArt dataset shows severe class imbalance across all three tasks. The style classification task is most extreme, with Impressionism comprising 13,060 training samples while Action_painting contains only 98, representing a **136:1** ratio. Without mitigation, the model would overpredict common classes and achieve artificially inflated accuracy while failing on rare classes.

Two complementary mechanisms address this imbalance:

**Inverse-frequency class weighting in the loss function:** Each class receives a weight inversely proportional to its frequency using the formula `w_c = total / (n_classes × count_c)`. This amplifies gradient penalties for rare-class errors, ensuring consistent optimization pressure across all classes.

**WeightedRandomSampler in the DataLoader:** Loss-function weighting corrects gradients but does not alter per-batch class distributions. The sampler rebalances training batches, providing approximately equal representation for rare and common classes across epochs. Weighting is driven by style class frequency due to its most severe imbalance.

Both mechanisms are necessary for effective training: the sampler ensures consistent exposure to rare classes, while the weighted loss ensures proportional optimization emphasis when errors occur.

### Label Smoothing

Label smoothing with coefficient 0.1 was applied to all three CrossEntropyLoss functions. Artistic classification contains inherent uncertainties - stylistic categories overlap temporally and technically, and artists often worked across multiple styles throughout their careers. Requiring maximum confidence in single-class predictions penalizes legitimate boundary cases. Label smoothing relaxes target confidence to 0.9, reducing overconfidence particularly for stylistically adjacent categories.

### Differential Learning Rates

Pretrained backbone weights (ConvNeXt-Small, HRNet-W18) employ a learning rate of `1e-5`, ten times lower than the `1e-4` rate applied to the fusion block, BiLSTM, and classification heads. This discrepancy preserves pre-trained ImageNet features while permitting rapid learning in new layers. A cosine annealing schedule with 3-epoch linear warmup prevents excessive gradient steps from randomly initialized head weights during early training.

---

## 4. Evaluation Metrics

### Primary: Macro F1-Score

Macro F1-score serves as the primary evaluation metric, computing F1-score independently for each class and averaging the results. This approach assigns **equal weight to all classes regardless of frequency distribution**. Standard accuracy proves inadequate for imbalanced datasets, as uniform prediction of the most common class achieves approximately 16% accuracy on the style task while failing completely on 26 other categories.

### Secondary: Top-1 and Top-5 Accuracy

Top-1 accuracy provides a standard baseline metric. Top-5 accuracy proves particularly informative for artist and style classification due to inherent ambiguity in artistic attribution - multiple artists worked in overlapping styles, and historical attributions undergo revision. Whether the correct class appears among the top-5 predictions better reflects learned representation quality than strict top-1 accuracy.

### Results

| Task                | Top-1 Accuracy | Top-5 Accuracy | Macro F1   |
| ------------------- | -------------- | -------------- | ---------- |
| Artist (23 classes) | ~90%           | ~98%           | 89.65%     |
| Genre (10 classes)  | ~81%           | ~99%           | 80.57%     |
| Style (27 classes)  | ~48%           | ~82%           | 52.30%     |
| **Average**         |                |                | **74.17%** |

Artist classification achieves the highest performance, as individual artist identity correlates strongly with brushwork characteristics that HRNet effectively captures. Style classification presents the greatest difficulty due to both severe class imbalance (136:1) and substantial visual overlap among style categories.

---

## 5. Outlier Detection

Two complementary methods were used to identify paintings that do not fit their assigned artist or genre label.

### Method 1: Per-Sample Cross-Entropy Loss Ranking

Per-sample cross-entropy loss is computed individually for each validation image (using `reduction='none'`). Samples are ranked by decreasing loss magnitude; high values indicate model confidence in predictions contradicting the assigned label, identifying primary outliers.

Analysis of these outliers reveals two categories: genuine dataset mislabelings (where the model has learned correct assignments contradicted by labels) and transitional works (paintings created during style transitions, legitimately ambiguous). Both categories provide valuable insights into dataset quality and labeling consistency.

### Method 2: Embedding Distance from Class Centroid

The 512-dimensional BiLSTM embedding (preceding classification heads) is extracted for all validation images. For each class, the centroid is computed as the mean of member embeddings. Cosine distance from the class centroid identifies images distant in embedding space from class-typical representations, indicating stylistic divergence from peers.

This approach identifies outliers distinct from loss-based methods. Correctly classified paintings may still exhibit unusual characteristics within their class - for example, a Picasso Blue Period work classified as Cubism may remain distant from typical Cubist representations due to chronological precedence relative to the artist's Cubist phase.

### Method 3: t-SNE Visualisation

Embeddings are projected to 2D using t-SNE for visualization. Well-separated clusters indicate successful learning of discriminative style representations. Points appearing in incorrect color clusters represent visually identifiable outliers - paintings whose learned representations align with non-assigned style categories. This visualization confirms the quantitative outlier detection results.

---

## 6. Discussion and Limitations

**Style classification** achieves 52.3% macro F1, reflecting the inherent difficulty of distinguishing 27 art historical styles with substantial temporal and technical overlap. The confusion matrix reveals most errors occur between adjacent styles - Impressionism/Post-Impressionism, Early Renaissance/High Renaissance, and Cubism/Analytical Cubism - categories whose boundaries remain contested within art historical scholarship.

**NaN training loss** at certain epochs results from extreme class weight amplification on rare-class batches, producing loss spikes exceeding fp16 representable range. The AMP gradient scaler detects and skips such steps, allowing training continuation. Future implementations should employ weight clamping (e.g., `torch.clamp(w, max=20.0)`) to prevent such numerical issues.

**For Task 2 (Similarity)**, the 512-dimensional embedding from the trained model serves directly as a feature extractor. The embedding space incorporates three complementary supervisory signals - artist identity, genre, and style - producing a rich representation sufficient for painting similarity tasks without requiring retraining from scratch.
