# ArtExtract Task 1 - Convolutional-Recurrent Architecture for Painting Classification

## Overview

This notebook implements a multi-task deep learning model for classifying paintings by **Artist** (23 classes), **Genre** (10 classes), and **Style** (27 classes) simultaneously. The model is built on a hybrid convolutional-recurrent architecture using the WikiArt/ArtGAN dataset.

---

## 1. Dataset and Data Strategy

The WikiArt dataset is distributed across six task-specific CSV files ~~-~~ separate train and validation splits for each of the three tasks. A key challenge is that the three tasks cover different subsets of the full image collection:

- Style labels exist for **80,097** images (broadest coverage)
- Genre labels exist for **64,994** images
- Artist labels exist for only **19,052** images (restricted to the 23 target artists)

Rather than discarding images that lack labels for all three tasks, a **unified master dataframe** was constructed by merging all six CSVs on filename. Images missing a label for a given task are assigned `-1` for that task. During training, `CrossEntropyLoss(ignore_index=-1)` automatically skips unlabelled rows, ensuring every image trains whichever heads it has labels for. This is known as **partial-label multi-task learning** and maximises the data available to each head without any artificial label imputation.

---

## 2. Architectural Strategy

### Why a Dual CNN Backbone?

Painting classification is fundamentally different from natural photo classification. The discriminative signals that separate artists and styles operate at two distinct spatial scales simultaneously:

**Macro scale - global texture, color, composition:**
These are signals like the overall palette of a Monet, the compositional balance of a Rembrandt, or the bold flat planes of Cubism. They require a wide receptive field and semantic abstraction. **ConvNeXt-Small** is ideal here : its 7×7 depthwise convolutions give it a larger receptive field than ResNet or EfficientNet, capturing long-range texture statistics across the canvas. It downsamples to a 8×8 feature map at the final stage, encoding rich global information at the cost of spatial precision.

**Micro scale - fine detail, brushwork, edge quality:**
These are the signals that separate artists within the same style - Van Gogh's swirling impasto strokes versus Monet's soft horizontal dabs, or Rembrandt's smooth tonal blending versus Frans Hals's loose brushwork. ConvNeXt destroys this information through aggressive downsampling. **HRNet-W18** solves this by maintaining a high-resolution branch throughout the entire network, never fully collapsing the spatial dimensions. Its multi-resolution fusion ensures that a thin brushstroke visible in the original image is still represented in the final feature map.

Together, the two backbones are complementary: ConvNeXt captures _what a painting looks like globally_, HRNet captures _how it was physically made_. Their feature maps are projected to equal channel dimensions and concatenated, then normalised with a BatchNorm layer before being passed to the sequence model.

### Why BiLSTM over the Spatial Tokens?

After the fusion block produces a 8×8×512 feature map (at 256px input), it is flattened into a sequence of **64 spatial tokens**, each of dimension 512. These tokens are passed through a **Bidirectional LSTM**.

The rationale for this design is that paintings contain **spatial dependencies** - the brushstroke character in the top-left corner of a canvas is not independent of the character in the bottom-right. An artist's style is not localised to one region; it manifests consistently across the whole surface. The LSTM models these cross-spatial dependencies by reading the sequence of feature tokens and building a global context representation that cannot be achieved by simple global average pooling alone.

Bidirectionality is essential here because the spatial sequence has no natural reading direction , unlike text or audio, a painting's spatial tokens are equally valid read left-to-right or right-to-left. A unidirectional LSTM would introduce an arbitrary directional bias. The bidirectional pass ensures every token is contextualised by all other tokens in both directions, producing a mean-pooled embedding that represents the full painting.

A single BiLSTM layer with hidden size 256 was chosen over two layers. The second layer would learn dependencies between already-contextualised representations across only 64 tokens , adding parameters and compute with negligible accuracy benefit at this dataset scale.

### Model Summary

```
Input Image (256×256×3)
    │
    ├─── ConvNeXt-Small ──── 8×8×768   (global texture, composition)
    └─── HRNet-W18      ──── 8×8×2048  (fine detail, brushwork)
              │
         1×1 Conv projection → 8×8×512
         BatchNorm fusion
              │
         Flatten → 64 spatial tokens × 512
              │
         Bidirectional LSTM (hidden=256, layers=1)
         Mean-pool → 512-dim embedding
              │
    ┌─────────┼─────────┐
  Artist    Genre     Style
   (23)      (10)      (27)
```

**Total parameters:** ~71M
**Pretrained weights:** ImageNet (ConvNeXt-Small), ImageNet (HRNet-W18)
**New layers trained from scratch:** fusion block, BiLSTM, classification heads

---

## 3. Training Strategy

### Handling Class Imbalance

The WikiArt dataset has severe class imbalance across all three tasks. The style task is the most extreme : Impressionism has 9,020 training images while Action_painting has only 66, a ratio of **136:1**. Without correction, the model would learn to predict "Impressionism" for ambiguous inputs and achieve misleadingly high accuracy while failing completely on rare styles.

Two complementary mechanisms were applied:

**Inverse-frequency class weights in the loss function:** Each class receives a weight inversely proportional to its frequency (`w_c = total / (n_classes × count_c)`). This amplifies the gradient penalty for errors on rare classes, forcing the model to take them seriously during every parameter update.

**WeightedRandomSampler in the DataLoader:** Class weights in the loss correct the gradient but do not change what the model sees per batch. The sampler rebalances the training batches themselves, giving rare classes approximately equal representation to common ones across each epoch. This is driven by style class frequency since style has the most severe imbalance.

Both of these mechanisms are required to for proper training . The sampler ensures rare classes are seen regularly; the weighted loss ensures their errors are penalised proportionally when they occur.

### Label Smoothing

`label_smoothing=0.1` was applied to all three CrossEntropyLoss functions. Art classification has inherently ambiguous boundaries Impressionism bleeds into Post-Impressionism, many artists painted in multiple styles throughout their careers. Hard labels that force the model toward 1.0 confidence on the correct class penalise legitimate ambiguity. Label smoothing softens the targets to 0.9, preventing overconfident predictions particularly on the boundary cases between stylistically adjacent categories.

### Differential Learning Rates

Pretrained backbone weights (ConvNeXt-Small, HRNet-W18) used a learning rate of `1e-5` - ten times lower than the `1e-4` used for the fusion block, BiLSTM, and classification heads. This prevents catastrophic forgetting of the ImageNet representations during early training while allowing the new layers to learn quickly. A cosine scheduler with 3-epoch linear warmup was used to prevent large gradient steps from randomly initialised head weights during the first few epochs.

---

## 4. Evaluation Metrics

### Primary: Macro F1-Score

Macro F1 is the primary evaluation metric for all three tasks. It computes the F1 score independently for each class and averages them, giving **equal weight to every class regardless of its frequency**. This metric is one of the correct metric for imbalanced datasets - accuracy would be misleading since a model predicting "Impressionism" for everything would achieve approx 16% accuracy on the style task while completely failing on 26 other styles.

### Secondary: Top-1 and Top-5 Accuracy

Top-1 accuracy provides a standard baseline comparison. Top-5 accuracy is particularly informative for artist and style classification - painting attribution is inherently ambiguous (many artists worked in similar styles, and attribution has been revised by art historians), so knowing whether the correct class appears in the model's top-5 predictions is a more meaningful measure of representation quality than strict top-1 correctness.

### Results

| Task                | Top-1 Accuracy | Top-5 Accuracy | Macro F1   |
| ------------------- | -------------- | -------------- | ---------- |
| Artist (23 classes) | ~90%           | ~98%           | 89.65%     |
| Genre (10 classes)  | ~81%           | ~99%           | 80.57%     |
| Style (27 classes)  | ~48%           | ~82%           | 52.30%     |
| **Average**         |                |                | **74.17%** |

Artist classification performs best, reflecting that individual artist identity is strongly encoded in brushwork texture which the HRNet branch captures well. Style classification is hardest due to the 136:1 class imbalance and the genuine visual overlap between many style categories.

---

## 5. Outlier Detection

Two complementary methods were used to identify paintings that do not fit their assigned artist or genre label.

### Method 1: Per-Sample Cross-Entropy Loss Ranking

For every image in the validation set, the cross-entropy loss is computed individually (using `reduction='none'`). Images are sorted by descending loss : the highest loss values correspond to cases where the model was strongly confident in a prediction that contradicted the assigned label. These are the primary outliers.

Interpreting these outliers reveals two patterns: genuine mislabellings in the dataset (where the model has learned the correct style and is confused by an incorrect label), and transitional works (paintings made at a period when an artist was shifting between styles, making them legitimately ambiguous). Both types are valuable for understanding dataset quality.

### Method 2: Embedding Distance from Class Centroid

The 512-dimensional embedding produced by the BiLSTM layer (before the classification heads) is extracted for every validation image. For each class, the centroid is computed as the mean of all member embeddings. Cosine distance from this centroid is then computed for every class member - images that are far from their class centroid in embedding space are stylistically dissimilar from the majority of their class peers.

This method captures a different kind of outlier from the loss-based method. A painting might be correctly classified (low loss) but still be unusual within its class - for example, a Picasso painting from his Blue Period classified correctly as Cubism but positioned far from other Cubist works in embedding space because it predates his Cubist phase.

### Method 3: t-SNE Visualisation

The embeddings are projected to 2D using t-SNE for visual inspection. Well-separated clusters indicate that the model has learned discriminative representations for each style. Points that appear in the wrong colour cluster are visually identifiable outliers - paintings whose learned representation places them closer to a different style category than their assigned one. This provides an intuitive visual confirmation of the quantitative outlier detection methods above.

---

## 6. Discussion and Limitations

**Style classification** at 52.3% macro F1 is the hardest task and reflects the genuine difficulty of distinguishing 27 art historical styles that overlap significantly in time period and technique. The confusion matrix shows that the most common errors are between adjacent styles - Impressionism/Post-Impressionism, Early Renaissance/High Renaissance, and Cubism/Analytical Cubism - which are categories that art historians themselves debate the boundaries of.

**The NaN training loss** observed at certain epochs is attributable to extreme class weight amplification on a small number of rare-class batches producing loss spikes that exceed fp16 representable range. The AMP gradient scaler detects and skips these steps, so the model continues training correctly. Clamping class weights to a maximum value (e.g. `torch.clamp(w, max=20.0)`) would prevent this in future runs.

**For Task 2 (Similarity)**, the 512-dim embedding from this trained model is directly reusable as a feature extractor. The embedding space has been shaped by three complementary supervisory signals - artist identity, genre, and style - making it a rich representation for painting similarity queries without requiring retraining from scratch.
