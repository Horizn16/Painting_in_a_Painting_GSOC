# ArtExtract

ML pipeline for painting classification and visual similarity search, trained on the WikiArt and NGA Open datasets.

---

## Project Structure

```
.
├── README.md
├── Task_1
│   ├── checkpoints
│   ├── ckpt_best.pth
│   ├── logs
│   ├── output
│   ├── Task1.ipynb             # Painting classification model
│   └── Task1_Strategy.md       # Architecture and training decisions for Task 1
└── Task_2
    ├── nga
    ├── output
    ├── output.zip
    ├── Task2.ipynb             # Painting similarity search
    └── Task2_Strategy.md       # Architecture and training decisions for Task 2

```

---

## Task 1: Painting Classification

Classifies paintings across three tasks simultaneously: Artist (23 classes), Genre (10 classes), and Style (27 classes).

### Model Architecture

A hybrid convolutional-recurrent architecture with two parallel backbones:

- **ConvNeXt-Small** -- captures global texture, color palette, and compositional structure via large receptive fields (7x7 depthwise convolutions)
- **HRNet-W18** -- preserves fine-grained brushwork and edge detail through high-resolution branches maintained throughout the network
- **BiLSTM** -- models spatial dependencies across 64 feature tokens produced by the fused backbone output

Feature maps from both backbones are projected to equal channel dimensions, concatenated, normalized, and passed as a token sequence to the BiLSTM. The mean-pooled BiLSTM hidden state forms the 512-dimensional painting embedding used for all three classification heads.

### Training

| Component                   | Learning Rate |
| --------------------------- | ------------- |
| ConvNeXt-Small, HRNet-W18   | 1e-5          |
| Fusion block, BiLSTM, heads | 1e-4          |

- Loss: CrossEntropyLoss with `ignore_index=-1` for partial labels, inverse-frequency class weights, and label smoothing (0.1)
- Sampler: WeightedRandomSampler to address class imbalance (up to 136:1 ratio in style)
- Schedule: Cosine annealing with 3-epoch linear warmup
- Pretrained weights: ImageNet for both backbones

### Results

| Task                | Top-1 Accuracy | Top-5 Accuracy | Macro F1   |
| ------------------- | -------------- | -------------- | ---------- |
| Artist (23 classes) | ~90%           | ~98%           | 89.65%     |
| Genre (10 classes)  | ~81%           | ~99%           | 80.57%     |
| Style (27 classes)  | ~48%           | ~82%           | 52.30%     |
| **Average**         |                |                | **74.17%** |

---

## Task 2: Painting Similarity Search

Retrieves visually similar paintings from the NGA Open Dataset (~4,000 paintings) using embeddings from the Task 1 backbone.

### Pipeline

**Stage 1 -- Feature extraction (frozen backbone)**
The Task 1 backbone (ConvNeXt-Small + HRNet-W18 + BiLSTM) is kept fully frozen. All backbone parameters are fixed before any NGA embedding is extracted. Each painting is encoded into a 512-dimensional L2-normalised embedding.

**Stage 2 -- Projection head training**
A lightweight 3-layer MLP (512 -> 512 -> 512 -> 256, BatchNorm + GELU + L2 norm) is trained on top of the frozen embeddings using Supervised Contrastive Loss (SupCon). Only the projection head parameters are updated. The pseudo-labels used are artist attributions from the `attributioninverted` column, filtered to attributions with at least 3 works.

- Temperature: 0.07
- Batch size: 512
- Epochs: 20
- Schedule: OneCycleLR, max LR 1e-3, 10% warmup

### Similarity Search

Two FAISS indices (`IndexFlatIP`, exact search) are built in parallel:

- `faiss_raw.bin` -- 512-dim raw embeddings (baseline)
- `faiss_proj.bin` -- 256-dim projected embeddings (operational)

Inner product on L2-normalised vectors is equivalent to cosine similarity.

### Evaluation

Three proxy metrics are used in the absence of ground-truth similarity labels:

| Metric                                   | Description                                                                             |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| Attribution Match (Precision@5, MRR@10)  | Fraction of top-K results sharing the same artist or school                             |
| Category Coherence (Precision@5, MRR@10) | Fraction of top-K results sharing the same content category (portrait, landscape, etc.) |
| Intra/Inter Cosine Ratio                 | Ratio of within-artist to cross-artist mean cosine similarity; target range 5 to 15     |

Category coherence is the primary metric as it directly measures visual content similarity.

### Output Files

| File                   | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `embeddings.npy`       | Raw 512-dim embeddings for all NGA paintings          |
| `embeddings_proj.npy`  | Projected 256-dim embeddings                          |
| `faiss_raw.bin`        | FAISS index over raw embeddings                       |
| `faiss_proj.bin`       | FAISS index over projected embeddings                 |
| `proj_head_supcon.pth` | Trained projection head weights                       |
| `index_meta.csv`       | Metadata for all indexed paintings                    |
| `nga_meta.csv`         | Full NGA painting metadata                            |
| `eval_results.json`    | Evaluation results for both tracks, raw and projected |

---

## Dataset Sources

- **WikiArt / ArtGAN** -- 80,097 paintings across style, genre, and artist labels (Task 1 training)
- **NGA Open Dataset** -- 4,430 paintings filtered from 144,000+ objects by `visualbrowserclassification == "painting"` (Task 2)
- NGA images fetched via IIIF API at 256px resolution using primary views only

---

## Dependencies

- PyTorch
- timm (ConvNeXt-Small, HRNet-W18)
- FAISS
- NumPy, Pandas
- scikit-learn
- Matplotlib
