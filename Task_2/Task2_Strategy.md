# ArtExtract Task 2 - Painting Similarity Search

---

## 1. Problem Framing

Task-2 requires building a model to find similarities in paintings, with portraits having similar faces or poses as an explicit example. This is an **unsupervised retrieval problem**. The NGA Open Dataset provides no labelled similarity pairs - there is no ground truth that says painting A is similar to painting B. This distinction determines every architectural and evaluation decision made in this notebook.

The most important part is defining what "similar" means. Two paintings can be similar in multiple ways simultaneously: compositional similarity (same pose, same spatial layout), stylistic similarity (same period, same technique, same school), or content similarity (same subject matter, same iconography). A good retrieval model should surface all of these without collapsing them into a single dimension.

---

## 2. Why is the Task 1 Backbone Reused

The Task 1 model (ConvNeXt-Small + HRNet-W18 + BiLSTM) was trained on 80,097 WikiArt paintings across three supervisory signals: artist identity, genre, and style. This training produced an embedding space that already encodes painting-specific visual features - brushstroke texture, compositional structure, colour palette distribution, spatial layout - that are directly relevant to visual similarity.

Training a similarity model from scratch on NGA alone would require either large volumes of labelled similarity pairs (which do not exist in this dataset) or self-supervised pretraining on enough painting data to learn these domain-specific representations. The Task 1 backbone already provides this at no additional cost.

The backbone is kept entirely frozen throughout Task 2. Every parameter in ConvNeXt-Small, HRNet-W18, and the BiLSTM is frozen before any embedding is extracted. This is done for two reasons. First, fine-tuning the backbone with the NGA dataset (approximately 4,000 paintings, far smaller than WikiArt) risks destroying the learned WikiArt representations through catastrophic forgetting. Second, NGA provides no explicit similarity labels, so any fine-tuning signal would come from proxy labels that do not directly supervise the backbone layers.

---

## 3. Architecture: Frozen Backbone and Projection Head

The pipeline operates in two stages.

**Stage 1 :** The frozen backbone extracts a 512-dimensional embedding for each NGA painting. These embeddings are L2-normalised onto the unit hypersphere, which converts the FAISS inner product search into equivalent cosine similarity search. This single pass through the backbone is the computationally expensive step and runs once.

**Stage 2 :** A lightweight projection head (a 3-layer MLP: 512 to 512 to 512 to 256 with BatchNorm, GELU activations, and L2 normalisation at the output) is trained on top of the frozen embeddings. Only this head's parameters update. The projection head reshapes the 512-dimensional embedding space into a 256-dimensional space that is explicitly optimised for similarity retrieval rather than classification.

The projection head uses three layers rather than two. The additional hidden layer provides more capacity to non-linearly transform the embedding space, which is important because the original space was shaped by three classification objectives that do not directly optimise for metric retrieval.

---

## 4. Loss Function: Supervised Contrastive Loss

The projection head is trained with **Supervised Contrastive Loss** rather than triplet loss.

Triplet loss trains on (anchor, positive, negative) triplets sampled one at a time. Each training step uses exactly one positive pair and one negative pair per anchor, which is informationally wasteful given that batches of 512 embeddings contain many more usable positive pairs.

SupCon uses the entire batch simultaneously. For a batch of 512 embeddings, every pair sharing the same attribution label is treated as a positive pair. With 512 items and multiple attributions represented per batch, each anchor typically has several positives and many negatives, all contributing to the gradient in a single step. This makes the loss signal substantially denser than triplet loss and allows the embedding space to converge more consistently.

The attribution label used as pseudo-ground truth is drawn from the `attributioninverted` column (format: "Firstname Lastname" or "Workshop of Artist") rather than the `attribution` column (format: "Lastname, Firstname"). This ensures consistent string matching when comparing attributions across paintings, since the same artist name would otherwise produce non-matching strings depending on which format a particular row uses.

Only attributions with at least three works are used for training. Attributions with fewer than three works cannot provide at least one anchor, one positive, and one negative within the same batch, making them uninformative for contrastive learning.

The temperature parameter is set to 0.07, which is the standard value from the original SupCon paper. Lower temperature sharpens the similarity distribution, pushing the model to be more discriminative. A temperature of 0.07 has been validated across a range of visual retrieval tasks and was not tuned for this specific dataset.

The training uses OneCycleLR scheduling with a maximum learning rate of 1e-3 and 10% warmup over 20 epochs. OneCycleLR is preferred over cosine annealing for small datasets because the aggressive learning rate increase during warmup helps the randomly initialised projection head escape flat regions early, while the cosine decay in the later phase prevents overshooting as the space converges.

---

## 5. Dataset: NGA Filtering and Image Acquisition

The NGA Open Dataset contains 144,000+ objects spanning prints, drawings, photographs, sculptures, and other categories. Of these, 4,430 are classified as paintings under the `visualbrowserclassification` column. This column was used rather than the `classification` column because `visualbrowserclassification` uses standardised lowercase values ("painting", "print", "drawing") while `classification` contains mixed capitalisation and additional subcategory values.

Images are downloaded via the IIIF API using the `iiifthumburl` field from `published_images.csv`. The thumbnail URL is modified to request images at 256 pixels, matching the input size used during Task 1 training. Only the primary view (`viewtype == 'primary'`) of each painting is used to avoid duplicate representations of the same work from different angles or photography sessions.

The category used for retrieval queries is derived from painting titles using regular expression keyword matching. The NGA `classification` column assigns nearly all paintings the label "Painting" without further subcategory, making it useless for filtering by content type. Title keywords reliably identify portraits, landscapes, religious scenes, and still lifes because NGA titles follow consistent naming conventions (e.g. "Portrait of a Lady", "Landscape with River", "Madonna and Child").

---

## 6. Similarity Search: FAISS

Two FAISS indices are built and maintained in parallel: `faiss_raw.bin` using raw 512-dim embeddings and `faiss_proj.bin` using projected 256-dim embeddings. Both use `IndexFlatIP` (exact inner product search). Since all embeddings are L2-normalised to the unit sphere before indexing, inner product is mathematically equivalent to cosine similarity.

Approximate search indices (such as FAISS IVF or HNSW) were not used. With approximately 4,000 paintings, exact search is computationally trivial at query time and produces exact results without approximation error. Approximate indices become necessary at scales of hundreds of thousands of vectors or more.

The raw index serves as the baseline for evaluation. The projected index is the operational index used for all retrieval in the similarity visualisations and evaluation metrics.

---

## 7. Evaluation Metrics

The absence of ground-truth similarity labels requires proxy metrics. Two evaluation tracks are used, together with one embedding space geometry metric.

**Attribution Match (Precision@5 and MRR@10)**

Attribution match asks the question that do the top-K retrieved paintings share the same artist or school as the query. This is a reasonable proxy since artists have consistent compositional preferences, pose conventions, and subject matter. An artist X's portrait retrieved in response to a different X's portrait query is a good similarity hit.

The ceiling for this metric is constrained by dataset sparsity. With approximately 4,000 paintings distributed across hundreds of artists and workshops, the average NGA collection size per attribution is around 13 works. The theoretical maximum Precision@5 under these conditions is approximately 0.24, assuming that the top 5 results always come from the same attribution. This ceiling is important context when interpreting the numeric results: a value near 0.24 indicates that the model is essentially saturating the available positive pairs, not that it is underperforming.

**Category Coherence (Precision@5 and MRR@10) - Primary Metric**

Category coherence asks whether top-K results for a portrait query include other portraits. This metric is primary because it directly measures whether the retrieval system surfaces visually similar content as specified by the task.

Retrieving a portrait by artist Y in response to a query of artist Z's portrait counts as correct despite different attributions, when both works share similar style, period, and compositional structure. This metric prioritizes visual correspondence—matching faces and poses—over artist identity. Track A classifies this result as incorrect; Track B classifies it as correct.

Category labels are derived from title keywords rather than the `classification` column, for the reasons described in Section 5.

**Intra/Inter Cosine Similarity Ratio**

This metric measures the geometric quality of the entire embedding space, independent of any particular query. For each artist attribution, the mean pairwise cosine similarity among that artist's works is computed (intra-class similarity). This is then divided by the mean cosine similarity between that artist's works and randomly sampled works from different attributions (inter-class similarity). A ratio above 1.0 indicates that the embedding space clusters paintings by artist/school more tightly than random. The target range is 5 to 15 times.

A ratio substantially above 20 indicates dimensional collapse: inter-class similarity has approached zero, meaning the model has learned to map all paintings to a small region of the embedding space rather than distributing them meaningfully. The previous version of this notebook (with triplet loss) produced a ratio of 38, which was indicative of this collapse. The SupCon formulation with temperature 0.07 avoids this by normalising against all pairs in the batch simultaneously, preventing the inter-class distances from collapsing.

All three metrics are computed on both raw and projected embeddings, and the delta is printed explicitly. This demonstrates whether the projection head training improved the embedding space, rather than simply asserting improvement without evidence.

---

## 8. Alternative Approaches: Why They Were Not Pursued

**Backbone fine-tuning.** Fine-tuning the full ConvNeXt-Small and HRNet-W18 on NGA data was considered and rejected. NGA has approximately 4,000 paintings, compared to the 80,000 used in Task 1 training. Fine-tuning 50 million backbone parameters on a dataset this small would result in overfitting the NGA distribution and forgetting the broader WikiArt-learned representations. The projection head approach is the principled choice: it adapts the output of the backbone without modifying the backbone itself.

**Self-supervised contrastive pretraining on NGA from scratch.** Methods such as SimCLR or MoCo, which learn representations without labels by contrasting augmented views of the same image, were not applied. These methods require thousands of training images to learn meaningful representations and typically need very large batch sizes (256 to 4096) to work effectively. The NGA painting subset (4,000 images) is too small for stable self-supervised pretraining from scratch, and the domain overlap with WikiArt makes transfer from Task 1 a more reliable starting point.

**Ground-truth pair collection.** Manual annotation of similarity pairs was not performed. While this would produce the cleanest possible evaluation labels, it is outside the scope of what can be done programmatically within a single notebook and would require expert art-historical judgement for non-trivial cases.

**HNSW or IVF approximate search.** These indexing strategies improve query speed at the cost of result exactness. At 4,000 vectors, exact search with IndexFlatIP takes microseconds per query. There is no performance justification for introducing approximation error at this dataset size.

---

## 9. Output Files

| File                    | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| `embeddings.npy`        | Raw 512-dim L2-normalised embeddings for all NGA paintings        |
| `embeddings_proj.npy`   | Projected 256-dim embeddings after SupCon fine-tuning             |
| `faiss_raw.bin`         | FAISS index over raw embeddings (baseline)                        |
| `faiss_proj.bin`        | FAISS index over projected embeddings (use for retrieval)         |
| `proj_head_supcon.pth`  | Trained projection head weights                                   |
| `index_meta.csv`        | Metadata for every indexed painting (title, attribution, path)    |
| `nga_meta.csv`          | Full NGA painting metadata including medium and display date      |
| `eval_results.json`     | All numeric evaluation results for both tracks, raw and projected |
| `supcon_loss_curve.png` | Training loss curve for the projection head                       |
| `tsne_projected.png`    | t-SNE visualisation of projected embedding space                  |
| `portrait_*.png`        | Portrait similarity retrieval grids                               |
| `landscape_*.png`       | Landscape similarity retrieval grids                              |
| `religious_*.png`       | Religious painting similarity retrieval grids                     |
| `still_*.png`           | Still life similarity retrieval grids                             |
