# Project Explanation: Deep Learning for SAR Target Recognition

**What this document covers**: The problem being solved, why it is hard, and a plain-language explanation of every model and technique used — written so it can be presented or discussed quickly.

---

## 1. The Problem

### What is SAR?

Synthetic Aperture Radar (SAR) is a type of radar mounted on aircraft or satellites. Unlike a camera, it does not need sunlight or clear weather — it works at night, through clouds, and through smoke. It emits radio waves and records how they bounce back off the ground. The result is a grayscale image where bright pixels mean the surface strongly reflected the radar signal, and dark pixels mean it absorbed or deflected it.

Because of this, SAR is extremely valuable for military surveillance, disaster response, and border monitoring — you get an image of the scene regardless of conditions.

### What is ATR?

Automatic Target Recognition (ATR) means having a computer automatically identify objects in a SAR image — specifically, distinguishing between different types of ground vehicles. Instead of a human analyst staring at hundreds of images, an algorithm reads the image and says "that is a T72 tank" or "that is a BMP2 armored vehicle."

The dataset used here is **MSTAR** (Moving and Stationary Target Acquisition and Recognition), a publicly available benchmark from the US Air Force Research Laboratory. It contains SAR images of 10 vehicle types photographed at various aspect angles (the radar circles around the target recording it from different directions).

### Why is it hard?

There are three overlapping problems:

**Problem 1 — Data Scarcity**
Getting real SAR images of military vehicles is expensive and restricted. You cannot simply collect thousands of labeled images the way you would for a face recognition dataset. In the MSTAR sample used here, there are only 100 images per vehicle class — far fewer than typical deep learning scenarios. A standard CNN trained on 100 images per class will overfit and fail to generalize.

**Problem 2 — Open-Set Recognition**
In real deployment, the radar might see a vehicle type it was never trained on — an enemy vehicle not in the training database. A standard classifier will still assign it to one of the known classes (it has no choice — it always picks a winner). The system needs to say "I recognize this as a T72 tank" for known vehicles AND say "I don't recognize this — it is something unknown" for novel vehicles. This is called **open-set recognition (OSR)**.

**Problem 3 — Physical Variation**
SAR images of the same vehicle look very different depending on:
- **Aspect angle**: the angle from which the radar looks at the vehicle. A tank photographed from the front looks completely different from one photographed from the side.
- **Depression angle**: whether the radar is looking steeply down or at a shallow angle.
- **Speckle noise**: SAR images contain a specific granular noise pattern caused by the coherent nature of radar waves. This noise changes randomly between images of the same vehicle.

A model that does not account for these variations will fail when deployed in conditions slightly different from training.

---

## 2. The Solution

The system addresses all three problems together using a pipeline of complementary techniques:

```
Raw SAR image
      │
      ▼
[CNN Feature Extractor]  ──── extracts a compact 64-D signature
      │
      ▼
[GNN Relation Space]  ──── compares signatures across a small set of labeled examples
      │
      ▼
Prediction: seen class (one of C known types) OR "unseen" (unknown vehicle)
```

Supporting this core are:
- A **GAN** that generates extra synthetic training images to compensate for data scarcity
- A **physics loss** that encourages the model to respect known radar scattering properties
- **Augmentation transforms** (rotation, speckle noise) for invariance

Each component is explained in detail below.

---

## 3. Why Few-Shot Learning?

### The core idea

Instead of training a model that memorizes 100 images per class and then classifies, **few-shot learning (FSL)** trains a model to *compare*. At test time, you give the model:

- A **support set**: a few labeled examples (e.g., 5 images each of 3 vehicle types)
- A **query image**: the unknown image you want to classify

The model learns to ask: "Does this query image look more like the T62 examples, the BMP2 examples, or none of them (unseen)?" It compares rather than memorizes.

This generalizes much better with small data because the model is learning a universal *comparison skill*, not class-specific patterns. Training happens across thousands of randomly generated *episodes*, each with a different subset of classes and different images — forcing the model to be general.

### The episodic training setup

Each training step is one **episode** (also called a task):
- Pick C classes randomly from the training set
- Sample K images per class → **support set** (C × K images total)
- Sample 1 query image from one of those classes, or from the *unseen* class → **query**
- Ask the model to classify the query

In this project: C = 3 (or 5 or 7), K = 20. The unseen class is T72.

---

## 4. Why CNN? (The Feature Extractor)

### What it does

The Convolutional Neural Network (CNN) is the first stage. It takes a raw 100×100 grayscale SAR image and compresses it into a **64-dimensional feature vector** — a list of 64 numbers that captures the essential visual characteristics of the image (shape, texture, scattering pattern).

Think of it as the model's "perception" — it reads the raw pixels and extracts a meaningful summary.

### Why 4 layers with 32 filters each?

This is directly specified by the reference paper (Zhou et al., Sensors 2023). The paper shows that:
- 4 layers with uniform 32 filters strike the right balance for 100×100 SAR images — deep enough to capture texture patterns, not so deep that it overfits with 100 training images per class.
- Doubling the filters each layer (32→64→128→...) is common in general vision but wastes capacity on small grayscale SAR images.

The original code used 3 layers with doubling channels — a mismatch from the paper that hurt accuracy.

### Why SE (Squeeze-and-Excitation) attention?

After each convolutional layer, a small attention module asks: "Which of the 32 feature channels are actually informative for this image?" It computes a score for each channel and amplifies the useful ones while suppressing noise.

The mechanism:
1. **Squeeze**: average-pool the entire spatial map into one number per channel — "how active is this channel overall?"
2. **Excite**: pass those 32 numbers through a small neural network → produce 32 weights between 0 and 1
3. **Scale**: multiply each channel by its weight

For SAR images, different channels respond to different scattering patterns. The attention lets the model focus on the channels that are discriminative for the current image and ignore noisy ones. The original code had this module written but commented out — it was dead code. Re-activating it is one of the key fixes.

---

## 5. Why GNN? (The Relation Space)

### What it does

The Graph Neural Network (GNN) takes the 64-D CNN embeddings of all support images and the query image and reasons about how they relate to each other.

Imagine placing all the images as nodes in a graph. The GNN passes messages between nodes — each node updates its understanding of itself based on what it hears from its neighbors. After a few rounds of message passing, the query node accumulates evidence from all the support examples and produces a score for each possible class.

### Why not just use distance (nearest neighbor)?

Nearest-neighbor (compare query to each support image by Euclidean distance, pick the closest class) is simple but brittle:
- It treats all support examples equally — even the atypical ones that happen to look like the query for the wrong reason.
- It does not let the support examples learn from each other — a "confused" support image cannot be corrected by its neighbors.

The GNN learns *adaptive, task-specific similarity* — the edge weights between nodes are computed by the model and change depending on the specific set of images in the episode. Two support images from the same class will strongly connect to each other and weakly connect to the query from a different class.

### The specific structure (3 adjacency + 2 update)

Following the reference paper exactly:

```
Initial node features (CNN embeddings + class label one-hot)
        │
   Adj0: compute pairwise similarity matrix
        │
   Upd0: each node aggregates from neighbors → new partial features
        │
   Concatenate (original + new) → features grow
        │
   Adj1: recompute similarity on richer features
        │
   Upd1: aggregate again
        │
   Concatenate again → features grow further
        │
   Adj2: final similarity on fully enriched features
        │
   Read out query node → FC layer → class scores
```

**Why this alternating structure?** The adjacency matrix (who is similar to whom) improves as the node features improve, and the node features improve as the adjacency matrix improves. Alternating them creates a feedback loop — each round of message passing makes the graph smarter about similarity.

The original code used a simpler formula (standard spectral graph convolution: A·X·W) which does not have this feedback property. Switching to the paper's formula is the second key fix.

### How open-set detection works

Every episode has C+1 output slots: one per known class, plus one "unknown" slot. The GNN learns to assign a high score to the unknown slot when the query image does not match any of the support examples well. This emerges naturally from the training setup — the model sees unseen-class queries half the time during training and must learn to distinguish "looks like one of the support classes" from "looks like none of them."

---

## 6. Why GAN? (Solving Data Scarcity)

### The problem it solves

With only 100 real images per vehicle class, the training set is small. The GAN learns to generate additional synthetic SAR images that look like they could have been photographed by the radar.

### What a Conditional DCGAN is

**GAN** (Generative Adversarial Network): Two networks trained against each other.
- The **Generator** takes random noise + a class label → produces a synthetic image
- The **Discriminator** looks at an image + its claimed class label → says "real" or "fake"

The generator tries to fool the discriminator. The discriminator tries to catch the generator. Through this competition, the generator learns to produce increasingly realistic images.

**Conditional**: The class label is given as input to both networks. This ensures the generator produces images of a specific vehicle type, not random vehicles.

**DC** (Deep Convolutional): Uses convolutional layers (good for images) rather than fully connected layers — standard practice since DCGAN (Radford et al., 2015).

### How it integrates

The GAN is trained separately on real MSTAR images. After training, it generates 100 synthetic images per class saved to disk. During few-shot training with `--gan_augment`, each support set is augmented with K synthetic images per class (matching the K real images). The model sees more variety in the support set, reducing overfitting.

### Quality check

After generation, the synthetic images are fed through the trained CNN classifier. If the classifier cannot recognize them at ≥70% accuracy per class, it means the GAN has not learned the right features for that class and a warning is shown. This prevents silently training on garbage data.

---

## 7. Why Physics Loss? (Imposing Domain Knowledge)

### The problem it solves

Pure data-driven learning ignores what physicists know about radar. SAR images obey specific statistical laws:
- Speckle noise follows a **Rayleigh distribution** (it is not Gaussian noise)
- The same vehicle type, viewed from similar angles, should produce similar electromagnetic backscatter — meaning similar feature vectors

The model has no way to learn these laws from 100 images alone. The physics loss explicitly encodes one of these laws as a training constraint.

### What the physics loss penalizes

For each vehicle class in the support set, it collects the CNN embedding vectors of all K support images of that class and computes their **variance**. If the K embeddings of the same vehicle class are spread far apart in feature space, it adds a penalty to the loss. This pushes same-class embeddings to cluster together.

```
L_total = L_classification + λ × L_physics

where L_physics = average variance of embeddings within each support class
```

**Why this makes physical sense**: The same vehicle photographed under slightly different conditions should scatter radar waves in a consistent way — the feature vector should not jump around randomly. Minimizing within-class variance enforces this physics-grounded expectation.

**Why not full electromagnetic constraints?** Implementing wave propagation equations (Maxwell's equations) as a loss term requires specialized radar simulation software (XPATCH, OpenEMS) and detailed target 3D models — beyond the scope of the available tools. The variance penalty is a pragmatic approximation that achieves a similar regularization effect using only the existing data.

---

## 8. Why Rotation and Speckle Augmentation?

### The problem they solve

At test time, vehicles will appear at angles the model has never seen and with different noise realizations. Augmentation during training synthetically creates this variety so the model learns to be robust.

### Rotation (RandomRotation360)

SAR sensors orbit around a target, so the same vehicle can appear rotated at any angle from 0° to 360° in the image. Training images in MSTAR cover a range of aspect angles but not uniformly. Random rotation augmentation synthetically covers all angles, teaching the model that the same vehicle at different orientations is still the same vehicle.

**Why 360° and not the typical ±30°?** Optical images of objects rarely appear fully upside down, so ±30° is standard for optical recognition. SAR images routinely appear at all orientations depending on the flight path — a full 360° range is correct for this domain.

### Speckle Noise (SpeckleNoise)

SAR speckle is **multiplicative** noise — it multiplies into the signal rather than adding to it. Standard Gaussian noise (additive) is the wrong model for SAR. The implementation uses Rayleigh-distributed multiplicative noise, which is the physically correct statistical model for SAR speckle.

By training with this noise injected, the model learns to recognize vehicles through the noise rather than memorizing noise-specific patterns.

---

## 9. Why a Separate Evaluation Module?

### The problem it solves

During training, it is enough to track overall accuracy. But for a research paper, you need:
- **Per-class precision and recall**: Did the model fail specifically on T62? On unseen-class detection?
- **F1-score**: The harmonic mean of precision and recall — a single number that captures both.
- **Confusion matrix**: A table showing which class was confused for which — reveals systematic errors.
- **Reproducible JSON reports**: So you can run multiple experiments and compare them programmatically.

The `evaluate.py` module handles all of this and saves a JSON file for every evaluation run. The `print_comparison_table` function loads multiple JSON files and prints a Markdown table — ready to paste into a paper.

---

## 10. How Everything Connects

```
                        MSTAR Dataset (10 classes, 100 images/class)
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
               Seen classes (9)                 Unseen class (T72)
                        │                               │
              80% train / 20% test            All images in open-set pool
                        │
          ┌─────────────┴──────────────┐
          │                            │
   Real images              GAN-generated images
   (mstar_sampled_data/)    (gan_output/)
          │                            │
          └──────────┬─────────────────┘
                     │
              Training Episode
              ┌──────────────────┐
              │ Support set:     │  ← C classes × K shots
              │ Query image:     │  ← 1 image (seen OR T72 unseen)
              └──────────────────┘
                     │
             ┌───────▼────────┐
             │  CNN (4 layers │  ← encodes each image to 64-D
             │  + SE attention│
             └───────┬────────┘
                     │   embeddings
             ┌───────▼────────┐
             │  GNN (3 adj +  │  ← reasons about similarity
             │  2 update)     │
             └───────┬────────┘
                     │   logits (C+1 classes)
             ┌───────▼────────┐
             │  Loss:         │
             │  NLL (class)   │
             │  + λ × Physics │  ← optional variance penalty
             └────────────────┘
                     │
              Backpropagation
              + Adam optimizer
                     │
              Best checkpoint
              (highest overall acc)
                     │
              Evaluation (5000 tasks)
              seen acc / unseen acc / overall acc
              per-class F1 / confusion matrix
              → results/report.json
```

---

## 11. Summary: Why Each Component Exists

| Component | What it does | Why we need it |
|-----------|-------------|----------------|
| **CNN (4-layer, SE attention)** | Converts raw SAR pixels → 64-D feature vector | Images must be compressed into a comparable form; SE attention focuses on informative channels in noisy SAR data |
| **GNN (3 adj + 2 update)** | Compares query against support examples using a learned graph | Few-shot comparison is more generalizable than memorization; graph structure captures cross-sample relationships |
| **Episodic training** | Trains on randomly generated C-way K-shot tasks | Forces the model to generalize — it never sees the same task twice |
| **Open-set output (C+1 classes)** | Adds an "unknown" slot to the classifier output | Detects novel vehicles not in training, which is the real-world requirement |
| **T72 as unseen class** | Held out entirely from training, used only as open-set test | Provides a controlled way to measure unknown-class detection |
| **Conditional DCGAN** | Generates synthetic SAR images per vehicle class | Data scarcity — 100 real images per class is not enough; synthetic data increases effective training set size |
| **Physics loss (variance penalty)** | Penalizes spread of same-class embeddings | Encodes the radar physics prior that same-class targets should produce consistent features |
| **Rotation augmentation (360°)** | Randomly rotates support images during training | SAR targets appear at all orientations; the model must be invariant to this |
| **Speckle noise augmentation** | Adds Rayleigh-distributed multiplicative noise | Trains robustness to the specific noise statistics of SAR imaging |
| **5000-task evaluation** | Runs 5000 test episodes before reporting accuracy | Enough tasks for statistically stable accuracy estimates (±2% variance) |
| **evaluate.py** | Computes precision, recall, F1, confusion matrix; saves JSON | Research requires systematic, reproducible metrics across all model variants |
| **TrainerBaseline (CNN only)** | Trains a standard classifier without few-shot learning | Establishes the performance floor — how bad things are without FSL |
