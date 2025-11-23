# Technical_Interview_Scienta_Lab
I have put all the files related to the task given in the technical interview in this Github repo. Further more I will try to make a presentation to better present what I did.

## 1\. Data  Pipeline

This project implements a custom, memory-efficient ETL (Extract, Transform, Load) pipeline designed to handle high-dimensional genomic data on limited-resource environments (e.g., Google Colab, Standard Consumer GPUs).

  * **Stream Processing & Chunking:**
      * Utilizes `pd.read_csv(chunksize=5000)` to process large genomic datasets iteratively, preventing RAM overflow (OOM errors).
      * Implements aggressive type-casting (converting `float64` $\to$ `float32` and labels to `int8`) to reduce memory footprint by \~50%.
  * **Feature Alignment ("The Ruler Method"):**
      * Solves the "Feature Mismatch" problem common in bioinformatics where training and testing sets have different gene counts.
      * Dynamically generates a zero-padded canvas based on the training set's feature space and overlays test data, ensuring consistent input dimensions ($N_{train\_features}$) for the model regardless of input file variance.
  * **Statistical Preprocessing:**
      * **Log-Normalization:** Applies `np.log1p` (Natural Logarithm of $1+x$) to handle the power-law distribution of gene expression counts and dampen the effect of outliers.
      * **Z-Score Scaling:** Standardizes features ($\mu=0, \sigma=1$) using `sklearn.StandardScaler` to ensure stable convergence during gradient descent.

<img width="1317" height="590" alt="image" src="https://github.com/user-attachments/assets/3839ea99-bd3b-47b6-bb25-a2f317f56949" />

## 2\. Model Architecture: GeneBERT

The core classifier is a Transformer-based architecture adapted for tabular biological data. Unlike standard ML models (RF/SVM), this architecture utilizes self-attention mechanisms to learn non-linear interactions within the latent gene embedding space.


### Architecture Breakdown:

1.  **Input Projection (Embedding Layer):**

      * **Input:** High-dimensional gene vector ($R^{N}$).
      * **Operation:** Linearly projects raw gene features into a dense, lower-dimensional latent space ($d_{model} = 128$).
      * **Regularization:** Applies `GELU` activation and `LayerNorm` for stabilization.

2.  **Transformer Encoder Block:**

      * **Structure:** Uses the PyTorch `TransformerEncoder` with **2 stacked layers**.
      * **Multi-Head Self-Attention:** Utilizes **4 attention heads** to capture different subspaces of gene-feature relationships simultaneously.
      * **Feed-Forward Network:** Expands the internal dimension to 512 ($4 \times d_{model}$) with a dropout rate of 0.2 to prevent overfitting.
      * **Context:** The model treats the entire gene profile sample as a single sequence token (`Seq_Len=1`), allowing the transformer to refine the sample representation via self-looping attention mechanisms.

3.  **Classification Head:**

      * **Pooling:** Global Average Pooling (reducing sequence dimension).
      * **MLP:** A Multi-Layer Perceptron (`Linear` $\to$ `ReLU` $\to$ `Linear`) mapping the refined 128-dim embedding to the 3 target classes (Healthy, RA, SLE).

## 3\. Hyperparameters & Configuration

  * **Optimizer:** AdamW (Weight Decay handling)
  * **Loss Function:** CrossEntropyLoss
  * **Learning Rate:** $1e^{-4}$
  * **Batch Size:** 32
  * **Embedding Dimension:** 128
  * **Encoder Layers:** 2
  * **Attention Heads:** 4

### 4\. Training and Inference
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/18713f7e-be0a-47a2-bc4e-939d775ec4c0" />

### 5\. Confusion Matrix
## Accuracy 41.28%

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/02094993-e864-4ea3-9384-3c9dd468b1e1" />




