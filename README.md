# News Recommendation System (NRMS on MIND-small)

## Project Overview

This project implements an end-to-end news recommendation system using the MIND (Microsoft News Dataset). The objective is to predict which news articles a user is most likely to click based on their reading history and a set of candidate articles. The system follows the complete machine learning pipeline from data preprocessing to model evaluation. News recommendation is challenging due to: Rapidly changing user interests, Short article lifespans, and Cold-start problems.

---

## Learning Objectives

This project demonstrates the ability to:
- Work with real-world behavioral datasets  
- Apply NLP techniques (tokenization, embeddings)  
- Build deep learning models using PyTorch  
- Implement attention-based architectures  
- Evaluate ranking systems using AUC, MRR, and nDCG  
- Analyze model performance and limitations  

---

## Dataset

We use the MIND-small dataset, which contains:
- ~50,000 users  
- ~65,000 news articles  
- ~230,000 impressions  

Each impression contains:
- User history (clicked articles)
- Candidate articles with labels (clicked / not clicked)

---

## Project Phases

### Phase 1 — Environment Setup & Data Acquisition
- Downloaded MIND-small dataset  
- Downloaded GloVe 300d embeddings  
- Organized directory structure  

---

### Phase 2 — Exploratory Data Analysis (EDA)
- Analyzed user activity and click distribution  
- Studied category distribution and behavior patterns  
- Identified sparsity and imbalance  

---

### Phase 3 — Data Preprocessing & Feature Engineering

- Tokenized news titles  
- Built vocabulary with frequency filtering  
- Constructed GloVe embedding matrix  
- Encoded titles to fixed length (30 tokens)  
- Limited user history to last 50 articles  
- Parsed impressions into positive and negative samples  

**Negative Sampling:**
- 1 positive + 4 negatives per sample  

---

### Phase 4 — Model Architecture (NRMS)

#### News Encoder
- Word embeddings (GloVe)  
- Multi-head self-attention  
- Additive attention  

#### User Encoder
- Encodes clicked news history  
- Multi-head self-attention  
- Additive attention  

#### Prediction
- Dot product between user and candidate vectors  
- Produces ranking scores  

---

### Phase 5 — Training

- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 64  
- Epochs: 5  
- Gradient clipping applied  

---

### Phase 6 — Evaluation

Metrics used:
- AUC  
- MRR  
- nDCG@5  
- nDCG@10  

---

## Results & Evaluation

### Performance Across Epochs
<img width="675" height="202" alt="Screenshot 2026-04-20 at 10 38 31 AM" src="https://github.com/user-attachments/assets/c0e39c57-c063-4904-9fd9-7a5cd78f7716" />


|Epoch | AUC    | MRR    | nDCG@5 | Loss   |
|------|--------|--------|--------|--------|
| 1    | 0.6295 | 0.5559 | 0.6660 | 1.4825 |
| 2    | 0.6394 | 0.5673 | 0.6746 | 1.4126 |
| 3    | 0.6487 | 0.5754 | 0.6808 | 1.3804 |
| 4    | 0.6552 | 0.5834 | 0.6868 | 1.3579 |
| 5    | 0.6562 | 0.5842 | 0.6875 | 1.3420 |

---

### Evaluation Interpretation
The model demonstrates consistent improvement across all epochs, indicating stable and effective training.

Key observations: 
- AUC increases steadily, reaching ~0.65, which aligns with expected baseline performance for NRMS on MIND-small
- MRR and nDCG show strong ranking performance, meaning the model is successfully placing clicked articles near the top
- Loss decreases monotonically, confirming stable optimization.
- Performance gains begin to plateau after Epoch 4, suggesting that the model has converged.

---

### Important Evaluation Note

Evaluation was performed using: 1 positive + 4 sampled negatives (5 total candidates). This differs from the official MIND benchmark, which uses full candidate lists.

Implications:
- Easier ranking task  
- Higher MRR and nDCG values  

Therefore, results are **not directly comparable** to benchmark scores.

---

### Comparison with Expected Baseline

| Metric | Expected NRMS | This Model |
|--------|---------------|------------|
| AUC    | 0.62 – 0.66   | **0.6562** |
| MRR    | 0.28 – 0.31   | **0.5842** |
| nDCG@5 | 0.30 – 0.34   | **0.6875** |

---

## Error Analysis

### 1. Simplified Candidate Sampling
- Only 5 candidates per impression  
- Leads to inflated evaluation metrics
- 
### 2. Limited Text Features
- Only titles used  
- Abstracts ignored
  
### 3. Cold-Start Problem
- Users with limited history provide weak signals  

### 4. Lack of Hard Negatives
- Random negatives may be too easy  

---

## Future Work
In the future, applying the techniques below will provide better results: 

### 1. Full Candidate Evaluation
Use complete impression lists for realistic evaluation  

### 2. Use News Abstracts
Improve semantic understanding  

### 3. Hard Negative Sampling
Select more informative negatives  

### 4. Transformer Embeddings
Replace GloVe with BERT/DistilBERT  

### 5. Category-Aware Features
Include category and subcategory embeddings  

### 6. Cold-Start Solutions
Handle users with little interaction history  

### 7. Hyperparameter Tuning
Experiment with:
- Attention heads  
- History length  
- Dropout  

---

## How to Run

### Option 1 — Quick Start (Recommended)

```bash
git clone https://github.com/manandhar-shairu/mind-recommender.git
cd mind-recommender
pip install torch numpy pandas scikit-learn tqdm
Then run: notebooks/3_training.ipynb

### Option 2 — Full Pipeline (From scratch)
This section assumes the user has **nothing except the cloned repository**.


### 1. Clone the Repository

```bash
git clone https://github.com/manandhar-shairu/mind-recommender.git
cd mind-recommender

### 2. Install Dependencies
```bash
pip install torch numpy pandas scikit-learn tqdm matplotlib

Optional for GPU
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118

### 3. Download Required Data

You must manually download:
MIND-small Dataset from: https://msnews.github.io/

GloVe Embeddings (300d) from: https://nlp.stanford.edu/projects/glove/
Use glove.6B.300d.txt

### 4. Project Structure 
mind-recommender/
│
├── data/
│   ├── MINDsmall_train/
│   ├── MINDsmall_dev/
│   └── glove/
│
├── models/              # saved checkpoints (auto-created)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
│
├── results/             # plots + processed outputs
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   ├── news_encoder.py
│   ├── user_encoder.py
│   └── train.py
│
└── README.md

### 5. Run the Pipeline
You must run notebooks in order because each step generates data for the next:
1. notebooks/01_eda.ipynb
2. notebooks/02_preprocessing.ipynb
3. notebooks/03_training.ipynb

### 6. Evaluation
Evaluation runs automatically after each epoch using: src/evaluate.py
