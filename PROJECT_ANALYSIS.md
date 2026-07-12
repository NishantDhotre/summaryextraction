# Project Analysis: Text Summary Extraction Streamlit App

This document provides a comprehensive breakdown of the repository's architecture, directory structure, individual code components, and technical details of the three summary extraction techniques implemented in this project.

---

## 1. Project Overview
This repository contains a **Streamlit** web application for extractive and generative text summarization. Instead of using high-level machine learning libraries for summarization (like HuggingFace transformers), the project features manual implementations of classical natural language processing (NLP) and graph-based ranking algorithms.

The application allows users to either type/paste plain text or upload a `.txt` file, choose a summarization methodology, tune hyperparameters (such as number of clusters or top-$n$ sentences), and run the summarization pipeline to see pre-processed outputs and the generated summaries.

---

## 2. Directory Structure
The repository contains the following files and folders:

```
NishantDhotre_summaryextraction/
├── .devcontainer/
│   └── devcontainer.json        # Containerized development configuration for VS Code.
├── pages/
│   ├── MMR.py                    # Page implementing Maximal Marginal Relevance (MMR).
│   ├── TF-IDF.py                 # Page implementing standard TextRank using TF-IDF & Cosine Similarity.
│   └── k_means.py                # Page implementing K-Means Clustering combined with a Bigraph generator.
├── Hello.py                      # Main entrypoint of the Streamlit application.
├── README.md                     # Boilerplate Streamlit markdown overview file.
├── __init__.py                   # Standard package initialization file with Snowflake/Streamlit copyright headers.
├── nltk.txt                      # NLTK download requirements (stopwords, punkt, wordnet).
├── requirements.txt              # Project dependencies.
└── utils.py                      # Shared helper utility functions (e.g., showing code sidebars).
```

---

## 3. Detailed Component & Algorithm Breakdown

### A. Preprocessing Pipeline (`preprocessing` function)
All three pages implement a custom text preprocessing pipeline instead of using high-level libraries. This function resides locally in each page script and performs the following tasks sequentially:
1. **Punctuation Removal**: Filters out characters listed in `!()-[]{};:'"\,<>./?@#$%^&*_~`.
2. **Lowercasing**: Converts the text to lowercase.
3. **Tokenization**: Uses `nltk.word_tokenize` to split sentences into individual words.
4. **Stopwords Removal**: Removes common English stop words using `nltk.corpus.stopwords.words('english')`.
5. **Lemmatization**: Uses `WordNetLemmatizer` to reduce words to their base dictionary form (lemma).

### B. Custom TF-IDF Vectorizer (`matrix_calculation` function)
Instead of using `scikit-learn`'s `TfidfVectorizer`, the project implements a manual TF-IDF generator:
1. **Term Frequency (TF)**: Computes a `Counter` dictionary of terms for each sentence.
2. **Inverse Document Frequency (IDF)**: Iterates over the set of unique words in the text and computes IDF for each word as:
   $$\text{IDF}(w) = \ln\left(\frac{N}{1 + \text{count}(w)}\right)$$
   where $N$ is the total number of sentences and $\text{count}(w)$ is the number of sentences containing word $w$.
3. **TF-IDF Matrix Generation**: Multiplies individual term frequencies by the respective IDFs, aligning each sentence vector with a globally sorted list of unique vocabulary words to build a final 2D TF-IDF matrix.

---

### C. Summarization Methods

#### 1. PageRank & TF-IDF (TextRank) (`pages/TF-IDF.py`)
This page extracts the top-$n$ sentences based on a classic TextRank formulation:
- **Graph Construction**: Constructs a NetworkX graph $G$ where each sentence index is a node.
- **Edge Weighting**: Computes the Cosine Similarity between the TF-IDF vectors of every pair of sentences:
  $$\text{Similarity}(S_i, S_j) = \frac{S_i \cdot S_j}{\|S_i\| \|S_j\|}$$
  and sets this similarity value as the undirected edge weight.
- **Ranking**: Runs the PageRank algorithm on $G$ using a damping factor $\alpha = 0.9$.
- **Extraction**: The sentences with the highest PageRank values are extracted, sorted chronologically to preserve chronological readability, and rendered as the final summary.

#### 2. Maximal Marginal Relevance (MMR) (`pages/MMR.py`)
This page implements Maximal Marginal Relevance (MMR) to reduce redundancy and increase summary diversity:
- **Graph & PageRank**: First, it computes the PageRank scores of sentences as done in standard TF-IDF TextRank.
- **MMR Selection**: Iteratively selects sentences from the unselected pool ($rBs$) to add to the summary. In each iteration, it seeks to maximize:
  $$\text{MMR}(S_i) = \lambda \cdot \text{PageRank}(S_i) - (1 - \lambda) \cdot \max_{S_j \in \text{Selected}} \text{CosineSimilarity}(S_i, S_j)$$
  where $\lambda$ (hardcoded as `0.7` in the code) controls the trade-off between relevance and diversity.
- **Code Bug Warning**:
  There is a minor logical issue in [pages/MMR.py](file:///d:/Job_hunt/Github%20Agent/.git_clones/NishantDhotre_summaryextraction/pages/MMR.py#L90-L109). Inside the selection loop:
  ```python
  new_ranks[i] = (lamB * ranks[i]) - (1 - lamB) * temp
  sorted_new_rank = sorted(new_ranks, key=ranks.get, reverse=True)[:1]
  ```
  The variable `sorted_new_rank` is sorted using `key=ranks.get` (which points to the original PageRank scores) instead of sorting by `new_ranks.get` (the actual calculated MMR score). Consequently, the loop always selects the next highest PageRank sentence rather than the MMR-optimized sentence.

#### 3. K-Means Clustering & Bigram Graph Generator (`pages/k_means.py`)
This page implements a hybrid extractive-generative approach:
- **K-Means Clustering**:
  - Automatically initializes $K$ centroids using `random.sample()` from the TF-IDF sentence matrix.
  - Runs custom assignment and update steps based on Cosine Similarity distance.
  - Updates centroids to the mean vector of sentence coordinates in each cluster.
  - Continues until centroids converge (`np.allclose`).
- **Cluster Representative Selection ($S_1$ and $S_2$)**:
  - For each cluster, the sentence closest to the centroid is selected as $S_1$.
  - Another sentence in the cluster, $S_2$, is selected if it shares at least 3 bigrams in common with $S_1$.
- **Directed Bigram Graph**:
  - Builds a word-bigram adjacency graph where nodes are bigrams of $S_1$ and $S_2$.
  - Adds special `"start"` and `"end"` nodes mapping to the sentence beginnings and ends.
- **Text Generation (Walk)**:
  - Generates a summary sentence for the cluster by executing a random walk from `"start"` to `"end"` along the bigram graph edges.

---

## 4. Dependencies and Infrastructure

- **requirements.txt**:
  - `streamlit`: Serves the web-based interactive GUI.
  - `altair`, `pandas`, `pydeck`: Standard dependencies bundled with Streamlit template structures.
  - `numpy`, `scipy`: Used for linear algebra (norm, dot product, centroids calculation).
  - `networkx>=3.1`: Powering graph creation and PageRank calculations.
  - `nltk`: Powering tokenization, lemmatization, and stopword retrieval.
- **nltk.txt**:
  - Instructs the host/deployment environment to download the `stopwords`, `punkt`, and `wordnet` corpora upon setup.
- **.devcontainer/devcontainer.json**:
  - Standard development container configured for Python 3.9, which pre-installs requirements and exposes port 8501 (the default Streamlit port).

---

## 5. Execution Instructions
To run this application locally, execute the following commands in your terminal:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Streamlit**:
   ```bash
   streamlit run Hello.py
   ```
3. Open your web browser and navigate to the address shown in the terminal (typically `http://localhost:8501`).
