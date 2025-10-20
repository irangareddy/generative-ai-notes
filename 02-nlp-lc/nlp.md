# NLP

## **1. Foundations of Natural Language Processing (pp. 40–44)**

### **Definition**

> NLP (Natural Language Processing) is a subfield of AI that enables computers to understand, interpret, and generate human language in a meaningful way.

It blends **linguistics**, **computer science**, and **machine learning** to perform tasks like:

* Machine translation
* Sentiment analysis
* Summarization
* Chatbots
* Question answering

**Core idea:** Transform human language into structured data a machine can process.

### **Why NLP is Difficult**

Language understanding involves:

1. **Ambiguity** → “bank” = money bank or river bank
2. **Context dependence** → requires world knowledge
3. **Subtlety** → tone and sentiment can invert with small word changes
4. **Long-range dependencies** → “The student who forgot his homework was scolded” (subject far from verb)
5. **Representation of meaning** → machines need structured numerical encoding of semantics

### **Historical Evolution**

| Era              | Approach                 | Highlights                                           |
| --- | --- | --- |
| **1950s–70s**    | **Rule-based NLP**       | ELIZA chatbot; grammar-based logic                   |
| **1980s–90s**    | **Statistical NLP**      | HMMs, Maximum Entropy Models, corpus linguistics     |
| **2000s–2010s**  | **Machine Learning Era** | Naive Bayes, SVMs, feature-based text classification |
| **2013–Present** | **Deep Learning Era**    | Word2Vec, Transformers (BERT, GPT), LLMs             |

## **2. Mathematical & Statistical Foundations (pp. 45–54)**

Modern NLP relies heavily on **linear algebra, probability, and statistics** to represent words and learn language patterns.

### **Linear Algebra**

* **Scalars:** Single numeric value
* **Vectors:** Ordered list of numbers representing features (e.g., word frequency)
* **Matrices:** 2D arrays representing datasets or embeddings

  * Rows → documents
  * Columns → features
* **Dot Product:** Measures vector similarity
  [
  \text{similarity}(A,B) = A \cdot B = \sum_i A_i B_i
  ]
* **Cosine Similarity:**
  [
  \cos(\theta) = \frac{A \cdot B}{||A|| , ||B||}
  ]
  → 1 = identical, 0 = unrelated, -1 = opposite
  (Used in document similarity, word embeddings)

### **Eigenvalues and Eigenvectors**

* Capture principal directions of variance in data.
* Used in **PCA (Principal Component Analysis)** for dimensionality reduction.
* Eigenvalue = scaling factor; Eigenvector = direction preserved during transformation.

### **Probability & Random Variables**

* **PMF** (for discrete variables): probability of exact outcomes
* **PDF** (for continuous variables): probability density across intervals
* **Events:** can be independent, complementary, or mutually exclusive.

### **Maximum Likelihood Estimation (MLE)**

Used to estimate model parameters θ that make observed data most probable:
[
\hat{\theta} = \arg \max_\theta \prod_i P(x_i \mid \theta)
]
or equivalently,
[
\log L = \sum_i \log P(x_i \mid \theta)
]
because maximizing log-likelihood = maximizing likelihood.

Example: predicting the next word (“they”) in a sentence based on probabilities.

## **3. Text Preprocessing & Linguistic Analysis (pp. 55–64)**

**Goal:** Convert raw text → clean, tokenized, and structured data for modeling.

### **Text Normalization**

Common steps:

1. Lowercasing
2. Removing punctuation & special characters
3. Removing numbers
4. Correcting spelling
5. Removing stop words (“the”, “is”, “in”)
6. **Stemming:** “running”, “ran” → “run”
7. **Lemmatization:** “better” → “good”
8. **Tokenization:** splitting sentences into words/tokens

Example:

> “I can't believe it’s already 2025!!! #timeflies”
> → “i cant believe its already timeflies”

### **Linguistic Processing Tasks**

| Task                               | Description                                   | Example Tools             |
| - |  | - |
| **Named Entity Recognition (NER)** | Detect people, locations, organizations, etc. | spaCy, NLTK, Stanford NER |
| **POS Tagging**                    | Assign parts of speech (noun, verb, adj).     | spaCy, Flair              |
| **Parsing**                        | Build syntactic trees (phrase or dependency). | Regex, dependency parsing |
| **Syntactic Structures**           | Define word relationships (subject, object).  | Stanford Parser           |

### **Tokenization**

* **Word Tokenization:** splits text into words.
* **Sentence Tokenization:** splits text into sentences.
* **Subword Tokenization:** (used in BERT/GPT) splits rare words into smaller parts:
  “unhappiness” → “un” + “##happiness”
  → enables robust handling of unseen words.

## **4. Classical Text Representation Models (pp. 65–87)**

Before neural embeddings, text was represented statistically.

### **Bag of Words (BoW)**

* Represents documents as **word frequency vectors**.
* Ignores word order and grammar.
* Example:

  * “This movie is scary and long” → `[1,1,1,1,1,1]`
  * “This movie is not scary” → `[1,1,0,1,0,1]`

**Limitations:**

* Very sparse vectors
* High dimensionality
* Loses context (word order)

### **One-Hot Encoding**

* Each unique word = binary vector.
* Simple but does not capture meaning.
* Example:
  Vocab = [apple, banana, orange, grape]
  → “apple” = `[1,0,0,0]`

**Issue:** Large vocab → high-dimensional, sparse matrix.

### **N-Grams**

* Capture *local context* by grouping N consecutive words.
* Example:

  * Sentence: “The cat sat on the mat”
  * Bigrams: [“The cat”, “cat sat”, “sat on”, “on the”, “the mat”]
* Trade-off: better context but more features and sparsity.

### **TF–IDF (Term Frequency–Inverse Document Frequency)**

* Weighs words based on **importance** in a document vs. the entire corpus.
  $$[
  \text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}
  ]$$
  where:

  * TF = term frequency in document
  * DF = number of docs containing the term
  * N = total documents

**Example Workflow:**

1. Preprocess documents (remove stop words, lemmatize).
2. Build vocabulary.
3. Compute TF & IDF.
4. Create matrix.
5. Train classifier (e.g., Logistic Regression, SVM).

Used widely in **text classification**, **spam detection**, and **sentiment analysis**.

## **5. Word Embeddings & Deep Semantic Representation (pp. 68–105)**

Shift from **sparse statistical** → **dense semantic** representations.
Each word becomes a continuous vector in high-dimensional space capturing *meaning* and *context*.

### **What Are Word Embeddings?**

* Dense vectors learned from context using neural networks.
* Words with similar meanings have nearby vectors.
  Example:
  “king – man + woman ≈ queen”
* Enable clustering, analogy reasoning, and similarity tasks.

### **Common Embedding Models**

#### 1. **Word2Vec (Google, 2013)**

Two main architectures:

* **CBOW (Continuous Bag of Words):** predict target word from context.
* **Skip-Gram:** predict context words from a target word.

  * Input → Hidden layer → Output (softmax).
  * Learned embedding = hidden layer weights.

Example:

* Sentence: “The cat sits on the mat”
→ (“cat”, “sits”), (“cat”, “on”), (“cat”, “the”), (“cat”, “mat”)

#### 2. **GloVe (Global Vectors, Stanford)**

* Learns embeddings from **co-occurrence probabilities** between words.
* Ratio of co-occurrence reveals semantic relationships.
  Example:
  “ice” appears more with “solid”,
  “steam” appears more with “gas”.
  → Embeddings capture physical state relationships.

#### 3. **FastText (Facebook)**

* Improves on Word2Vec using **subword embeddings**.
* Handles out-of-vocabulary words by learning from word parts.
  Example: “unhappiness” → “un” + “happy” + “ness”.

#### 4. **Transformer-based Contextual Embeddings (BERT, GPT)**

* Represent the same word differently depending on context.
  Example:
  “bank” (river) ≠ “bank” (finance)
* Leverage **self-attention** to model long-range dependencies.

### **Embedding Applications**

* Semantic search & clustering
* Sentiment classification
* Information retrieval
* Transfer learning (pretrained embeddings applied to new tasks)

### **Visualization and Analogy Tasks**

* **t-SNE plots** show similar words clustering together.
* Vector arithmetic demonstrates relational reasoning:
  [
  \text{king} - \text{man} + \text{woman} = \text{queen}
  ]

### **Transfer Learning**

1. **Train from scratch** on large corpus (e.g., Google News).
2. **Use pre-trained embeddings** (Word2Vec, GloVe).
3. **Fine-tune** for domain-specific data (e.g., legal, medical).

## Summary

| Stage                            | Focus                                    | Outcome                                 |
| ----- | ----- | ------ |
| **1. NLP Foundations**           | What NLP is, challenges, evolution       | Understand human language complexity    |
| **2. Math & Probability**        | Linear algebra, cosine similarity, MLE   | Build intuition for model computations  |
| **3. Text Preprocessing**        | Cleaning, tokenization, tagging, parsing | Prepare raw text for modeling           |
| **4. Classical Representations** | BoW, TF–IDF, N-grams                     | Convert text → numerical features       |
| **5. Deep Embeddings**           | Word2Vec, GloVe, contextual models       | Capture meaning, semantics, and context |

### **In short:**

> The NLP module transitions from **statistical text processing → deep semantic representation**, bridging traditional linguistic methods with **modern transformer-based embeddings** — forming the conceptual backbone of all LLMs and language applications today.
