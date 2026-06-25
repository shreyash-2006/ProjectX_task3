# Paper Notes: Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)

## 1. Central Claim
The paper claims that high-quality, continuous vector representations of words can be learned efficiently using simple, log-linear architectures (CBOW and Skip-gram). 

The critical insight is that removing the non-linear hidden layer from traditional neural language models drastically reduces computational complexity. This allows training on massive datasets in a reasonable time. The resulting vectors are remarkably good at preserving linear regularities, meaning you can perform algebra like `vec("King") - vec("Man") + vec("Woman") ≈ vec("Queen")`. 

## 2. Core Architecture / Algorithm (for Skip-gram)
The paper proposes two novel architectures, focusing mainly on the **Skip-gram** model for our use:

- **Architecture**: It is a log-linear classifier. Given a center word, it predicts the surrounding context words within a fixed window. 
- **Optimization**: To make training on millions of words feasible, they use **Hierarchical Softmax** (a Huffman tree over the vocabulary). While the original paper uses this, our implementation uses **Negative Sampling** (which was introduced in their follow-up NIPS 2013 paper). Negative Sampling is a widely accepted alternative that simplifies the code and provides similar quality.
- **Key Hyperparameters**: They use a context window of up to 10 words, vector dimensionality ranging from 300 to 1000, and train for 3-5 epochs with SGD and a linearly decreasing learning rate (starting at 0.025).

## 3. Dataset, Evaluation, and Baselines
- **Training Data**: The paper reports results on several corpora, but the best results come from the Google News corpus (6 billion words). Our implementation uses the `text8` dataset (~17 million words) due to size constraints. 
- **Evaluation Metric**: They created a comprehensive **Semantic-Syntactic Word Relationship test set** (often called the "Google Analogy test set"). It contains ~8,869 semantic questions (e.g., capital cities, currency) and ~10,675 syntactic questions (e.g., comparative adjectives, plural nouns). 
  - The metric is **accuracy**: a question is correct only if the exact target word is retrieved via `vec(b) - vec(a) + vec(c)` (using cosine distance).
- **Baselines**: They compare against:
  - Feedforward NNLM (Bengio 2003)
  - Recurrent NNLM (Mikolov 2010)
  - Previously published public vectors (Collobert, Turian, etc.)
- **Results in Paper**: The Skip-gram model achieves **~55% semantic** and **~59% syntactic** accuracy on their full test set using 300-dim vectors trained on 783M words. On the full 6B words, it hits ~65% total accuracy.

**Context for our implementation**: Because we are limited to the ~17-million-word `text8` corpus, our accuracy will naturally be significantly lower (expected 25-35%). The model simply sees fewer examples of rare entities like "Tallahassee" or "Berlusconi", making the semantic analogies particularly difficult to learn.