# Word2Vec Skip-gram Implementation (PyTorch)

The training pipeline uses Negative Sampling for efficient training on the `text8` corpus and evaluates the learned vectors using the official semantic-syntactic word relationship test set (Google Analogy dataset).

## Dependencies

- Python 3.7+
- PyTorch >= 1.0
- NumPy

To install the required packages, run:
```bash
pip install torch numpy
```
To install start training run:
```bash
python main.py
```
## Code Flow
The script will automatically:
Download the text8 dataset (100MB) and extract it.
Build a vocabulary of the 50,000 most frequent words.
Train the Skip-gram model using Negative Sampling (2 epochs, 128-dimensional vectors).
Save the trained model weights (skipgram_text8.pt) and vocabulary mappings (vocab.pkl).
Download the questions-words.txt test set (Mikolov's analogy dataset).
Evaluate the model on the full test set and print the Total, Semantic, and Syntactic accuracies.

## Reasons for low accuracy
In the original 2013 paper, Skip-gram achieves ~53% total accuracy, but that model was trained on 783 million words (the Google News corpus). Our implementation uses the text8 dataset, which is 17 million words (roughly 50x smaller). Word2vec relies heavily on co-occurrence statistics; with text8, the model simply hasn't seen enough examples of rare entities (like world cities, specific currencies, or celebrity names) to solve the semantic analogies perfectly.

If you want to see numbers closer to the paper, you would need to swap text8 for a larger corpus.

## Code Strcuture
Data loading & preprocessing (download_text8, build_vocab)
Dataset class with Negative Sampling (SkipGramDataset)
Model definition (SkipGramModel)
Training loop (train)
Evaluation (evaluate_analogies)
Main execution block