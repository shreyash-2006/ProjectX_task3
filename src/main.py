import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
import urllib.request
import zipfile
import os
import pickle
import sys
# Download and prepare the text8 dataset
def download_text8(data_url='http://mattmahoney.net/dc/text8.zip', dataset='text8.zip'):
    if not os.path.exists(dataset):
        print(f"Downloading {dataset}...")
        urllib.request.urlretrieve(data_url, dataset)
    if not os.path.exists('text8'):
        print("Extracting...")
        with zipfile.ZipFile(dataset) as z:
            z.extractall()
    print("Dataset ready.")

def read_data(file_path='text8'):
    with open(file_path, 'r') as f:
        words = f.read().split()
    print(f"Total words in corpus: {len(words)}")
    return words

def build_vocab(words, vocab_size=50000):
    word_counts = Counter(words)
    most_common = word_counts.most_common(vocab_size - 1)
    word_to_ix = {'<UNK>': 0}
    ix_to_word = ['<UNK>']
    for word, _ in most_common:
        word_to_ix[word] = len(ix_to_word)
        ix_to_word.append(word)
    
    data = []
    unk_count = 0
    for word in words:
        if word in word_to_ix:
            data.append(word_to_ix[word])
        else:
            data.append(0)
            unk_count += 1
    
    word_counts['<UNK>'] = unk_count
    word_freqs = np.array([word_counts.get(word, 0) for word in ix_to_word], dtype=np.float32)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of <UNK> tokens: {unk_count}")
    return data, word_to_ix, ix_to_word, word_freqs


# Dataset & DataLoader with Negative Sampling

class SkipGramDataset(Dataset):
    def __init__(self, data, word_freqs, window_size=10, num_negatives=10):
        self.data = data
        self.word_freqs = word_freqs
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.vocab_size = len(word_freqs)
        self.neg_dist = word_freqs ** 0.75
        self.neg_dist /= self.neg_dist.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_word = self.data[idx]
        start = max(0, idx - self.window_size)
        end = min(len(self.data), idx + self.window_size + 1)
        context_words = [self.data[i] for i in range(start, end) if i != idx]
        if not context_words:
            return torch.tensor(center_word, dtype=torch.long), torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)
        
        neg_words = np.random.choice(self.vocab_size, self.num_negatives, p=self.neg_dist)
        return (torch.tensor(center_word, dtype=torch.long),
                torch.tensor(context_words, dtype=torch.long),
                torch.tensor(neg_words, dtype=torch.long))

def collate_fn(batch):
    centers, positives, negatives = zip(*batch)
    all_positives = [pos for pos_list in positives for pos in pos_list]
    flat_centers = [center for center, pos_list in zip(centers, positives) for _ in pos_list]
    neg_centers = [center for center, neg_list in zip(centers, negatives) for _ in neg_list]
    all_negatives = [neg for neg_list in negatives for neg in neg_list]
    return (torch.tensor(flat_centers, dtype=torch.long),
            torch.tensor(all_positives, dtype=torch.long),
            torch.tensor(neg_centers, dtype=torch.long),
            torch.tensor(all_negatives, dtype=torch.long))


# Skip-gram Model Definition

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super(SkipGramModel, self).__init__()
        self.embedding_center = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_context = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_pos, pos_words, center_neg, neg_words):
        center_embeds_pos = self.embedding_center(center_pos)
        pos_embeds = self.embedding_context(pos_words)
        positive_score = torch.sum(center_embeds_pos * pos_embeds, dim=1)

        center_embeds_neg = self.embedding_center(center_neg)
        neg_embeds = self.embedding_context(neg_words)
        negative_score = torch.sum(center_embeds_neg * neg_embeds, dim=1)

        return positive_score, negative_score


# Training Function

def train(model, dataloader, epochs=2, lr=0.025, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (center_pos, pos_words, center_neg, neg_words) in enumerate(dataloader):
            center_pos = center_pos.to(device)
            pos_words = pos_words.to(device)
            center_neg = center_neg.to(device)
            neg_words = neg_words.to(device)

            pos_score, neg_score = model(center_pos, pos_words, center_neg, neg_words)
            
            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            
            loss = criterion(pos_score, pos_labels) + criterion(neg_score, neg_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 1000 == 0:
                print(f'Epoch {epoch}, Step {i:05d}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')


# Evaluation

def load_embeddings_from_model(model, word_to_ix, ix_to_word, use_context=False):
    """Extract word vectors from trained model as dict."""
    model.eval()
    if use_context:
        emb_matrix = model.embedding_context.weight.data.cpu().numpy()
    else:
        emb_matrix = model.embedding_center.weight.data.cpu().numpy()
    
    word_vectors = {}
    for idx, word in enumerate(ix_to_word):
        word_vectors[word] = emb_matrix[idx]
    return word_vectors

def evaluate_analogies(word_vectors, analogy_file, verbose=True):
    """
    Evaluate on semantic-syntactic word relationship test set.
    Returns: (total_accuracy, semantic_accuracy, syntactic_accuracy,
              semantic_correct, semantic_total, syntactic_correct, syntactic_total)
    """
    # Prepare normalized vectors
    words = list(word_vectors.keys())
    vecs = np.array([word_vectors[w] / (np.linalg.norm(word_vectors[w]) + 1e-8) for w in words])
    
    def find_closest(vec, exclude_set):
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        sims = np.dot(vecs, vec_norm)
        best_idx = np.argsort(sims)[::-1]
        for idx in best_idx:
            cand_word = words[idx]
            if cand_word not in exclude_set:
                return cand_word, sims[idx]
        return None, 0.0

    semantic_correct = 0
    semantic_total = 0
    syntactic_correct = 0
    syntactic_total = 0
    current_category = None

    with open(analogy_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(':'):
                current_category = line[1:].strip().lower()
                continue
            
            parts = line.split()
            if len(parts) != 4:
                continue
            a, b, c, expected = parts
            
            if not all(w in word_vectors for w in [a, b, c, expected]):
                continue
            
            vec_a = word_vectors[a]
            vec_b = word_vectors[b]
            vec_c = word_vectors[c]
            target = vec_b - vec_a + vec_c
            
            closest, _ = find_closest(target, exclude_set={a, b, c})
            is_correct = (closest == expected)
            
            # Determine if syntactic (categories often contain 'gram' or specific keywords)
            if 'gram' in current_category or 'syntactic' in current_category or \
               current_category in ['capital-common-countries', 'capital-world', 'currency', 'city-in-state', 'family'] is False:
                # Actually, semantic categories in the original set: capital-common-countries, capital-world, currency, city-in-state, family
                # We'll use a simple rule: if category contains 'gram' it's syntactic, else semantic
                if 'gram' in current_category or 'suffix' in current_category or 'plural' in current_category or 'comparative' in current_category:
                    syntactic_total += 1
                    if is_correct:
                        syntactic_correct += 1
                else:
                    semantic_total += 1
                    if is_correct:
                        semantic_correct += 1
            else:
                # Default semantic
                semantic_total += 1
                if is_correct:
                    semantic_correct += 1
            
            if verbose and not is_correct:
                print(f"Failed: {a} : {b} :: {c} : ? -> guessed {closest}, expected {expected}")
    
    total_correct = semantic_correct + syntactic_correct
    total_questions = semantic_total + syntactic_total
    total_acc = total_correct / total_questions * 100 if total_questions > 0 else 0
    semantic_acc = semantic_correct / semantic_total * 100 if semantic_total > 0 else 0
    syntactic_acc = syntactic_correct / syntactic_total * 100 if syntactic_total > 0 else 0
    
    return (total_acc, semantic_acc, syntactic_acc,
            semantic_correct, semantic_total,
            syntactic_correct, syntactic_total)

# Main Execution
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 50000
    EMBEDDING_DIM = 128
    WINDOW_SIZE = 10
    NUM_NEGATIVES = 10
    BATCH_SIZE = 512
    EPOCHS = 2
    LEARNING_RATE = 0.025
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare data
    download_text8()
    words = read_data()
    data, word_to_ix, ix_to_word, word_freqs = build_vocab(words, vocab_size=VOCAB_SIZE)
    
    # Dataset and DataLoader
    dataset = SkipGramDataset(data, word_freqs, window_size=WINDOW_SIZE, num_negatives=NUM_NEGATIVES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # Model
    model = SkipGramModel(VOCAB_SIZE, EMBEDDING_DIM)
    print(f"Model initialized. Training on {DEVICE}.")
    
    # Train
    train(model, dataloader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # Save model and vocabulary
    torch.save(model.state_dict(), 'skipgram_text8.pt')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump((word_to_ix, ix_to_word), f)
    print("Model and vocabulary saved.")
    
    # Evaluate on analogy test set
    analogy_url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
    analogy_file = "questions-words.txt"
    if not os.path.exists(analogy_file):
        print("Downloading analogy test set")
        urllib.request.urlretrieve(analogy_url, analogy_file)
        print("Downloaded")
    
    # Load word vectors from trained model
    word_vectors = load_embeddings_from_model(model, word_to_ix, ix_to_word, use_context=False)
    print("\nEvaluating on semantic-syntactic word relationship test set...")
    total_acc, sem_acc, syn_acc, sem_corr, sem_tot, syn_corr, syn_tot = evaluate_analogies(
        word_vectors, analogy_file, verbose=False
    )
    
    print("\nEvaluation Results")
    print(f"Total accuracy: {total_acc:.2f}% ({sem_corr+syn_corr}/{sem_tot+syn_tot})")
    print(f"Semantic accuracy: {sem_acc:.2f}% ({sem_corr}/{sem_tot})")
    print(f"Syntactic accuracy: {syn_acc:.2f}% ({syn_corr}/{syn_tot})")