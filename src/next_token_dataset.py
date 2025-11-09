import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class NextTokenDataset(Dataset):
    def __init__(self, file_path, sequence_length=50):
        self.sequence_length = sequence_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]
        
        self._build_vocab()
    
    def _build_vocab(self):
        words = set()
        for text in self.texts:
            words.update(text.split())
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for word in words:
            self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()
        
        # Берем последовательность фиксированной длины
        if len(tokens) > self.sequence_length:
            tokens = tokens[:self.sequence_length]
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        # Явно указываем тип long
        input_ids = torch.tensor([self.word_to_idx.get(token, 1) for token in input_tokens], dtype=torch.long)
        target_ids = torch.tensor([self.word_to_idx.get(token, 1) for token in target_tokens], dtype=torch.long)
        
        return input_ids, target_ids

def collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # Паддинг до максимальной длины в батче
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded

def create_data_loaders():
    train_dataset = NextTokenDataset('./data/train_cleaned.txt')
    val_dataset = NextTokenDataset('./data/val_cleaned.txt')
    test_dataset = NextTokenDataset('./data/test_cleaned.txt')
    
    vocab = {
        'word_to_idx': train_dataset.word_to_idx,
        'idx_to_word': train_dataset.idx_to_word,
        'vocab_size': train_dataset.vocab_size
    }
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, vocab