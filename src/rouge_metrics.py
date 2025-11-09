from collections import Counter

def rouge_1(candidate, reference):
    """ROUGE-1: совпадение униграмм (отдельных слов)"""
    candidate_words = candidate.split()
    reference_words = reference.split()
    
    candidate_count = Counter(candidate_words)
    reference_count = Counter(reference_words)
    
    # Количество совпадающих слов
    overlap = sum((candidate_count & reference_count).values())
    
    # Precision, Recall, F1
    precision = overlap / len(candidate_words) if candidate_words else 0
    recall = overlap / len(reference_words) if reference_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def rouge_2(candidate, reference):
    """ROUGE-2: совпадение биграмм (пар слов)"""
    candidate_words = candidate.split()
    reference_words = reference.split()
    
    # Создаем биграммы
    candidate_bigrams = [tuple(candidate_words[i:i+2]) for i in range(len(candidate_words)-1)]
    reference_bigrams = [tuple(reference_words[i:i+2]) for i in range(len(reference_words)-1)]
    
    candidate_count = Counter(candidate_bigrams)
    reference_count = Counter(reference_bigrams)
    
    # Количество совпадающих биграмм
    overlap = sum((candidate_count & reference_count).values())
    
    # Precision, Recall, F1
    precision = overlap / len(candidate_bigrams) if candidate_bigrams else 0
    recall = overlap / len(reference_bigrams) if reference_bigrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_rouge(generated_text, reference_text):
    """Вычисляет ROUGE-1 и ROUGE-2 для пары текстов"""
    return {
        'rouge1': rouge_1(generated_text, reference_text),
        'rouge2': rouge_2(generated_text, reference_text)
    }