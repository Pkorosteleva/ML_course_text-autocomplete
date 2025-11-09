import torch
import torch.nn as nn
import torch.optim as optim
import time
from src.rouge_metrics import calculate_rouge

def evaluate_rouge(model, val_loader, vocab, num_samples=100):
    """Быстрая оценка ROUGE на ограниченном количестве примеров"""
    model.eval()
    rouge1_scores = []
    rouge2_scores = []
    
    # Получаем устройство модели
    device = next(model.parameters()).device
    
    samples_evaluated = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            # Переносим данные на устройство модели
            data = data.long().to(device)
            targets = targets.long().to(device)
            
            for i in range(data.size(0)):
                if samples_evaluated >= num_samples:
                    break
                    
                # Получаем входной текст
                input_indices = data[i]
                input_text = " ".join([vocab['idx_to_word'].get(idx.item(), '<UNK>') 
                                     for idx in input_indices if idx.item() != 0])
                
                # Получаем целевой текст (следующие слова)
                target_indices = targets[i]
                target_text = " ".join([vocab['idx_to_word'].get(idx.item(), '<UNK>') 
                                      for idx in target_indices if idx.item() != 0])
                
                if input_text and target_text:
                    # Генерируем продолжение
                    generated = model.predict_next_tokens(input_text, vocab, num_tokens=len(target_text.split()))
                    
                    # Вычисляем ROUGE
                    rouge_scores = calculate_rouge(generated, input_text + " " + target_text)
                    rouge1_scores.append(rouge_scores['rouge1']['f1'])
                    rouge2_scores.append(rouge_scores['rouge2']['f1'])
                    
                    samples_evaluated += 1
            
            if samples_evaluated >= num_samples:
                break
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    return avg_rouge1, avg_rouge2

def print_gpu_memory():
    if torch.cuda.is_available():
        # Получаем текущее устройство
        device = torch.cuda.current_device()
        
        # Полная информация о памяти
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - reserved
        
        print(f"GPU Memory - Total: {total:.2f} GB")
        print(f"             Free: {free:.2f} GB") 
        print(f"             Allocated: {allocated:.2f} GB")
        print(f"             Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available")

def train_model():
    from src.next_token_dataset import create_data_loaders
    from src.lstm_model import LSTMLanguageModel

    # Очистка памяти перед началом
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Уменьшите batch size в create_data_loaders
    train_loader, val_loader, test_loader, vocab = create_data_loaders()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMLanguageModel(vocab_size=vocab['vocab_size']).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    num_epochs = 5
    
    print("Начинаем обучение...")
    print(f"Размер словаря: {vocab['vocab_size']}")
    print(f"Количество батчей: {len(train_loader)}")
    print(f"Устройство: {device}")
    print("-" * 50)
    
    # Gradient Accumulation для сохранения эффективного batch size
    accumulation_steps = 4  # эффективный batch = 32 * 4 = 128
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        optimizer.zero_grad()  # обнуляем градиенты в начале эпохи
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.long().to(device)
            targets = targets.long().to(device)
            
            output = model(data)
            loss = criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))
            
            # Нормализуем loss для accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # возвращаем к оригинальному масштабу
            
            if batch_idx % 50 == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                print(f"Эпоха {epoch+1}/{num_epochs} | Батч {batch_idx}/{len(train_loader)} | Loss: {avg_loss_so_far:.4f}")
                
                # Периодически показываем использование памяти
                if batch_idx % 200 == 0:
                    print_gpu_memory()
        
        # Не забудьте сделать step для оставшихся градиентов
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Статистика после эпохи
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        # Очищаем кэш после каждой эпохи
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Вычисляем ROUGE только на последней эпохе для экономии времени
        if epoch == num_epochs - 1:
            print("Вычисляем ROUGE метрики на 100 примерах...")
            rouge1, rouge2 = evaluate_rouge(model, val_loader, vocab, num_samples=100)
            print(f"Эпоха {epoch+1} завершена:")
            print(f"  Средний Loss: {avg_loss:.4f}")
            print(f"  ROUGE-1 F1: {rouge1:.4f}")
            print(f"  ROUGE-2 F1: {rouge2:.4f}")
            print(f"  Время эпохи: {epoch_time:.2f} сек")
        else:
            print(f"Эпоха {epoch+1} завершена:")
            print(f"  Средний Loss: {avg_loss:.4f}")
            print(f"  Время эпохи: {epoch_time:.2f} сек")
        
        print("-" * 30)
    
    torch.save(model.state_dict(), './models/lstm_model_bs32_accum4.pth')
    print("Модель сохранена")
    
    # Примеры предсказаний
    print("\nПримеры предсказаний обученной модели:")
    examples = ["i love", "the weather is", "i want to"]
    for example in examples:
        prediction = model.predict_next_tokens(example, vocab)
        print(f"  '{example}' -> '{prediction}'")

if __name__ == "__main__":
    train_model()