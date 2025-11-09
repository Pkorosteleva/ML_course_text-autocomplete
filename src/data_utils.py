import os
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Очистка текста: нижний регистр, удаление упоминаний, ссылок, спецсимволов
    """
    text = text.lower()
    text = re.sub(r'@\w+', '', text)  # Удаляем @упоминания
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Удаляем URL
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем все кроме букв, цифр и базовой пунктуации
    text = ' '.join(text.split())  # Убираем лишние пробелы
    return text

def load_and_process_data(file_path, max_lines = 500000):
    """
    Загрузка и обработка данных, чтение не более max_lines строк
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            line = line.strip()
            if line:
                texts.append(line)
    
    print(f"Загружено {len(texts)} строк из файла")
    
    cleaned_texts = []
    for text in texts:
        cleaned = clean_text(text)
        if cleaned:  # Пропускаем пустые тексты после очистки
            cleaned_texts.append(cleaned)
    
    return cleaned_texts

def main():
    input_file = './data/tweets.txt'
    
    # Удаление существующих файлов результатов
    output_files = [
        './data/train_cleaned.txt',
        './data/val_cleaned.txt', 
        './data/test_cleaned.txt'
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Удален старый файл: {file_path}")

    # Этап 1: Загрузка и обработка данных
    texts = load_and_process_data(input_file)
    
    # Этап 2: Разделение на train/val/test (80/10/10)
    train_texts, temp_texts = train_test_split(texts, test_size=0.2, random_state=42)
    val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)
    
    # Этап 3: Сохранение обработанных данных
    os.makedirs('./data', exist_ok=True)
    
    with open('./data/train_cleaned.txt', 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\n')
    
    with open('./data/val_cleaned.txt', 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '\n')
    
    with open('./data/test_cleaned.txt', 'w', encoding='utf-8') as f:
        for text in test_texts:
            f.write(text + '\n')
    
    # Вывод статистики
    print(f"Train: {len(train_texts)} samples")
    print(f"Val: {len(val_texts)} samples") 
    print(f"Test: {len(test_texts)} samples")

if __name__ == "__main__":
    main()