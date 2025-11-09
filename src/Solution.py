# Импорт функции
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data-utils import preprocess_tweets

# Обработка конкретного файла
cleaned_tweets, tokenized_tweets = preprocess_tweets('data/tweets.txt', 'bert-base-uncased')

# Файлы будут сохранены как:
# data/tweets_cleaned.txt (очищенные твиты)
# data/tweets_tokenized.txt (токенизированные твиты)

print(f"Обработано {len(cleaned_tweets)} очищенных твитов")
print(f"Обработано {len(tokenized_tweets)} токенизированных твитов")

print("\nПервые 3 очищенных твитов:")
for i, tweet in enumerate(cleaned_tweets[:3], 1):
    print(f"{i}. {tweet}")

print("\nПервые 3 токенизированных твитов:")
for i, tweet in enumerate(tokenized_tweets[:3], 1):
    print(f"{i}. {tweet}")