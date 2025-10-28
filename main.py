import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
import random
import string

# Download only essential data
try:
    nltk.download('movie_reviews', quiet=True)
except:
    pass

# Simple tokenizer that doesn't require punkt_tab
def simple_tokenize(text):
    # Convert to lowercase and split on whitespace/punctuation
    text = text.lower()
    # Replace punctuation with spaces and split
    for punc in string.punctuation:
        text = text.replace(punc, ' ')
    return text.split()

# Preprocess function
def extract_features(words):
    return {word: True for word in words}

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
               for category in movie_reviews.categories()
               for fileid in movie_reviews.fileids(category)]

# Shuffle dataset 
random.shuffle(documents)

# Prepare dataset for training & testing
featuresets = [(extract_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]

# Train Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate The Classifier
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

classifier.show_most_informative_features(10)

# Simplified sentiment analysis function
def analyze_sentiment(text):
    # Use simple tokenization
    words = simple_tokenize(text)
    
    # Remove very short words (often not meaningful)
    words = [word for word in words if len(word) > 2]
    
    # Extract features and predict
    features = extract_features(words)
    return classifier.classify(features)

# Test classifier with some custom inputs
test_sentences = [
    "This movie is absolutely fantastic! The acting, the story, everything was amazing!",
    "I hated this movie. It was a waste of time & money.",
    "The plot was a bit dull, but the performances were great.",
    "I've mixed feelings about this film. It was okay, not great but not terrible either."
]

print("\n--- Sentiment Analysis Results ---")
for sentence in test_sentences:
    sentiment = analyze_sentiment(sentence)
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {sentiment}")
    print()