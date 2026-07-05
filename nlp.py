PRACTICAL 1
AIM: Write a program to implement sentence segmentation and word tokenization
CODE:
pip install nltk
#text_preprocessing packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
# download stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
#function for tokenization
def tokenize(text):
  """Tokenize the input text into words."""
  return word_tokenize(text)
# function for Stop-words Removal
def remove_stopwords(tokens):
  """Remove stop words from the list of tokens."""
  stop_words=set(stopwords.words('english'))
  return [word for word in tokens if word.lower()not in stop_words]
#function for text normalization
def normalized(text):
  """Normalizes the text by converting it to lowercase and removing punctuation."""
  text = text.lower() #convert to lowercase
  text = text.translate(str.maketrans('','',string.punctuation))
  return text
# preprocessing pipeline function
def preprocess_text(text):
  """Combines all text preprocessing steps."""
  normalized_text = normalized(text) # this is goa it has beaches
  tokens = tokenize(normalized_text) # this, is, goa, it, has, beaches
  #print(tokens)
  filtered_tokens = remove_stopwords(tokens) #goa beaches
  return ' '.join(filtered_tokens) #Return as a single string
  #preprocess_text("This is Goa . It has beaches")
input_file_path = '/content/SMSSpamCollection'
output_file_path = 'processed_output.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
  lines = file.readlines()
#Process each lines and store results
processed_lines = [preprocess_text(line.strip()) for line in lines]
#save the processed lines to a new text file
with open(output_file_path, 'w', encoding='utf-8') as file:
  for line in processed_lines:
    file.write(line + '\n')
print(f"Text preprocessing completed and saved to '{output_file_path}'.")
PRACTICAL 2.1
AIM : Write a program to Implement stemming and lemmatization
CODE:
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

data={'messages': ["The cats are playing in the garden.",
                   "He is running quickly to catch the bus.",
                   "The boys are enjoying their game.",
                   "She was reading a book.",
                   "I love to eat apples and bananas."]
      }
df = pd.DataFrame(data)

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
  #Tokenization
  tokens = word_tokenize(text)

  #Stop-words Removal
  tokens = [word for word in tokens if word.lower() not in stop_words]

  # Text Normalization (Lowercasing and Removing punctuation)
  tokens = [word.lower() for word in tokens if word.isalnum()]

  # Stemming
  tokens = [stemmer.stem(word) for word in tokens]
  return ' '.join(tokens)

# Apply the pre-processing pieline
df['processed_messages'] = df['messages'].apply(preprocess_text)

# Display the result
print(df[['messages','processed_messages']])
print("\n \n")

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
# Example sentence
sentence = "Unhappy Cats are playing"
# Prefix and suffix rules
prefixes = ["un"]
suffixes = ["ing", "ed", "s"]
# Tokenization
tokens = word_tokenize(sentence)
# Stopword removal
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word.lower() not in stop_words]
# Lowercase and remove punctuation
tokens = [word.lower().strip(string.punctuation) for word in tokens]
# Morphological analysis
processed_tokens = []
data = {}
prefix_dict = {}
suffix_dict = {}
lemma_dict = {}
for word in tokens:
    original_word = word
    # Prefix removal
    prefix_removed = ""
    for p in prefixes:
        if word.startswith(p):
            prefix_removed = p
            word = word[len(p):]
            break
    # Suffix removal
    suffix_removed = ""
    for s in suffixes:
        if word.endswith(s):
            suffix_removed = s
            word = word[:-len(s)]
            break
    lemma = word
    # Store results
    data[original_word] = [prefix_removed, lemma, suffix_removed]
    prefix_dict[original_word] = prefix_removed
    suffix_dict[original_word] = suffix_removed
    lemma_dict[original_word] = lemma
    processed_tokens.append(lemma)
# Unique tokens
unique_tokens = list(set(processed_tokens))
# Output
print("Processed Tokens:", unique_tokens)
print("Data =", data)
print("Prefix =", prefix_dict)
print("Lemma =", lemma_dict)
print("Suffix =", suffix_dict)
PRACTICAL 3
AIM: Write a program to Implement a tri-gram model
CODE:
# Step 1: Import Libraries
import nltk
nltk.download('punkt_tab')
from nltk.util import ngrams
from collections import defaultdict, Counter
import string

# Download tokenizer (run once)
nltk.download('punkt')

# Step 2: Input Corpus
corpus = """
Natural language processing (NLP) is a subfield of artificial intelligence (AI).
It enables computers to understand, interpret, and generate human language.
With advances in machine learning and deep learning, NLP has made significant strides.
Applications include sentiment analysis, machine translation, and chatbot development.
"""

# Step 3: Preprocessing
# Lowercase + remove punctuation
corpus = corpus.lower()
corpus = corpus.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokens = nltk.word_tokenize(corpus)

# Step 4: Generate Trigrams
trigrams = list(ngrams(tokens, 3))

# Step 5: Build Model (Count Frequencies)
model = defaultdict(Counter)

for w1, w2, w3 in trigrams:
    model[(w1, w2)][w3] += 1

# Step 6: Convert to Probabilities
trigram_prob = {}

for key in model:
    total = sum(model[key].values())
    trigram_prob[key] = {
        word: count / total for word, count in model[key].items()
    }

# Step 7: Prediction Function
def predict_next(word1, word2):
    key = (word1.lower(), word2.lower())
    if key in trigram_prob:
        return max(trigram_prob[key], key=trigram_prob[key].get)
    else:
        return "No prediction available"

# Step 8: Test Predictions
print("Prediction for ('natural', 'language'):", predict_next('natural', 'language'))
print("Prediction for ('machine', 'learning'):", predict_next('machine', 'learning'))
print("Prediction for ('artificial', 'intelligence'):", predict_next('artificial', 'intelligence'))
PRACTICAL 4
AIM: Write a program to Implement PoS tagging using HMM & Neural Model
CODE:
pip install transformers torch
# POS Tagging using a Neural Model (BERT)

from transformers import pipeline

def neural_pos_tagging(text):
    # Load pretrained neural POS tagging model
    pos_tagger = pipeline(
        "token-classification",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        aggregation_strategy="simple"
    )

    # Perform POS tagging
    results = pos_tagger(text)

    print("\nInput Sentence:")
    print(text)

    print("\nPOS Tags:\n")
    for token in results:
        word = token['word']
        pos = token['entity_group']
        print(f"{word:15} → {pos}")

# Example input
text = "Sanika is building a smart waste segregation system."

# Call function
neural_pos_tagging(text)
PRACTICAL 5
AIM: Write a program to Implement syntactic parsing of a given text
CODE:
!pip install "transformers==4.38.2"
!pip install "tokenizers==0.15.2"
!pip install nltk benepar
import nltk
import benepar

nltk.download('punkt')
nltk.download('punkt_tab')
benepar.download('benepar_en3')
parser = benepar.Parser("benepar_en3")
sentence = "The quick brown fox jumps over lazy dog"
tokens = nltk.word_tokenize(sentence)
tree = parser.parse(tokens)
print(tree)
tree.pretty_print()
PRACTICAL 6
AIM: Write a program to Implement dependency parsing of a given text
CODE:
import spacy
model = spacy.load("en_core_web_sm")
sentence = "The quick brown fox jumps over lazy dog"
depend = model(sentence)
print("Token\tPOS\tDep\tHead")
for token in depend:
    print(token.text,"\t",token.pos_,"\t",token.dep_,"\t",token.head.text)
PRACTICAL 7
AIM: Write a program to Implement Named Entity Recognition (NER)
CODE:
!pip install flair
from flair.models import SequenceTagger
from flair.data import Sentence
def get_user_input():
  user_input = input("Please enter a sentence: ")
  return user_input
def named_entities(text):
  tagger = SequenceTagger.load("ner")
def get_user_input():
  user_input = input("Please enter a sentence: ")
  return user_input
def named_entities(text):
  tagger = SequenceTagger.load("ner")
  sentence = Sentence(text)
  tagger.predict(sentence)
  print("\n Named Entities: \n")
  for entity in sentence.get_spans("ner"):
    print(f"- {entity.text}({entity.tag})")
text = get_user_input()
named_entities(text)
from flair.nn import Classifier
sentence = Sentence("Behavioral abnormslities in the Fmr1 KO2 Mouse Model of Fragile X syndrome")
tagger = Classifier.load("ner")
tagger.predict(sentence)
print(sentence)#entities of classes "Sprcies", "Disease" and "Gene"
PRACTICAL 8
AIM: Write a program to Implement Text Summarization for the given sample text
CODE:
from transformers import pipeline
!pip install sumy
#exttraction
import nltk
nltk.download('punkt')
nltk.download('punk_tab')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
def get_input():
  print("Please enter the text to be summarised (end with a blank line): ")
  lines=[]
  while True:
    line=input()
    if line == "":
      break
    lines.append(line)
  return "\n".join(lines)
def summarise(text, sentence_count=3):
  try:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])
  except Exception as e :
    return (f"Error occured {str(e)}")
def main():
  text = get_input()
  summary = summarise(text)
  with open('extractivesummary.txt', 'w') as file:
    file.write('Original text:\n')
    file.write(text + "\n \n")
    file.write("Summarised text: \n")
    file.write(summary)
  print("written summary to the file")
main()
!pip install transformers
!pip install sentencepiece
from transformers import pipeline
#User input function
def get_input():
    print(" Enter the text to summarize (end with the blank line):")
    lines =[]
    while True:
      line = input()
      if line == "":
        break
      lines.append(line)
    return "\n".join(lines)
def abstractive_summary(text):
    print("\n Abstractive Summary:\n")
    try:
      summarizer = pipeline("summarization", model = "t5-small", tokenizer = "t5-small")
      input_text = "summarize: " + text.strip().replace("\n"," ")
      summary = summarizer(input_text, max_length = 100, min_length = 30, do_sample = False)[0]
      print(summary['summary_text'])
    except Exception as e:
      print(f"Error: {e}")
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def abstractive_summary(text):
    print("\n* Abstractive Summary:\n")
    try:
        # Load model and tokenizer directly
        model_id = "google-t5/t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        # Prepare the input with the required T5 prefix
        input_text = "summarize: " + text.strip().replace("\n", " ")
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        # Generate the summary
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            min_length=30,
            do_sample=False
        )

        # Decode and print the output
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("-" + summary)

    except Exception as e:
        print(f"Error: {e}")
#Run
text = get_input()
abstractive_summary(text)
PRACTICAL 9
AIM: Consider a scenario of applying NLP in Customer Service. Design and develop an application that demonstrates NLP operations for working with tasks and data like voice calls, chats, Ticket Data, Email Data. Process the data to understand the voice of the Customer (intent mining, Top words, word cloud, classify topics). Identify issues, replace patterns and gain insight into sales chats.
CODE:
import json
import pandas as pd
import re
import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report

with open('/content/sales_conversations.json','r') as file:
    data = json.load(file)

print(data)

def preprocess(text):
    text = text.lower().strip() if text else ""
    text = re.sub(r'\d+','',text)
    text = re.sub(r'[^\w\s]','',text)
    return text

customer_dialogue = []
salesman_dialogue = []

for conversation in data['data']:
    for convo in conversation:
        #parser = json.loads(convo)
        if isinstance(convo,dict):
            # print(type(convo))
            # print(convo.get('Customer'))
            customer_dialogue.append(preprocess(convo.get('Customer')))
            salesman_dialogue.append(preprocess(convo.get('Salesman')))

print(customer_dialogue)

all_text = customer_dialogue + salesman_dialogue

stopwords = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords])

cleaned_customer = [remove_stopwords(text) for text in customer_dialogue]
cleaned_salesman = [remove_stopwords(text) for text in salesman_dialogue]

#Word cloud top words
from wordcloud import WordCloud
all_text_cleaned = cleaned_customer + cleaned_salesman
# print(temp)
# all_text_cleaned = " ".join(temp)

word_cloud = WordCloud(width=800, height=400).generate(" ".join(all_text_cleaned))

plt.figure(figsize=(10,5))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.show()
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

vectorizer = TfidfVectorizer(max_features=10)
tfidf_matrix = vectorizer.fit_transform(cleaned_customer + cleaned_salesman)

#  Print top words
features = vectorizer.get_feature_names_out()
top_words = np.array(features)
print(f"Top words: {top_words}")

# Identify topics using Latent Dirichlet Allocation (LDA)
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_topic_matrix = lda_model.fit_transform(tfidf_matrix)

#  Display the top word for each topic
for index, topic in enumerate(lda_model.components_):
    print(f"Topic#{index}:")
    print([features[i] for i in topic.argsort()[-1:]])


def classify_intent(text):
    if "purchase" in text or "buy" in text:
        return "Product Inquiry"
    elif "help" in text or "support" in text:
        return "Customer Support"
    elif "problem" in text or "issue" in text:
        return "Issue Reporting"
    else:
        return "General Inquiry"

#  Process the customer intents
customer_intent = [classify_intent(text) for text in cleaned_customer]

#  Print customer intent results
print(customer_intent)


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

#  Process customer sentiments and map labels
customer_sentiments = [get_sentiment(text) for text in cleaned_customer]
sentiment_labels = [
    "Positive"
    if sentiment > 0
    else "Negative"
    if sentiment < 0
    else "Neutral"
    for sentiment in customer_sentiments
]

sentiment_counts = Counter(sentiment_labels)
print(f"Sentiment Distribution: {sentiment_counts}")


df = pd.DataFrame(
    {
        "Customer Chat": customer_dialogue,
        "Salesman Chat": salesman_dialogue,
        "Intent": customer_intent,
        "Sentiment": sentiment_labels,
    }
)

df.to_csv("sales_chat_analysis.csv", index=False)
print("Sales chat analysis saved successfully to the file")
PRACTICAL 10
AIM: Consider a scenario of Online Review and demonstrate the concept of sentiment analysis and emotion mining by applying various approaches like lexicon-based approach and rule-based approaches.
CODE:
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

reviews = [
    "I love this product! It works wonderfully.",
    "The product is terrible,I hate it.",
    "It's okay, but could be better.",
    "What an amazing experience! Will buy again.",
    "Worst purchase ever. Don't buy it!"
]

# Sentiment Analysis
def analyze_sentiment(reviews):
    sentiment = SentimentIntensityAnalyzer()
    sentiments = []

    for review in reviews:
        sentiment_score = sentiment.polarity_scores(review)

        if sentiment_score['compound'] >= 0.05:
            sentiments.append('Positive')

        elif sentiment_score['compound'] <= -0.05:
            sentiments.append('Negative')

        else:
            sentiments.append('Neutral')

    return sentiments

result = analyze_sentiment(reviews)
print(result)
reviews = [
    "I love this product! It works wonderfully.",
    "The product is terrible, I hate it.",
    "The service was okay, nothing special.",
    "I am shocked by the amazing quality!",
    "The experience made me furious and disappointed."
]

result = [
    "Positive",
    "Negative",
    "Neutral",
    "Positive",
    "Negative"
]

emotion_keywords = {
    'happy':['love','amazing','wonderful','great','excited','joy'],
    'sad':['hate','terrible','disappointed','sad','worst'],
    'angry':['mad','furious','irritated','rage','angry'],
    'suprised':['suprised','shocked','unexpected','astonishing'],
    'neutral':['okay','fine','average','neutral']
}

def emotion_mining(reviews):
    emotions = []
    for review in reviews:
        review_lower = review.lower()
        detected_emotions = []

        for key, keywords in emotion_keywords.items():
            if any(keyword in review_lower for keyword in keywords):
                detected_emotions.append(key)

        if not detected_emotions:
            detected_emotions.append('neutral')

        emotions.append(detected_emotions)

    return emotions

emotions_res = emotion_mining(reviews)

for i, review in enumerate(reviews):
    print(f"Review: {review}")
    print(f"Sentiment: {result[i]}")
    print(f"Detected Emotions: {' '.join(emotions_res[i])}")
    print('-'*50)
!pip install flair

# --- Part 1: Sentiment Analysis using Flair ---
from flair.data import Sentence
from flair.nn import Classifier

sentence = Sentence('I love Mumbai and Delhi')

tagger = Classifier.load('sentiment')
tagger.predict(sentence)
print(sentence)

# --- Part 2: Emotion Classification using Hugging Face Transformers ---
from transformers import pipeline

# Note: Make sure 'reviews' is defined as a list of strings or text beforehand
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

print(emotion_classifier(reviews))
!pip install torch
!pip install transformers
from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

print(emotion_classifier(reviews))

[{'label': 'joy', 'score': 0.9666576981544495}, {'label': 'disgust', 'score': ...}]
##Practical 11
#Practical 11

with open("/content/SMSSpamCollection","rb") as file:
    file_data = file.read()
    text_data = file_data.decode('utf-8', errors='ignore')

print(text_data)
lines = text_data.split('\n')
labels = []
messages = []

for line in lines:
    if line.strip():
        label, message = line.split('\t')
        labels.append(label)
        messages.append(message)

print(labels[:10])
print(messages[:10])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.pipeline import make_pipeline

X = messages
y = labels

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

model.fit(X_train,y_train)

ypred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,ypred))
print("Classification Report: \n", classification_report(y_test,ypred))
PRACTICAL 11
AIM: Apply NLP in Banking, Financial Services, and Insurance. Design Application to detect frauds and work with SMS data
CODE:
with open ("/content/SMSSpamCollection","rb") as file:
  file_data = file.read()
  text_data = file_data.decode('utf-8', errors='ignore')
lines = text_data.split('\n')
labels = []
messages = []
for line in lines:
  if line.strip():
    label, message = line.split('\t')
    labels.append(label)
    messages.append(message)
print(labels[:10])
print(messages[:10])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
X = messages
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
ypred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test,ypred))
print("Classification Report: \n", classification_report(y_test,ypred))
PRACTICAL 12
AIM: Demonstrate the use of NLP in designing Virtual Assistants. Apply LSTM, build conversational Bots
CODE:
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
pairs = []

with open("/content/dialogues.txt", "r", encoding="latin-1") as f:
    for line in f:
        msgs = [x.strip().lower() for x in line.split("__eou__") if x.strip()]
        for i in range(len(msgs) - 1):
            pairs.append((msgs[i], msgs[i + 1]))
# Split inputs and responses
inputs = [p[0] for p in pairs]
responses = [p[1] for p in pairs]
# Tokenize inputs
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(inputs)
X = tokenizer.texts_to_sequences(inputs)
max_len = 20
X = pad_sequences(X, maxlen=max_len)
# Encode responses to labels
response_list = list(set(responses))
# 2. CREATE ID MAPPINGS AND LABELS
# ==========================================
response_to_id = {
  r: i for i, r in enumerate(response_list)
}
id_to_response = {
  i: r for r, i in response_to_id.items()
}
y = np.array([
  response_to_id[r]
  for r in responses
])
print("Pairs:", len(pairs))

questions = [p[0] for p in pairs]
responses = [p[1] for p in pairs]

print("Responses:", len(responses))

response_list = sorted(list(set(responses)))

print("Unique responses:", len(response_list))
print(response_list[:10])
# ==========================================
# 3. DEFINE, COMPILE, AND TRAIN MODEL
# ==========================================
# Tiny model
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(5000, 32),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(len(response_list), activation="softmax")
])
model.compile(
  optimizer="adam",
  loss="sparse_categorical_crossentropy",
  metrics=["accuracy"]
)
model.fit(X, y, epochs=35)
# ==========================================
# 4. CHAT FUNCTION AND INFERENCE LOOP
# ==========================================
def chat(text):
  seq = tokenizer.texts_to_sequences([text.lower()])
  seq = pad_sequences(seq, maxlen=max_len)
  pred = model.predict(seq, verbose=0)
  idx = np.argmax(pred)
  return id_to_response[idx]
# Demo loop
while True:
  msg = input("You: ")
  if msg.lower() == "quit":
    break
  print("Bot:", chat(msg))
