import numpy as np
import re

def show_similar_words(tokenizer, reps, n=10):
    for i, (word, token) in enumerate(tokenizer.word_to_token.items()):
        if i >= n:
            break
        rep = reps[token, :]
        sims = ((reps - rep) ** 2).sum(axis=1)
        nearest = np.argsort(sims)
        print(word)
        for j in nearest[:5]:
            print(" ", tokenizer.token_to_word[j])

class Tokenizer:
  def __init__(self, min_occur=10):
    self.word_to_token = {}
    self.token_to_word = {}
    self.word_count = {}
    
    self.word_to_token['<unk>'] = 0
    self.token_to_word[0] = '<unk>'
    self.vocab_size = 1
    
    self.min_occur = min_occur
    
  def fit(self, corpus):
    for review in corpus:
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
      
      for word in words:
        if word in self.word_to_token:
          self.word_count[word] += 1
        else:
          self.word_to_token[word] = self.vocab_size
          self.token_to_word[self.vocab_size] = word
          self.word_count[word] = 1
          
          self.vocab_size += 1
        
  def tokenize(self, corpus):
    tokenized_corpus = []
    for review in corpus:
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
      tokenized_review = []
      for word in words:
        if word not in self.word_count or self.word_count[word] < self.min_occur:
          tokenized_review.append(0)
        else:
          tokenized_review.append(self.word_to_token[word])
      tokenized_corpus.append(tokenized_review)
    return tokenized_corpus
    
  def de_tokenize(self, tokenized_corpus):
    corpus = []
    for tokenized_review in tokenized_corpus:
      review = []
      for token in tokenized_review:
        review.append(self.token_to_word[token])
      corpus.append(" ".join(review))
    return corpus


class CountVectorizer:
  def __init__(self, min_occur=10):
    self.tokenizer = Tokenizer()
    
  def fit(self, corpus):
    self.tokenizer.fit(corpus)

  def transform(self, corpus):
    n = len(corpus)
    X = np.zeros((n, self.tokenizer.vocab_size))
    for i, review in enumerate(corpus):
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
      for word in words:
        if word not in self.tokenizer.word_count or self.tokenizer.word_count[word] < self.tokenizer.min_occur:
          X[i][0] += 1
        else:
          X[i][self.tokenizer.word_to_token[word]] += 1
    return X


