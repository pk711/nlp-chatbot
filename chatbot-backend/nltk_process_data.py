import numpy as np
import re

def tokenize(sentence):
    return re.findall(r'\b\w+\b', sentence.lower())

def stemming(word):
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stemming(word) for word in tokenized_sentence]
    
    bag = np.zeros(len(words), dtype=np.float32)
    
    for index, w in enumerate(words):
        if w in sentence_words: 
            bag[index] = 1

    return bag