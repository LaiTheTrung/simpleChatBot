import nltk
from clean import remove_non_ascii
# nltk.download('punkt') download package to tokenize
from nltk.stem.porter import PorterStemmer
from snowballstemmer import stemmer
import numpy as np
stemmer = PorterStemmer()
def tokenize(sentence):
    sentence = remove_non_ascii(sentence)
    return nltk.word_tokenize(sentence)
  
def stem (word):
    return stemmer.stem(word)

def exclude(stemmed_sentence):
    sentence = stemmed_sentence.copy()
    ignore_letters = ['!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','}','|',':',';','"','<','>','?',',','~','`']
    for letter in stemmed_sentence:
        if letter in ignore_letters:
            sentence.remove(letter)
    return sentence

def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    
    return bag
# a = "Hi there, what can I do for you???"

# tokenized_a= tokenize(a)
# stemmed_a = [stem(w) for w in tokenized_a]
# exclude_a=exclude(stemmed_a)
