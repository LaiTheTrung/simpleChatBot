import nltk
import re, string, unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')
def remove_non_ascii(input):
    return input.encode('ascii', 'ignore').decode('utf-8', 'ignore')
def wn_output(input): # for data
    list_wn =[]
    for word in input:
        wn_word = wn.synsets(word)
        print(wn_word)
        for syns_word in wn_word:
            print(syns_word)
            world = syns_word.frame_strings().join()
        list_wn.append(wn_word)
    # # preprocess
    # original_list = list_wn.copy()
    # flatten_list = list(np.concatenate(original_list). flat)
    return list_wn
