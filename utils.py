import re
from typing import List
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# import spacy
# from spacy.tokenizer import Tokenizer
# from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex

# # Define a custom tokenizer function
# def custom_tokenizer(nlp):
#     prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
#     suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
#     custom_infixes = [r'\w+(?:-\w+)+']
#     infix_re = compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))
#     tokenizer = Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer)
#     return tokenizer

# nlp_spacy = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'lemmatizer'])
# nlp_spacy.tokenizer = custom_tokenizer(nlp_spacy)


ps = PorterStemmer()

def clean_text(text: str) -> List[str]:
    text = text.lower()
    
    # Remove stopwords from text using regex
    stopwords_list = set(nltk.corpus.stopwords.words('english'))
    stopwords_list.remove('as')
    stopwords_pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    text = re.sub(stopwords_pattern, '', text)

    # Replace newline, tab with space
    text = re.sub(r'[\n\t]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # remove # from markdown
    text = re.sub(r'#+', '', text)
    text = re.sub(r'<[^>]*>', '', text)

    # remove from user query
    text = re.sub(r'supergpt', '', text)
    text = re.sub(r'user', '', text)
    # words = [token.text for token in nlp_spacy(text) if token.pos_ not in ["VERB", "ADJ"]]
    # text = ' '.join(words)

    words = text.strip().split(" ")
    # remove leading and trailing non-alphabet characters
    words = [re.sub(r'^[^a-zA-Z0-9+]*|[^a-zA-Z0-9+]*$', '', t) for t in words]
    words = [t for t in words if t != "-" and t != ""]
    
    # words stemming
    # text = [ps.stem(t) for t in text]

    # words = nltk.pos_tag(words)
    # print(words)
    # words = [word for word, tag in words if not tag.startswith('VB') and not tag.startswith('JJ')]

    return words


def contains_alphabets_and_numbers(word):
    # Define a regex pattern that checks for at least one alphabet and one digit
    pattern = r'(?=.*[A-Za-z])(?=.*\d)'
    # Search for the pattern in the word
    if re.search(pattern, word):
        return True
    return False