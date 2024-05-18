import re
from typing import List
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
 
ps = PorterStemmer()

def clean_text(text: str) -> List[str]:
    text = text.lower()
    
    # Remove stopwords from text using regex
    stopwords_list = [
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'with', 'to', 'from',
        'at', 'by', 'for', 'about', 'against', 'between', 'into', 'through', 'during', 
        'before', 'after', 'above', 'below', 'of', 'off', 'over', 'under', 'again', 
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
        't', 'can', 'will', 'just', 'should', 'now'
    ]
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

    text = text.strip().split(" ")
    # remove leading and trailing non-alphabet characters
    text = [re.sub(r'^[^a-zA-Z0-9+]*|[^a-zA-Z0-9+]*$', '', t) for t in text]
    
    text = [t for t in text if t != "-" and t != ""]
    # words stemming
    # text = [ps.stem(t) for t in text]
    return list(set(text))