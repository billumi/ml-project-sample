
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_text(text_series):
    v=TfidfVectorizer()
    return v.fit_transform(text_series), v
