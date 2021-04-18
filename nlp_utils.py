import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def clean_text_data(df, var, stopwords_list=None):
    """
    Function to clean a text column in a Pandas DataFrame
    - converts string to lowercase
    - removes stopwords
    - removes numbers and special characters
    - removes multiple spaces
    """
    if stopwords_list is None:
        stopwords_list = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

    df_copy = df.copy()
    df_copy[var] = df_copy[var].apply(lambda x: x.lower())
    df_copy[var] = df_copy[var].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))
    df_copy[var] = df_copy[var].apply(lambda x: re.sub('[^a-z ]', "", x))
    df_copy[var] = df_copy[var].apply(lambda x: ' '.join(x.split()))

    return df_copy


def stem_docs(docs):
    """Return stemmed documents"""
    stemmer = EnglishStemmer()
    return [" ".join([stemmer.stem(word) for word in sentence.split(" ")]) for sentence in docs]


def before_and_after_stemming(string, vocab, vocab_stemmed):
    """Prints vocabulary words beginning in a sequence of characters before and after stemming"""
    r = re.compile("^" + string)
    vocab_sample = list(filter(r.match, vocab))
    vocab_sample_stem = list(filter(r.match, vocab_stemmed))

    print(f'Stemming example for words starting with "{string}"')
    print('Original vocabulary')
    print(vocab_sample)

    print('Stemmed vocabulary')
    print(vocab_sample_stem, '\n')


def plot_word_freq(x_vec, word_names, n_words=10, title='Word Frequency', figsize=(10, 8), color='orange'):
    """Plots word frequencies from a trained count vectorizer"""
    vocab_df = pd.DataFrame()
    vocab_df['words'] = word_names
    vocab_df['counts'] = x_vec.sum(axis=0)
    vocab_df.sort_values('counts', inplace=True)

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.barh(vocab_df['words'][-n_words:], vocab_df['counts'][-n_words:], color=color, alpha=0.6)
    plt.show()


def get_wordnet_pos(treebank_tag):
    """Returns WordNet part of speech"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def tag_and_lemmatize(tokens, lemmatizer):
    """Tag and lemmatize a list of word tokens"""
    transformed_docs = []
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            lemma = lemmatizer.lemmatize(word)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        transformed_docs.append(lemma)
    return transformed_docs


def lemmatize_docs(docs):
    """Return lemmatized documents"""
    lemmatizer = WordNetLemmatizer()
    # lemmed = [[lemmatizer.lemmatize(word) for word in word_tokenize(doc)] for doc in docs]
    lemmed = [tag_and_lemmatize(word_tokenize(doc), lemmatizer) for doc in docs]
    return [" ".join(x) for x in lemmed]


def feature_vectorisation(x_train, x_test, stem=True, count_vectorizer=True, min_df=1, ngram_range=(1, 1),
                          scale_data=False):
    """Applies varying settings for feature vectorisation"""
    if stem:
        x_train = stem_docs(x_train)
        x_test = stem_docs(x_test)
    else:
        x_train = lemmatize_docs(x_train)
        x_test = lemmatize_docs(x_test)

    if count_vectorizer is True:
        vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

    vectorizer.fit(x_train)

    x_train_vec = vectorizer.transform(x_train).toarray()
    x_test_vec = vectorizer.transform(x_test).toarray()

    if scale_data:
        scaler = StandardScaler()
        x_train_vec = scaler.fit_transform(x_train_vec)
        x_test_vec = scaler.transform(x_test_vec)

    return x_train_vec, x_test_vec


def label_sentences(corpus, label_type):
    """
    Create document tags
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled


def get_vectors(model, documents):
    """
    Get vectors from trained Doc2Vec model
    """
    corpus_size = len(documents)
    vector_size = model.vector_size
    vectors = np.zeros((corpus_size, vector_size))

    for i, doc in enumerate(documents):
        vectors[i] = model.infer_vector(doc[0])
    return vectors


def feature_vectorisation_d2v(x_train, x_test, dm=0, vector_size=300, window=15, min_count=5,
                              sample=10e-5, alpha=0.025, epochs=20):
    """Create Doc2Vec vectors"""
    x_train_tagged = label_sentences(x_train, 'Train')
    x_test_tagged = label_sentences(x_test, 'Test')

    d2v_model = Doc2Vec(dm=dm, vector_size=vector_size, window=window, min_count=min_count,
                        sample=sample, alpha=alpha, epochs=epochs)
    d2v_model.build_vocab(x_train_tagged)
    d2v_model.train(x_train_tagged, total_examples=len(x_train_tagged), epochs=d2v_model.epochs)

    train_vectors_dbow = get_vectors(d2v_model, x_train_tagged)
    test_vectors_dbow = get_vectors(d2v_model, x_test_tagged)

    return train_vectors_dbow, test_vectors_dbow
