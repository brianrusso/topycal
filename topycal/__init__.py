from __future__ import print_function
from time import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import glob, os.path, codecs, json
from collections import Sequence
DEFAULT_TOPIC_MODEL = 'nmf'

class Topycal(Sequence):
    # array of dictionaries/simple objects
    def __init__(self, document_array, content_key, num_topics=10, num_topic_words = 20, num_features = 1000, topic_key='topics', topic_threshold=0.01):
        
        self.num_topics = num_topics
        self.num_topic_words = num_topic_words
        self.num_features = num_features
        self.topic_threshold = topic_threshold
        self.drop_word_len = 4 # will internally drop words shorter than this. i.e. this is the min
        self.docs = document_array
        self.content_key = content_key
        self.naked_docs = self.get_naked_docs(self.docs, self.content_key)
        
        # These store the doc topic distributions per model as well as the associated topic info
        self.doc_topic_distrib = {}
        self.topics = {}
        
        # output
        self.topic_key = topic_key

    def model_with_lda(self):
        # Raw Count vectors
        tf_vectors, tf_vectorizer = self.vectorize_as_tf(self.naked_docs)
        lda = LatentDirichletAllocation(n_topics=self.num_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
        lda.fit(tf_vectors)
        tf_feature_names = tf_vectorizer.get_feature_names()
        self.topics['lda'] = self.get_topics(lda, tf_feature_names, self.num_topic_words)
        self.doc_topic_distrib['lda'] = lda.transform(tf_vectors)
        
    
    def model_with_nmf(self):
        # TfIdf vectors
        tfidf_vectors, tfidf_vectorizer = self.vectorize_as_tfidf(self.naked_docs)
        nmf = NMF(n_components=self.num_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf_vectors)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        self.topics['nmf'] = self.get_topics(nmf, tfidf_feature_names, self.num_topic_words)
        self.doc_topic_distrib['nmf'] = nmf.transform(tfidf_vectors)

    
        
    def get_topics_for_doc(self, doc_num, threshold, model=DEFAULT_TOPIC_MODEL):
        if model in self.doc_topic_distrib:
            high_score_indices = self.get_topicidxs_for_doc(self.doc_topic_distrib[model], threshold, doc_num)
            if self.topic_names:
                high_score_topics = []
                for index in high_score_indices:
                    high_score_topics.append(self.topic_names[index])
                return high_score_topics
            return high_score_indices
        else:
            raise Exception("Model {} not supported".format(model))
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, index):
        orig_doc = self.docs[index]
        orig_doc[self.topic_key] = self.get_topics_for_doc(index, threshold=self.topic_threshold)
        return orig_doc
    
    @staticmethod
    def get_topicidxs_for_doc(doc_topic_distrib, threshold, doc_num):
        scores = doc_topic_distrib[doc_num]
        high_score_indices = np.where(scores > threshold)
        high_scores = high_score_indices[0].tolist()
        return high_scores
        
    @staticmethod
    def get_topicidx_for_doc(doc_topic_distrib, doc_num):
        max_topic_idx = np.argmax(doc_topic_distrib[doc_num])
        return [max_topic_idx]
        
    # Makes a list of documents with content only.
    # Must remain same length/order as docs
    def get_naked_docs(self, docs, content_key):
        content_only = []
        for elem in docs:
            try:
                if self.drop_word_len:
                    this_content = ' '.join(word for word in elem[content_key].split() if len(word)>=self.drop_word_len)
                    content_only.append(this_content)
                else:
                    content_only.append(elem[content_key])
            except KeyError: # simpler and probably as memory-efficient as tracking separately
                content_only.append('')
        return content_only
    
    # simple list where position corresponds to topic #
    # e.g. pos 0, -> topic 0, pos 3 -> topic 3, etc..
    def set_topic_names(self, topic_names):
        self.topic_names = topic_names
        
    def get_topics(self, model, feature_names, n_top_words):
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            topics[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return topics

    def vectorize_as_tfidf(self, naked_docs):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                max_features=self.num_features,
                                #ngram_range=self.ngram_range,
                                stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(naked_docs)
        return (tfidf, tfidf_vectorizer)

    def vectorize_as_tf(self, naked_docs):
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=self.num_features,
                                #ngram_range=self.ngram_range,
                                stop_words='english')
        tf = tf_vectorizer.fit_transform(naked_docs)
        return (tf, tf_vectorizer)
