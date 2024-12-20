import pandas as pd
import nltk
import torch


nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)


class Vectorizer():
    """
    abstract class for all vectorizers
    self._vectorize has to be implemented
    """

    vocab_ = None

    def _update_vocabulary(self, data: pd.DataFrame):
        """
        learn a vocabulary of new words by mapping them to indexes by occurrence
        """
        for doc in data:
            for token in doc:
                self.vocab_.setdefault(token, len(self.vocab_))
    
    def _normalize_text(self, text: str) -> list:
        """
        clear text and create a list of lemmatized words for every document
        """
        text = text.lower()
        
        #tokenize, remove stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        tokens = nltk.word_tokenize(text)
        
        #tag pos
        tagged_tokens = nltk.pos_tag(tokens)
        tagged_tokens = map(lambda x: (x[0], self._nltk_pos_tagger(x[1])), tagged_tokens)
        
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmetized = []
        
        for token, tag in tagged_tokens:
            if token.isalpha():
                if tag is not None:
                    token = lemmatizer.lemmatize(token, tag)
                if token not in stopwords:
                    lemmetized.append(token)
        
        return lemmetized
    
    def _nltk_pos_tagger(self, nltk_tag):
        """
        mapping of nltk pos_tag to wordcloud pos_tag
        """
        if nltk_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif nltk_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        return None
        
    def fit(self, data: pd.DataFrame):
        """
        learn a vocabulary of new words from data
        """
        data = data.apply(self._normalize_text)
        self._update_vocabulary(data)
    
    def transform(self, data: pd.DataFrame) -> torch.Tensor:
        """
        create matrix of occurence of words from vocabulary
        """
        data = data.apply(self._normalize_text)
        data = self._vectorize(data)
        return data
        
    def fit_transform(self, data: pd.DataFrame) -> torch.Tensor:
        """
        learn vocabulary of the words from data and return matrix of occurence of words
        """
        data = data.apply(self._normalize_text)
        self._update_vocabulary(data)
        data = self._vectorize(data)
        return data

class CountVectorizer(Vectorizer):
    def __init__(self):
        self.vocab_ = { }
    
    def _vectorize(self, data: pd.DataFrame) -> torch.sparse_coo_tensor:
        """
        create a coordinate sparse matrix (W, D) with word counts as values
        """
        word_indexes, document_indexes, values = [], [], []
        
        #iterating through all documents, mapping each document to its doc_idx
        for doc_idx, doc in data.items():
            token_count = { }
            doc_word_count = len(doc)
            
            #count the amount of all words in each document
            for token in doc:
                token_count[self.vocab_[token]] = token_count.get(self.vocab_[token], 0) + 1
            
            for word in token_count.keys():
                #append new record to sparse matrix
                document_indexes.append(doc_idx)
                word_indexes.append(word)
                values.append(token_count[word])
        
        return torch.sparse_coo_tensor((word_indexes, document_indexes), values).coalesce()

class TfidfVectorizer(Vectorizer):
    def __init__(self):
        self.vocab_ = { }

    def __token_idf(self, data):
        """
        calculate idf for each unique token from data
        """
        #count of token unique appearances in docs
        doc_count = torch.tensor(len(data), dtype=torch.float32)
        token_idf = { }
        
        for doc in data:
            for token in set(doc):
                token_idf[token] = token_idf.get(token, torch.tensor(0.0)) + 1

        for token in token_idf.keys():
            token_idf[token] = torch.log(doc_count / token_idf[token])
    
        return token_idf

    def _vectorize(self, data: pd.DataFrame):
        """
        create a sparse tf-idf matrix
        """
        #(word, doc): value
        matrix_dict = { }
        token_idf = self.__token_idf(data)
        
        #iterating through all documents, mapping each document to its doc_idx
        for doc_idx, doc in data.items():
            token_count = { }
            doc_word_count = len(doc)
            
            #count the amount of all words in each document
            for token in doc:
                token_count[token] = token_count.get(token, 0) + 1
            
            #calculate tf-idf
            for token in token_count.keys():
                token_freq = token_count[token] / doc_word_count
                inv_doc_freq = token_idf[token]
                matrix_dict[(self.vocab_[token], doc_idx)] = token_freq * inv_doc_freq

        return torch.sparse_coo_tensor(tuple(zip(*matrix_dict.keys())), list(matrix_dict.values())).coalesce()