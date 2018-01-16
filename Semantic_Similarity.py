
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog
import time

def vectorize(sentence, corpus):
	similar = [sentence]
	train_set = similar + corpus
	vectorizer = TfidfVectorizer(max_df=0.5,stop_words='english')
	X = vectorizer.fit_transform(train_set)
	lsa = TruncatedSVD(n_components=500, algorithm='randomized',n_iter=20,random_state=101)
	X = lsa.fit_transform(X)
	doc_vectorized = Normalizer(copy=False).fit_transform(X)
	return doc_vectorized, train_set

def cosines(doc_vectorized, train_set):
	cosine_similarities = cosine_similarity(doc_vectorized[0:1],doc_vectorized).flatten()
	related_docs_indices = cosine_similarities.argsort()[:-3:-1]
	most_similar_sentence = train_set[related_docs_indices[1]]
	return most_similar_sentence

def find_similar_sentence(input_sentence, corpus_file):
	start_time = time.time()
	with open(corpus_file) as f:
		corpus = f.readlines()
	corpus = [x.strip() for x in corpus]
	vectorized_document, training_set = vectorize(input_sentence, corpus)
	most_similar_sentence = cosines(vectorized_document, training_set)
	print ("Took {} seconds".format(time.time() - start_time))
	return most_similar_sentence

root = tk.Tk()
root.withdraw()
corpus_file_path = filedialog.askopenfilename()
sentence = input("Enter the sentence for similarity check: ")
most_sim_sentence = find_similar_sentence(sentence,corpus_file_path)
print ("most_sim_sentence: {}".format(most_sim_sentence))

