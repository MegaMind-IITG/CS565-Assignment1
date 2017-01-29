#Assignment 1 Part 1
#We use the freely available inaugural corpus (inaugural address of all American presidents since 1789)

import nltk
# from nltk.corpus import inaugural
# from nltk.tokenize import sent_tokenize
# from nltk import word_tokenize
# from nltk.util import ngrams
# from nltk.corpus.reader import PlaintextCorpusReader
from collections import Counter
# import progressbar
# from time import sleep
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ##Combining all documents in one raw text
# print 'Reading corpus...'
# corpus_root = './inaugural/'
# inaugural = PlaintextCorpusReader(corpus_root, '.*')

##Corpus information:
##Average word length, average sentence length, average number of times a vocabulary item appears in corpus 
# for fileid in inaugural.fileids():
# 	num_chars = len(inaugural.raw(fileid))
# 	num_words = len(inaugural.words(fileid))
# 	num_sents = len(inaugural.sents(fileid))
# 	num_vocab = len(set(w.lower() for w in inaugural.words(fileid)))
	# print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)

"""
#Sentence segmentation
print "Segmenting text into sentences..."
sent_tokenize_list = []
for fileid in inaugural.fileids():
	sent_tokenize_list = sent_tokenize_list + sent_tokenize(inaugural.raw(fileid))
	
sent_count = len(sent_tokenize_list)
print "Total number of sentences = ",sent_count

#Creating dictionary, counting unigrams, bigrams and trigrams
print "Creating dictionary, counting unigrams, bigrams and trigrams..."
inaugural_dict = set()
unigrams = Counter([])
bigrams = Counter([])
trigrams = Counter([])
bar = progressbar.ProgressBar(maxval=sent_count,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i,sent in enumerate(sent_tokenize_list):
	tokens = word_tokenize(sent)
	for token in tokens:
		inaugural_dict.add(token)
	unigrams += Counter(tokens)
	bigrams += Counter(ngrams(tokens,2))
	trigrams += Counter(ngrams(tokens,3))
	bar.update(i+1)
bar.finish()

print "Dictionary size = ",len(inaugural_dict)
print "Number of unigrams",len(unigrams)
print "Number of bigrams",len(bigrams)
print "Number of trigrams",len(trigrams)

##Saving data in pickle file
with open('inaugural_data.pickle', 'wb') as f:
	pickle.dump(unigrams,f)
	pickle.dump(bigrams,f)
	pickle.dump(trigrams,f)
"""

#Loading from pickle file
with open('inaugural_data.pickle', 'rb') as f:
	unigrams = pickle.load(f)
	bigrams = pickle.load(f)
	trigrams = pickle.load(f)

#Plotting frequency distributions
unigram_counts = []
bigram_counts = []
trigram_counts = []
for x in unigrams:
	unigram_counts.append(unigrams[x])
for x in bigrams:
	bigram_counts.append(bigrams[x])
for x in trigrams:
	trigram_counts.append(trigrams[x])

unigram_counts = np.array(unigram_counts)
bigram_counts = np.array(bigram_counts)
trigram_counts = np.array(trigram_counts)

# f = plt.figure(1)
# plt.hist(unigram_counts,range=(1,20), bins=20)
# plt.xlabel('Frequency')
# plt.ylabel('No. of unigrams')
# f.show()

# g = plt.figure(2)
# plt.hist(bigram_counts,range=(1,20), bins=20)
# plt.xlabel('Frequency')
# plt.ylabel('No. of bigrams')
# g.show()

# h = plt.figure(3)
# plt.hist(trigram_counts,range=(1,10), bins=10)
# plt.xlabel('Frequency')
# plt.ylabel('No. of trigrams')
# h.show()

##Few Basic Questions

corpus_size = np.sum(unigram_counts)
print "Total corpus size =",corpus_size
uni_dec = np.sort(unigram_counts)[::-1]
bi_dec = np.sort(bigram_counts)[::-1]
tri_dec = np.sort(trigram_counts)[::-1]

print np.argmin(uni_dec.cumsum() < 0.9*corpus_size),"most frequent words (total",len(unigrams),"words) make up 90%% of the corpus."
print np.argmin(bi_dec.cumsum() < 0.8*corpus_size),"most frequent bigrams (total",len(bigrams),"bigrams) make up 80%% of the corpus."
print np.argmin(tri_dec.cumsum() < 0.7*corpus_size),"most frequent trigrams (total",len(trigrams),"trigrams) make up 70%% of the corpus."

"""
Total corpus size = 144997
2267 most frequent words (total 9873 words) make up 90%% of the corpus.
39517 most frequent bigrams (total 63678 bigrams) make up 80%% of the corpus.
77189 most frequent trigrams (total 111011 trigrams) make up 70%% of the corpus.
"""