import numpy as numpy
import spacy
import nltk
from nltk.corpus import inaugural
import textacy
import time
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt

## Requirements:
# nltk.download('inaugural')
# nltk.download('punkt')

def sentenceSegmentation(corpus, name):
	#Sentence Segmentation
	print "Loading English Data ... (may take some time)"
	nlp = spacy.load('en')
	print "Loading completed."

	print "Segementing text into sentences ..."
	segmentedSentences = []

	for id in corpus.fileids():
		print "Segmenting sentences of file: " + id
		for sent in nlp(corpus.raw(id)).sents:
			segmentedSentences.append(unicode(str(sent), 'utf-8'))
	print("Sentence Segmentation completed. Total %d sentences in all." % (len(segmentedSentences)))
	pickle.dump(segmentedSentences, open(name + '.sent', 'wb'))

def sentsToCorpus(name):
	sentences = pickle.load(open(name + '.sent', 'r'))

	print("Converting the sentences to corpus ... (may take some time)")
	# corpus = textacy.Corpus(lang=u'en', texts=sentences)
	corpus = textacy.Corpus(lang=u'en', texts=sentences)
	print("Docs to Corpus converted!")
	return corpus
	# pickle.dump(corpus, open(name + '.corpus', 'wb'))

def createDict(corpus):
	# Ignores stop words! 
	corpusDict = corpus.word_freqs(lemmatize=False, lowercase=False, as_strings=True)
	return corpusDict

#Recognising UniGrams, BiGrams and Trigrams
def findNGrams(corpus, n, name = "corpus", filter_stops = True, lemmatize = False):
	ngrams = {}
	# print(list(corpus[0:3]))
	print("Starting to calculate %dgrams for %s" % (n, name))
	start = time.time()
	for sent in corpus:
		dict1 = sent.to_bag_of_terms(ngrams = n, lemmatize=lemmatize, lowercase=False, normalize=False, as_strings=True, filter_stops=filter_stops)
		ngrams = { k: dict1.get(k, 0) + ngrams.get(k, 0) for k in set(dict1) | set(ngrams) }
	end = time.time()
	print("Took %d seconds to complete. Storing %dgram to file %s.%dgram" % ((end - start), n, name, n))
	
	exceptions = []
	for key in ngrams.keys():
		if len(key.split(' ')) > n:
			exceptions.append(key)

	# pickle.dump(ngrams, open(name + '.' + str(n) + 'gram', 'wb'))
	return ngrams, exceptions

def count(freqDict):
	counts = []
	for x in freqDict:
		counts.append(freqDict[x])
	counts = np.array(counts)
	return counts	

def plot(counts):
	fig = plt.figure()
	plt.hist(counts,range=(1,20), bins=20)
	plt.xlabel('Frequency')
	plt.ylabel('No. of NGrams')
	fig.show()
	
def countAndPlot(freqDict):
	counts = count(freqDict)
	plot(counts)
	return counts

corpusName = 'inaugural'
unigrams = {}
bigrams = {}
trigrams = {}

# sentenceSegmentation(inaugural, 'inaugural')
# corpus = sentsToCorpus(corpusName)

# corpusDict = createDict(corpus)
# unigrams = findNGrams(corpus, 1, corpusName, filter_stops = True)
# bigrams = findNGrams(corpus, 2, corpusName, filter_stops = True)
# trigrams = findNGrams(corpus, 3, corpusName, filter_stops = True)

# unigrams = findNGrams(corpus, 1, corpusName, filter_stops = True, lemmatize=True)
# bigrams = findNGrams(corpus, 2, corpusName, filter_stops = True, lemmatize=True)
# trigrams = findNGrams(corpus, 3, corpusName, filter_stops = True, lemmatize=True)

# unigrams, uniExcep = findNGrams(corpus, 1, corpusName, filter_stops = False)
# bigrams, biExcep = findNGrams(corpus, 2, corpusName, filter_stops = False)
# trigrams, triExcep = findNGrams(corpus, 3, corpusName, filter_stops = False)

unigram_counts = countAndPlot(unigrams)
bigram_counts = countAndPlot(bigrams)
trigram_counts = countAndPlot(trigrams)


##Few Basic Questions
corpus_size = np.sum(unigram_counts)
print "Total corpus size =",corpus_size
uni_dec = np.sort(unigram_counts)[::-1]
bi_dec = np.sort(bigram_counts)[::-1]
tri_dec = np.sort(trigram_counts)[::-1]

print np.argmin(uni_dec.cumsum() < 0.9*corpus_size),"most frequent words (total",len(unigrams),"words) make up 90% of the corpus."
print np.argmin(bi_dec.cumsum() < 0.8*corpus_size),"most frequent bigrams (total",len(bigrams),"bigrams) make up 80% of the corpus."
print np.argmin(tri_dec.cumsum() < 0.7*corpus_size),"most frequent trigrams (total",len(trigrams),"trigrams) make up 70% of the corpus."

# Note: Punctuations are ignored. 

##### Removed stop words and then calculated. 
# Took 16 seconds to complete. Storing 1gram to file inaugural.1gram
# Took 15 seconds to complete. Storing 2gram to file inaugural.2gram
# Took 19 seconds to complete. Storing 3gram to file inaugural.3gram

# Total corpus size = 58038
# 4619 most frequent words (total 9632 words) make up 90% of the corpus.
# 0 most frequent bigrams (total 11352 bigrams) make up 80% of the corpus.
# 0 most frequent trigrams (total 16135 trigrams) make up 70% of the corpus.

##### Lemmatized and without stop words and then calculated.
# Took 9 seconds to complete. Storing 1gram to file inaugural.1gram
# Took 12 seconds to complete. Storing 2gram to file inaugural.2gram
# Took 17 seconds to complete. Storing 3gram to file inaugural.3gram

# Total corpus size = 58038
# 2724 most frequent words (total 6821 words) make up 90% of the corpus.
# 0 most frequent bigrams (total 10946 bigrams) make up 80% of the corpus.
# 0 most frequent trigrams (total 15967 trigrams) make up 70% of the corpus.

##### Stop Words included and No Lemmatization
# Took 14 seconds to complete. Storing 1gram to file inaugural.1gram
# Took 107 seconds to complete. Storing 2gram to file inaugural.2gram
# Took 172 seconds to complete. Storing 3gram to file inaugural.3gram

# Total corpus size = 134413
# 2556 most frequent words (total 10080 words) make up 90% of the corpus.
# 43198 most frequent bigrams (total 56840 bigrams) make up 80% of the corpus.
# 74375 most frequent trigrams (total 89535 trigrams) make up 70% of the corpus.
