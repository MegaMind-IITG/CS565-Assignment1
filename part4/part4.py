import math
import random

import nltk
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
def word_tokenize(text):
    toks = nltk.word_tokenize(text)
    res = []
    for tok in toks:
        tok1 = ""
        for x in tok:
            if str.isalpha(x):
                tok1 += x
        if len(tok1) > 0:
            res += [str.lower(tok1)]
    return res

def rm_one_(gram):
    return tuple(list(gram)[:-1])
        
class ngram_stats:
    def __init__(self, sentences, dev):
        self.stats = [None, {}, {}, {}]
        self.unique = [None, {}, {}]
        self.num_words = 0
        for n in [1,2,3]:
            num_grams = 0
            for sent in sentences:
                toks = ["<BEGIN>", "<BEGIN>"] + word_tokenize(sent)
                self.num_words += len(toks)
                for ngram in ngrams(toks,
                                    n, pad_left=True,
                                    pad_symbol="<BEGIN>"):
                    if ngram not in self.stats[n]:
                        self.stats[n][ngram] = 1
                        if n > 1:
                            if not rm_one_(ngram) in self.unique[n-1]:
                                self.unique[n-1][rm_one_(ngram)] = 1
                            self.unique[n-1][rm_one_(ngram)] += 1
                    else:
                        self.stats[n][ngram] += 1
                    num_grams += 1
        for x in self.stats[1]:
            if x not in self.unique[1]:
                self.unique[1][x] = 1
        for x in self.stats[2]:
            if x not in self.unique[2]:
                self.unique[2][x] = 1

        # Learn the lambda hyper-parameters
        best_lam, best_ll = [], None
        for l1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for l2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                if l1 + l2 > 0.9:
                    continue
                self.lam = [None, l1, l2, 1 - l1 - l2]
                ll = self.avg_log_likelihood(development, 'interpol')
                if best_ll is None or ll > best_ll:
                    best_ll = ll
                    best_lam = self.lam
        self.lam = best_lam
        print("Chose lambdas as %s" % str(self.lam))
        best_laplace, best_ll = None, None
        for l1 in range(100):
            self.laplace_coeff = 0.01 * (l1 + 1)
            ll = self.avg_log_likelihood(development, 'laplacian')
            if best_ll is None or ll > best_ll:
                best_ll = ll
                best_laplace = self.laplace_coeff
        self.laplace_coeff = best_laplace
        print("Chose the laplace coefficient as %f" % self.laplace_coeff) 

        # Learn the laplace coefficient hyperparameter

    def prob(self, gram, sub=False):
        gamma = 0
        if sub: gamma = 0.5
        if len(gram) == 3:
            if not gram in self.stats[3]:
                return 1.0 / len(self.stats[1])
            return 1.0 * (self.stats[3][gram] - gamma) / self.stats[2][rm_one_(gram)]
        elif len(gram) == 2:
            if not gram in self.stats[2]:
                return 1.0 / len(self.stats[1])
            return 1.0 * (self.stats[2][gram] - gamma)/ self.stats[1][rm_one_(gram)]
        elif len(gram) == 1:
            if not gram in self.stats[2]:
                return 1.0 / len(self.stats[1])
            return 1.0 * (self.stats[1][gram] - gamma) / self.num_words
        else:
            assert(False)
        
    def backoff_prob(self, gram):
        assert(len(gram) <= 3)
        if not gram in self.stats[3]:
            if not rm_one_(gram) in self.stats[2]:
                if not (gram[0],) in self.stats[1]:
                    return 1.0 / len(self.stats[1])
                #print((gram[0],) in self.unique[1], (gram[0],) in self.stats[1])
                alpha2 = 0.5 * self.unique[1][(gram[0],)] / self.stats[1][(gram[0],)]
                #print(2, alpha2)
                return alpha2 * self.prob((gram[0],))
            #print(rm_one_(gram) in self.unique[2], rm_one_(gram) in self.stats[2])
            alpha3 = 0.5 * self.unique[2][rm_one_(gram)] / self.stats[2][rm_one_(gram)]
            #print(3, alpha3)
            return alpha3 * self.prob(rm_one_(gram), True)
        return self.prob(gram, True)

    def interpol_prob(self, gram):
        assert(len(gram) == 3)
        return self.lam[3] * self.prob(gram) + \
            self.lam[2] * self.prob(rm_one_(gram)) + \
            self.lam[1] * self.prob(rm_one_(rm_one_(gram)))

    def laplacian_prob(self, gram):
        assert(len(gram) == 1)
        if not gram in self.stats[1]:
            return self.laplace_coeff / (self.num_words + self.laplace_coeff*len(self.stats[1]))
        return 1.0 * (self.stats[1][gram] + self.laplace_coeff) / (self.num_words + self.laplace_coeff*len(self.stats[1]))
    
    def avg_log_likelihood(self, sentences, prob_type):
        res, num_known = 0, 0
        for sent in sentences:
            toks = ["<BEGIN>", "<BEGIN>"] + word_tokenize(sent)
            for i in range(len(toks)):
                if prob_type == 'backoff':
                    prob = self.backoff_prob((toks[i-2], toks[i-1], toks[i]))
                elif prob_type == 'interpol':
                    prob = self.interpol_prob((toks[i-2], toks[i-1], toks[i]))
                elif prob_type == 'laplacian':
                    prob = self.laplacian_prob((toks[i],))
                else:
                    assert(False)

                res += math.log(prob)
                num_known += 1
        #print(res, num_known)
        return res / num_known

# Split sentences into three parts
book = open('book1.txt', 'r')
sentences = nltk.sent_tokenize(book.read())
train_dev = []
test = []
for sent in sentences:
    if random.random() < 0.1:
        test += [sent]
    else:
        train_dev += [sent]
        
for trial in range(5):
    training = []
    development = []
    for sent in train_dev:
        if random.random() < 0.1:
            development += [sent]
        else:
            training += [sent]
            
    model = ngram_stats(training, development)
    
    avg_log_likelihood = model.avg_log_likelihood(test, 'laplacian')
    perplexity = math.exp(-avg_log_likelihood)
    print(perplexity, avg_log_likelihood)
    
    avg_log_likelihood = model.avg_log_likelihood(test, 'backoff')
    perplexity = math.exp(-avg_log_likelihood)
    print(perplexity, avg_log_likelihood)
    
    avg_log_likelihood = model.avg_log_likelihood(test, 'interpol')
    perplexity = math.exp(-avg_log_likelihood)
    print(perplexity, avg_log_likelihood)
