from nltk.corpus import inaugural
import re
import string
from collections import Counter
import sys
import os  
reload(sys)
 
sys.setdefaultencoding('utf8')
with open("test.txt", "a") as myfile:
	for fileid in inaugural.fileids():
		myfile.write(inaugural.raw(fileid).encode('utf-8').strip())



f1 = open('test.txt', 'r')
f2 = open('x1.txt', 'w')
for line in f1:
    f2.write(line.replace('.', '.<end>').replace('?', '?<end>').replace('!', '!<end>').replace(':', ':<end>'))
f1.close()
f2.close()

f1 = open('x1.txt', 'r')
f2 = open('x2.txt', 'w')
for line in f1:
    f2.write(line.replace('.<end>"', '."<end>').replace('?<end>"', '?"<end>').replace('!<end>"', '!"<end>').replace(':<end>"', ':"<end>').replace('Mr.<end>', 'Mr.').replace('Dr.<end>', 'Dr.').replace('vs.<end>', 'vs.'))
f2.close()
f1.close()

f1 = open('x2.txt', 'r')
f2 = open('x3.txt', 'w')
for line in f1:
    f2.write(line.replace('\n','<end> ').replace('<end><end>','<end> '))
f2.close()
f1.close()


#to make list of sentences
with open('x3.txt', 'r') as myfile:
    sample=myfile.read().replace('\n', '')


sample = re.sub(r"<end>([a-z])", r"\1", sample)

sentences = sample.split('<end>')
sent_count = len(sentences)
print "Total number of sentences = ",sent_count


tok_sent=[]
for sentence in sentences:
	temp1=sentence.lower()
	temp=[word.strip(string.punctuation) for word in temp1.split(" ")]
	temp = filter(None, temp)
	tok_sent.append(temp)


############################################
# question 1 ###############################
all_words = []    #this list contains all the words
for sentence in tok_sent:
	all_words=all_words+ sentence

counts = Counter(all_words)
total_words=sum(counts.values())
word_count = total_words
print "Total number of words = ",word_count
ninety_per=0.9*total_words

temp_int=0
freq_words=[]
ans1=0
for w in sorted(counts, key=counts.get, reverse=True):
  temp_int=temp_int +counts[w]
  if temp_int <= ninety_per:
  	freq_words.append(w)
  	ans1=ans1+1
  else:
  	break
print "Words required for 90% coverage"
print ans1
#print freq_words

############################################
# question2 ################################
bigrams=[]
for sentence in tok_sent:
    for i in range(len(sentence)):
        if i < len(sentence)-1:
            bigrams.append(str(sentence[i]) +" "+ str(sentence[i+1]))

#bigrams = [b for l in tok_sent for b in zip(l[:-1], l[1:])]
eighty_per=0.8*total_words

counts = Counter(bigrams)


temp_int=0
freq_words=[]
ans2=0
for w in sorted(counts, key=counts.get, reverse=True):
  temp_int=temp_int +counts[w]
  if temp_int <= eighty_per:
  	freq_words.append(w)
  	ans2=ans2+1
  else:
  	break
print "Bigrams required for 80% coverage"
print ans2
#print freq_words

#############################################
# question3 #################################
trigrams=[]
for sentence in tok_sent:
    for i in range(len(sentence)):
        if i < len(sentence)-2:
            trigrams.append(str(sentence[i]) +" "+ str(sentence[i+1])+" "+ str(sentence[i+2]))

#bigrams = [b for l in tok_sent for b in zip(l[:-1], l[1:])]
seventy_per=0.7*total_words

counts = Counter(trigrams)


temp_int=0
freq_words=[]
ans3=0
for w in sorted(counts, key=counts.get, reverse=True):
  temp_int=temp_int +counts[w]
  if temp_int <= seventy_per:
  	freq_words.append(w)
  	ans3=ans3+1
  else:
  	break
print "Trigrams required for 70% coverage"
print ans3
#print freq_words


################################################
# question 4 ###################################
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

tok_sent=[]
#print lmtzr.lemmatize('cars')
for sentence in sentences:
	temp1=sentence.lower()
	temp=[]
	for word in temp1.split(" "):
		strp_word=word.strip(string.punctuation)
		lem_word=lmtzr.lemmatize(strp_word).encode('utf-8').strip()
		temp.append(lem_word)
	#temp=[lmtzr.lemmatize(word.strip(string.punctuation)) for word in temp1.split(" ")]
	temp = filter(None, temp)
	tok_sent.append(temp)


all_words = []    #this list contains all the words
for sentence in tok_sent:
	all_words=all_words+ sentence

counts = Counter(all_words)
ninety_per=0.9*total_words

temp_int=0
freq_words=[]
ans1=0
for w in sorted(counts, key=counts.get, reverse=True):
  temp_int=temp_int +counts[w]
  if temp_int <= ninety_per:
  	freq_words.append(w)
  	ans1=ans1+1
  else:
  	break
print "lemetized Words required for 90% coverage"
print ans1
#print freq_words

############################################
# question5 ################################
bigrams=[]
for sentence in tok_sent:
    for i in range(len(sentence)):
        if i < len(sentence)-1:
            bigrams.append(str(sentence[i]) +" "+ str(sentence[i+1]))

#bigrams = [b for l in tok_sent for b in zip(l[:-1], l[1:])]
eighty_per=0.8*total_words

counts = Counter(bigrams)


temp_int=0
freq_words=[]
ans2=0
for w in sorted(counts, key=counts.get, reverse=True):
  temp_int=temp_int +counts[w]
  if temp_int <= eighty_per:
  	freq_words.append(w)
  	ans2=ans2+1
  else:
  	break
print "lemetized Bigrams required for 80% coverage"
print ans2
#print freq_words

#############################################
# question6 #################################
trigrams=[]
for sentence in tok_sent:
    for i in range(len(sentence)):
        if i < len(sentence)-2:
            trigrams.append(str(sentence[i]) +" "+ str(sentence[i+1])+" "+ str(sentence[i+2]))

#bigrams = [b for l in tok_sent for b in zip(l[:-1], l[1:])]
seventy_per=0.7*total_words

counts = Counter(trigrams)


temp_int=0
freq_words=[]
ans3=0
for w in sorted(counts, key=counts.get, reverse=True):
  temp_int=temp_int +counts[w]
  if temp_int <= seventy_per:
  	freq_words.append(w)
  	ans3=ans3+1
  else:
  	break
print "lemetized Trigrams required for 70% coverage"
print ans3
#print freq_words
os.remove('test.txt') 

###################################################
# CHI SQUARE TEST##################################
def chisquare(A1,A2,alpha ):
	w1w2 = A1 + " " +A2
	N=float(len(all_words))
	O11=float(sentences.count(w1w2))
	O21=float(all_words.count(A1)-O11)
	O12=float(all_words.count(A2)-O11)
	O22=float(len(all_words)-O12 -O21 + O11)
	# print O11
	# print O12
	# print O21
	# print O22 
	chisq= float(( float(N) * ((O11*O22) - (O12*O21) ) *((O11*O22) - (O12*O21) ) )/( (O11+O12)*(O11+O21)*(O12+O22)*(O21+O22) ))
	stat_value = {'0.995': 0.000, '0.990': 0.000, '0.975': 0.001, '0.950' : 0.004, '0.900' : 0.016, '0.100':2.706, '0.050':3.841, '0.025':5.024, '0.010':6.635, '0.005':7.879 }
	alp=stat_value[alpha]
	# print chisq
	if chisq >= alp:
		temp=w1w2
		with open("chisq.txt", "a") as myfile:
			myfile.write(temp)
			myfile.write('\n')
	return


alpha='0.050'
print "starting chisquare test"
for idx, val in enumerate(all_words):
	chisquare(all_words[idx],all_words[idx+1],alpha) 
   	
