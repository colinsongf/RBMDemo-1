from __future__ import print_function
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import csv
import numpy
import cPickle as pickle
import regex as re
import random
import json

# read JSON file
data_file_name = "Arts.json" #art 17308 docs
with open(data_file_name, "rb") as datafile:
    data = json.load(datafile)

#number of documents
num_of_doc=len(data)-1
print (num_of_doc)


tokenizer = RegexpTokenizer(r'\w+')
# preprocessed texts
texts=[]
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# Create English stop words
stopset = stopwords.words('english')

#check if word contains digits
def isContainPorD(s):
    return re.search(r'(\d)', s)

#loop through document list
for index,docu in enumerate(data):
    if index>4000:
        break
    content=docu["Content"].lower()
    #remove stop words and digits and punctuations
    removed_tokens = [i for i in tokenizer.tokenize(content) if i not in stopset and not isContainPorD(i)]
    #stem tokens
    tokens = [p_stemmer.stem(i) for i in removed_tokens]
    #preprocessed texts
    texts.append(tokens)


num_of_testcase=50# testcase number
test_data=texts[:num_of_testcase]

input_data=[]
extra_data=[]
for text in test_data:
    # random.shuffle(text)
    input_data.append(text[:len(text)/2])
    extra_data.append(text[len(text)/2:])
train_data=extra_data+texts[num_of_testcase:]

dictionary = corpora.Dictionary(texts)
#save the dictionary
dictionary.save("wiki-dict-4000-txt.txt")

print(len(dictionary))

num_of_unqiuewords=len(dictionary)

def normalizing(texts):
    linedone = 0#flag to see
    wordcounts=[]
    for index,text in enumerate(texts):
        if (index % 1000 == 0):
            linedone += 1
            print(linedone)
        wordcount = numpy.zeros(num_of_unqiuewords)
        for id,counts in dictionary.doc2bow(text):
            wordcount[id]=counts
        wordcounts.append(wordcount)
    # turn our tokenized documents into a id <-> term dictionary
    wordcounts=numpy.array(wordcounts)
    # print(numpy.size(wordcounts))

    #normalization #1 a/sum(row)
    for i,row in enumerate(wordcounts):
        sumvalue=sum(row)
        wordcounts[i]=row/sumvalue
    return wordcounts

input_data=normalizing(input_data)
train_data=normalizing(train_data)

file=open("data-4000.bin","wb")
numpy.savez(file,input_data=input_data,train_data=train_data)
file.close()

#save input result
# newOpenFile=open("processed.txt","w")
# for item in wordcounts:
#     newOpenFile.write("%s\n" % item)
print ("finish!")