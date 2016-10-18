from __future__ import print_function
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import numpy
import regex as re
import random
import json

# read JSON file
data_file_name = "Arts.json"
# data_file_name = "Biology.json"
# data_file_name = "Engineering.json" #art 17308 docs
with open(data_file_name, encoding="utf-8") as datafile:
    data = json.load(datafile)

#number of documents
print (len(data))

num_of_testcase=150# testcase number

#random.shuffle(data)
data=data[:num_of_testcase]
print (len(data))
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
    content=docu["Content"].lower()
    #remove stop words and digits and punctuations
    removed_tokens = [i for i in tokenizer.tokenize(content) if i not in stopset and not isContainPorD(i)]
    #stem tokens
    tokens = [p_stemmer.stem(i) for i in removed_tokens]
    #preprocessed texts
    texts.append(tokens)

first_data=[]
second_data=[]
for text in texts:
    first_data.append(text[:len(text)//2])
    second_data.append(text[len(text)//2:])


#load the dictionary
dictionary=corpora.Dictionary.load("full-dict-3categories.dict")
print(len(dictionary))

num_of_unqiuewords=len(dictionary)

def normalizing(texts):
    wordcounts=[]
    for index,text in enumerate(texts):
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

first_data=normalizing(first_data)
second_data=normalizing(second_data)

file=open("data/random_testcase_150art.bin","wb")
numpy.savez(file,first_data=first_data,second_data=second_data)
file.close()
