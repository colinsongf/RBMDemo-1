from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import numpy
import regex as re
# read txt file, each line stands for a article
data=[]
data_file_name = "raw/Arts.txt" #art 17308 docs
with open(data_file_name, "r") as datafile:
	for i,line in enumerate(datafile):
		if i==4000:
			break
		data.append(line)
data_file_name="raw/Biology.txt"
with open(data_file_name, "r") as datafile:
	for i,line in enumerate(datafile):
		if i==4000:
			break
		data.append(line)
data_file_name="raw/Engineering.txt"
with open(data_file_name, "r") as datafile:
	for i,line in enumerate(datafile):
		if i==4000:
			break
		data.append(line)
num_of_doc=len(data)
print ("Total doc num is: {}".format(num_of_doc))

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
for docu in data:
	#remove stop words and digits and punctuations
	removed_tokens = [i for i in tokenizer.tokenize(docu) if i not in stopset and not isContainPorD(i)]
	#stem tokens
	tokens = [p_stemmer.stem(i) for i in removed_tokens]
	#preprocessed texts
	texts.append(tokens)

#save dictionary
dictionary = corpora.Dictionary(texts)
print("Dictionary's size: {}".format(len(dictionary)))
dictionary.save("raw/full-dict-3categories.dict")










