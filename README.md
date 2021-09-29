# project_it_tmsl
FOLLOWING ARE THE STEPS USED IN CREATING THE MODEL: -
1) Import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

2) STEPS TO CLEAN THE REVIEWS: -
  a) Remove HTML Tags
  def clean(text):
   cleaned = re.compile(r'<.*?>')
   return re.sub(cleaned,'',text)
  data.review = data.review.apply(clean)
  data.review[0]
  b) Remove Special Characters
  def is_special(text):
   rem = ''
   for i in text:
   if i.isalnum():
   rem = rem + i
   else:
   rem = rem + ' '
   return rem
  data.review = data.review.apply(is_special)
  data.review[0]
  c) Convert Everything to Lowercase
  def to_lower(text):
   return text.lower()
  data.review = data.review.apply(to_lower)
  data.review[0]
  d) Remove Stopwords
  def rem_stopwords(text):
   stop_words = set(stopwords.words('english'))
   words = word_tokenize(text)
   return [w for w in words if w not in stop_words]
  data.review = data.review.apply(rem_stopwords)
  data.review[0]
  e) Stem the Words
  def stem_txt(text):
   ss = SnowballStemmer('english')
   return " ".join([ss.stem(w) for w in text])
  data.review = data.review.apply(stem_txt)
  data.review[0]
3) CREATING THE MODEL: -
  a) Creating Bag of Words
  b) Train Test Split
  c) Defining the Models and Training them
  d) Prediction and Accuracy Metrices
