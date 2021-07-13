# Analysis-Sentiment
This is my final task to get bachelor degree. This system can classify sentiment base on word frequents. Method that I use is Support Vector Machine

## The Flow System



## Analysis Sentimen.py

The code in this file is use for build GUI. The GUI is use PyQt6 to make simple. For the desain, we can use QtDesigner to make simple build

## Connection.py

The code in this file is use for connection to database. In this case I use MySQL database to save the data. The data that save in database there are:
- Stopword Dictionary
  
  For the data stopword, I get from library sotpword that Pyhton have
- Alay Dictionary
  
  For the data alay, I get from the last research. The resource https://github.com/nasalsabila/kamus-alay/blob/master/colloquial-indonesian-lexicon.csv
- Result the trainning machine learning

## Preprocessing.py

The code in this file is use for preprocessing text before the machine learning to classification. The library that I use is Sastrawi. The process of preprocessing are:
- Tokenizing
  
  In this step the system will remove unnecessary likes Usertag, Hastag, URL, and punctuation

- Languange Normalization

  In this step the system will normalize the term in tweet. In Indonesia there are a lot of language, so we need normalize the term in tweet to standar term

- Filtering

  In this step the system use stopword library in Indonesia. This step will remove the term that unnecessary for analysis sentiment
  
- Stemming

  In this step the system use Sastrawi library to change the term to basic term

## TFIDF.py

The code in this file is use to vectorization the term. This system use Term Frequency - Inverse Document Frequency (TF-IDF) algorithm
