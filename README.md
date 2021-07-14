# Analysis-Sentiment
This is my final task to get bachelor degree. This system can classify sentiment base on word frequents. Method that I use is Support Vector Machine

## The Flow System

![Arsitektur - Copy-alur sistem](https://user-images.githubusercontent.com/71368358/125415212-9e47af68-4291-4bd0-a70e-0cdfcfcb279c.jpg)

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

![Arsitektur - Copy-Contoh Preprocessing](https://user-images.githubusercontent.com/71368358/125415356-a359d934-66c1-4480-a888-4d104560b514.jpg)

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

- The formula for Term Frequency

![1_RL68ajkTXoWnUaK3hOLTew](https://user-images.githubusercontent.com/71368358/125415900-970972d3-5bf6-41b5-851f-44943061bbcb.png)

(source : https://miro.medium.com/max/875/1*RL68ajkTXoWnUaK3hOLTew.png)

- The formula for Inverse Document Frequency

![1_b8sxMQwGBH75DQkMPekvCw](https://user-images.githubusercontent.com/71368358/125416201-33cb9b2d-fd1c-4664-a7af-db82f2ee39ed.png)

(source : https://miro.medium.com/max/853/1*b8sxMQwGBH75DQkMPekvCw.png)

After calculate the TF and IDF, we must multiply both of them

## Training.py

The code in this file is use to make model of machine learning. Machine learning method that I used is Support Vector Machine

## Testing.py

The code in this file is use to test the model of machine learning and I try to make algorithm that the system can show user the possibility sentiment when user input some sentence but the result is must to try alot
