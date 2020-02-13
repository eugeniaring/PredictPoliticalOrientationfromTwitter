#######################1 Load dataframes##############################################
import re
import fasttext
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

df75=pd.read_csv('df75.csv').iloc[:,1:]
df75.head()

df1=pd.read_csv('df1.csv').iloc[:,1:]
df1.head()

###################2 cleaning################################

#####df75


#######################df75################################################

import re
#remove urls
df75["tweets"] = df75["Full_text"].apply(lambda s: ' '.join(re.sub("(\w+:\/\/\S+)", " ", s).split()))

#remove punctuations
df75["tweets"] = df75["tweets"].apply(lambda s: ' '.join(re.sub("[\.\,\!\?\:\;\-\=\'\...\"\@\#\_]", " ", s).split()))

#lower
df75["tweets"] = df75["tweets"].apply(lambda s: s.lower())

#remove emoji since the package translates to english, too lazy to make custom
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

df75["tweets"] = df75["tweets"].apply(lambda s: deEmojify(s))

 #---------Removal of stopwords in italian------
from nltk.corpus import stopwords
stop = set(stopwords.words('italian'))

# Code to remove noisy words from a text

def _remove_noise(input_text):
    words = input_text.lower()
    words = words.split()
    noise_free_words = [word for word in words if word not in stop] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

#remmove stopwords
df75["tweets"] = df75["tweets"].apply(lambda s: _remove_noise(s))

# text_without_stopwords = _remove_noise("@Corriere Grandi insegnamenti, è dalle basi che si costruisce un paese, no da 4 pilastri ignoranti.")

#---------Removal of stopwords in english------
from nltk.corpus import stopwords
stop = set(stopwords.words('english')) 

def rem_en(input_txt):
    words = input_txt.lower()
    words = words.split()
    noise_free_words = [word for word in words if word not in stop] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

#remove stopwords
df75["tweets"] = df75["tweets"].apply(lambda s: rem_en(s))

#####count number of users in left and right party
df75['Party'].value_counts()
#1    2432
#0    1185
#Name: Party, dtype: int64

###########df1

##################################2 Cleaning ###########################################àà

#######################df75################################################

df1["tweets"] = df1["Full_text"].apply(lambda s: ' '.join(re.sub("(\w+:\/\/\S+)", " ", s).split()))

#remove punctuations
df1["tweets"] = df1["tweets"].apply(lambda s: ' '.join(re.sub("[\.\,\!\?\:\;\-\=\'\...\"\@\#\_]", " ", s).split()))

#lower
df1["tweets"] = df1["tweets"].apply(lambda s: s.lower())

#remove emoji since the package translates to english, too lazy to make custom
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

df1["tweets"] = df1["tweets"].apply(lambda s: deEmojify(s))

 #---------Removal of stopwords in italian------

# Code to remove noisy words from a text

#remove stopwords
df1["tweets"] = df1["tweets"].apply(lambda s: _remove_noise(s))

#---------Removal of stopwords in english------
from nltk.corpus import stopwords
stop = set(stopwords.words('english')) 

#remove stopwords
df1["tweets"] = df1["tweets"].apply(lambda s: rem_en(s))


#####count number of users in left and right party

df1['Party'].value_counts()
#1    2059
#0    1115
#Name: Party, dtype: int64



#######################3 undersampling (only for left and right, not for movimento5stelle)

######undersampling

#################df75###################################

df_left = df75[df75['Party']==0]
df_right = df75[df75['Party']==1]

col_names = ['label', 'tweets']
newdf75_all = pd.DataFrame(columns = col_names)

#take 530 users of left party and 560 of right party
for i in range(1185):
        newdf75_all  = newdf75_all.append({'label': '__label__0',
                                     'tweets': df_left.iat[i,3]}, ignore_index = True)
        newdf75_all  = newdf75_all.append({'label': '__label__1',
                                     'tweets': df_right.iat[i,3]}, ignore_index = True)
newdf75_all.shape
#(2370, 2)

newdf75_all['label']=newdf75_all['label'].astype('category')

newdf75_all.info()

newdf75_all.to_csv('df75bc.csv')


################df1###################################

#################df1###################################

df_left = df1[df1['Party']==0]
df_right = df1[df1['Party']==1]

col_names = ['label', 'tweets']
newdf1_all = pd.DataFrame(columns = col_names)

#take 530 users of left party and 560 of right party
for i in range(1115):
        newdf1_all  = newdf1_all.append({'label': '__label__0',
                                     'tweets': df_left.iat[i,3]}, ignore_index = True)
        newdf1_all  = newdf1_all.append({'label': '__label__1',
                                     'tweets': df_right.iat[i,3]}, ignore_index = True)
newdf1_all.shape
#(2230, 2)

newdf1_all['label']=newdf1_all['label'].astype('category')
newdf1_all.info()

newdf1_all.to_csv('df1bc.csv')

###################bar plots

#df75 
import matplotlib.pyplot as plt

ax=newdf75_all.groupby('label').tweets.count().plot.bar(ylim=(0, 2000),color=['red','green'])
ax.set_xticklabels(['left','right'])
plt.title('Agreement 75%')
plt.xlabel('Party')
plt.ylabel('Number of users for party')
plt.show()

#df1

#barplot 

ax=newdf1_all.groupby('label').tweets.count().plot.bar(ylim=(0, 2000),color=['red','green'])
ax.set_xticklabels(['left','right'])
plt.title('Agreement 100%')
plt.xlabel('Party')
plt.ylabel('Number of users for party')
plt.show()
