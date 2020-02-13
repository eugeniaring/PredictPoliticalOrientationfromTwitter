#################################CREATION DATASETS################################Ã 

############only right and left##############

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from langdetect import detect

#read file task23
accounts_df = pd.read_excel("LabelledAccounts.xlsx",header=1)
accounts_df.head()
ac=accounts_df.iloc[:,[1,11,12,13,14,15,16]]
ac.head()

def filter_party(df,n_agree):
    lusers,lp,mov5stars,unclass=[],[],[],[]
    for i in range(df.shape[0]):
      #0==missing data  
      if df.iloc[i,1]==1:
        unclass.append(df.iloc[i,0])
      #1==Lega  --> Lega=1 right
      elif (df.iloc[i,2]>n_agree)*(n_agree==0.75) or (df.iloc[i,2]==n_agree):                 
        lusers.append(df.iloc[i,0])
        lp.append('1')
      #2==PD--> PD=0 left
      elif (df.iloc[i,3]>n_agree)*(n_agree==0.75) or (df.iloc[i,3]==n_agree):                 
        lusers.append(df.iloc[i,0])
        lp.append('0')
      #3==FDI --> FDI=1 right
      elif (df.iloc[i,4]>n_agree)*(n_agree==0.75) or (df.iloc[i,4]==n_agree):                 
        lusers.append(df.iloc[i,0])
        lp.append('1')
      #4 = m5s --> list
      elif (df.iloc[i,5]>n_agree)*(n_agree==0.75) or (df.iloc[i,5]==n_agree):                 
        mov5stars.append(df.iloc[i,0]) 
      #5=FI  --> FI=1 right
      elif (df.iloc[i,6]>n_agree)*(n_agree==0.75) or (df.iloc[i,6]==n_agree):
        lusers.append(df.iloc[i,0])
        lp.append('1')
    return lusers,lp,mov5stars,unclass   

#create list with tweets and dates, that I need for the dataframe
#I want the tweets between (tw["created_at"] > '2019-06-01', tw["created_at"] < '2019-12-03')
#I check if the user has @ in the beggining or not

import os

#twitter_users function check if @ in the file and changes the string if there isn't 
def twitter_users(u):
    if os.path.exists('.\Tweets\%s_tweets.csv'%u)==True:
        return u
    elif os.path.exists('.\Tweets\%s_tweets.csv'%u[1:])==True:
        return u[1:]
    else:
        return u

#filter between september and december 
#add number of tweets, ntw, as field        
def filter_tweets(ul,pl):
    path='.\Tweets'
    tweets,t_id,pa,lusersrem,ntw=[],[],[],[],[]
    for i in range(len(ul)):
       #function twitter_users 
       t= twitter_users(ul[i])
       if os.path.exists('.\Tweets\%s_tweets.csv'%t)==True:
            tw=pd.read_csv('.\Tweets\%s_tweets.csv'%t)
            tw=tw.loc[np.logical_and(tw["created_at"] >= '2019-06-01', tw["created_at"] <= '2019-12-01')]
            #errore:   if len(list(tw.full_text))>=10 and len(list(tw.full_text))<=200:
            if len(list(tw.full_text))>0:
             l=list(tw.full_text)
             ntw.append(len(l))
             tweets.append(' '.join(l))
             t_id.append(t)
             pa.append(pl[i]) 
            else:
              lusersrem.append(t)
       else:
           lusersrem.append(t)
            
    return tweets,t_id,pa,ntw 



##############################################

#75% agreement

lusers75,lp75,mov5stars75,uncl75=filter_party(ac,0.75)
print(len(lusers75),len(lp75))

tweets75,t_id75,pa75,ntw75=filter_tweets(lusers75,lp75)
print(len(tweets75),len(t_id75),len(pa75))

df75=pd.DataFrame({'Party':pa75,'Full_text':tweets75,'tweets_count':ntw75})
df75['Party']=df75['Party'].astype('category')
df75.info()
#memory usage: 61.7+ KB without t_id75
#memory usage: 90.6+ KB with t_id75

df75.head(30)
df75.describe()

############delete users with tweets in other languages

'''
r=df75.shape[0]
l=[]
for i in range(r):
    if detect(df75.Full_text[i][:100])!='it':
        l.append(False)
    else:
        l.append(True)
'''


print('Dataset size:',df75.shape)
print('Columns are:',df75.columns)

df75['Party'].value_counts()
#1    2432
#0    1185
#Name: Party, dtype: int64


#barplot 
ax=df75.groupby('Party').Full_text.count().plot.bar(ylim=(0, 3000),color=['red','green'])
ax.set_xticklabels(['left','right'])
plt.title('Agreement 75%')
plt.ylabel('Number of users for party')
plt.show()

#boxplot of Number of Tweets for party
b=sns.catplot(x='Party',y='tweets_count',palette=['red','green'],kind='box',data=df75)
b.fig.suptitle('75% agreement')
b.set(ylim=(10, 250))
(b.set_axis_labels("", "Number of Tweets for party").set_xticklabels(['red','green']).despine(left=True))     



#####################################


#100% agreement

lusers1,lp1,mov5stars1,uncl1=filter_party(ac,1)
print(len(lusers1),len(lp1))

tweets1,t_id1,pa1,ntw1=filter_tweets(lusers1,lp1)
print(len(tweets1),len(t_id1),len(pa1))

df1=pd.DataFrame({'Party':pa1,'Full_text':tweets1,'tweets_count':ntw1})
#convert type Party, from object to category(goal: use less memory)
df1['Party']=df1['Party'].astype('category') 
df1.head(30)
df1.info()

df1.describe()

print('Dataset size:',df1.shape)
print('Columns are:',df1.columns)

df1['Party'].value_counts()
#1    2059
#0    1115
#Name: Party, dtype: int64

#barplot
ax=df1.groupby('Party').Full_text.count().plot.bar(ylim=(0, 3000),color=['red','green'])
ax.set_xticklabels(['left','right'])
plt.title('Agreement 100%')
plt.ylabel('Number of users for party')
plt.show()

#boxplot
b=sns.catplot(x='Party',y='tweets_count',palette=['red','green'],kind='box',data=df1)
b.fig.suptitle('100% agreement')
b.set(ylim=(10, 250))
(b.set_axis_labels("", "Number of Tweets for party").set_xticklabels(['red','green']).despine(left=True))


##############export

df75.to_csv('df75.csv')
df1.to_csv('df1.csv')

