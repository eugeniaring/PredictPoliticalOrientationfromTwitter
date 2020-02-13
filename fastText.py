import re
import fasttext
import pandas as pd
import numpy as np

newdf75_all=pd.read_csv('df75bc.csv').iloc[:,1:]
newdf1_all=pd.read_csv('df1bc.csv').iloc[:,1:]

############################4 fastText############

#df75
#80% of partecipants in training set and 20% of users in test set
col_names = ['__label__',"tweets"]
train_df75 = pd.DataFrame(columns = col_names)
test_df75 = pd.DataFrame(columns = col_names)
n75=int(newdf75_all.shape[0]*0.8)

train_df75 = newdf75_all.iloc[:n75+1,:]
test_df75 = newdf75_all.iloc[n75+1:,:]

#df1
#80% of partecipants in training set and 20% of users in test set
col_names = ['__label__',"tweets"]
train_df1 = pd.DataFrame(columns = col_names)
test_df1 = pd.DataFrame(columns = col_names)
train_df1 = newdf1_all.iloc[:1785,:]
test_df1 = newdf1_all.iloc[1785:,:]



######Train - Predict - Test

 ### df75
###create train and test file txt for fastText
import csv
train_df75['tweets']= train_df75['tweets'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
test_df75['tweets']= test_df75['tweets'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
train_df75.to_csv('train75.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ") 
test_df75.to_csv('test75.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
#it's okay to have the warning

#train on train data 

#input: the path to our training data
#lr: Learning rate. We set it at 0.01
#wordNgrams: An n-gram is a contiguous sequence of max n words from a given sample of text, tweet here. We set it at 2
#epoch=Number times we go through the entire dataset
#dim: Dimension of word vector. We use 20

model75 = fasttext.train_supervised(input="train75.txt",epoch=1000)
model75.save_model("model_twitter75.bin")

#look if there is a difference between the prediction and test set
test_df75.head(2)

test_df75.iat[1,1]

model75.predict(test_df75.iat[1,1])
#(('__label__0',), array([0.99628234]))

model75.test("test75.txt")
#(473, 0.8752642706131079, 0.8752642706131079)


###################read dfmov5stelle

dfmov75=pd.read_csv('dfmov75cl.csv').iloc[:,1:]
dfmov75.head(10)
#for i in range(dfmov75.shape[0]):
M5_y_predict75=[]
for i in range(dfmov75.shape[0]):
 data= str(dfmov75.iloc[i,1]).replace('\n','')
 M5_y_predict75.append(model75.predict(data)[0][0])


M5_left75=M5_y_predict75.count('__label__0')/len(M5_y_predict75)
M5_right75=M5_y_predict75.count('__label__1')/len(M5_y_predict75)
print('M5S left: {}, M5S right: {}'.format(M5_left75,M5_right75))
#M5S left: 0.6706827309236948, M5S right: 0.3293172690763052


###df1
###create train and test file txt for fastText
import csv
train_df1['tweets']= train_df1['tweets'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
test_df1['tweets']= test_df1['tweets'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
train_df1.to_csv('train1.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ") 
test_df1.to_csv('test1.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
#it's okay to have the warning

#train on train data 

#input: the path to our training data
#lr: Learning rate. We set it at 0.01
#wordNgrams: An n-gram is a contiguous sequence of max n words from a given sample of text, tweet here. We set it at 2
#epoch=Number times we go through the entire dataset
#dim: Dimension of word vector. We use 20

model1 = fasttext.train_supervised(input="train1.txt",epoch=1000)
model1.save_model("model_twitter1.bin")

#look if there is a difference between the prediction and test set
test_df1.head(2)
#           label                                             tweets
#1785  __label__1  b legasalvini \xc3\x88 vergognoso una persona ...
#1786  __label__0  nonha stata marco bello1 ubriaco qualcuno cono...

test_df1.iat[0,1]
model1.predict(test_df1.iat[0,1])

#Out[33]: (('__label__1',), array([1.00001001]))

model1.test("test1.txt")
#(445, 0.6764044943820224, 0.6764044943820224)

####read dfmov5stelle
dfmov1=pd.read_csv('dfmov1cl.csv').iloc[:,1:]

M5_y_predict1=[]
for i in range(dfmov1.shape[0]):
 data= str(dfmov1.iloc[i,1]).replace('\n','')
 M5_y_predict1.append(model1.predict(data)[0][0])


M5_left1=M5_y_predict1.count('__label__0')/len(M5_y_predict1)
M5_right1=M5_y_predict1.count('__label__1')/len(M5_y_predict1)
print('M5S left: {}, M5S right: {}'.format(M5_left1,M5_right1))

#M5S left: 0.6517857142857143, M5S right: 0.3482142857142857
