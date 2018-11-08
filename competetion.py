from gensim.models import word2vec,KeyedVectors
import pandas as pd
import jieba
import codecs, sys
import numpy as np
import tensorflow as tf
import random


def delstopword(wordList, stopkey):
    sentence = []
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            sentence.append( word)
    return sentence


def getRawData(path):
    data = pd.read_csv( path,encoding='utf-8' )
    print(data.columns)
    print(data.index)
    rawdata = data.loc[:,['content']]
    rawdata = rawdata.values
    Y = data.iloc[:, 2]#location_distance_from_business_district
    for i in range(20):
        print(Y[i])
    Y = Y.values + 2
    print(type(rawdata), "trainY:", type(Y), " shape:", Y.shape, "old Y:", Y[0])
    print(" Y len=",len(Y))
    Y = Y.flatten()
    Y=np.eye(4)[Y]
    print("rawdata0::",rawdata[0])
    print("rawdata.shape:", rawdata.shape)
    words = []
    print("---------------"*20)
    stopkey = [w.strip() for w in codecs.open('stopwords/hagongda.txt', 'r', encoding='utf-8').readlines()]
    for sentence in rawdata:
        # seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
        # print(list(seg_list))
        # print(sentence)
        # print("type(sentense)=",type(sentence.values))
        # print("sentence.type=", type(sentence[0]))
        # print("type('''我来到北京清华大学'''=", type("我来到北京清华大学"))
        cutwords = jieba.cut(sentence[0] ,cut_all=False)
        cutwords = list(cutwords)
        newSentence = delstopword(cutwords, stopkey)
        words.append(newSentence)
    print("words::", words[0])
    return words,Y

max_length = 0
longest_element = 0
rows=0


def dataPrepare( path):
    sentences,Y = getRawData(path)

    print("type of sentences: ", type(sentences))
    # model = word2vec.Word2Vec.load()

    # model = word2vec.Word2Vec(sentences, size=100,min_count=1,workers=4)
     # model=word2vec.Word2Vec.save_word2vec_format("data/w2v.model",binary=False)
    # model = KeyedVectors.load_word2vec_format("data/w2v.model", binary=False)
    # model.load()

    model.train(sentences,total_examples=len(sentences),epochs=2)
    print("train word 2 vector")
    model.save("mymodel")
    # mode = word2vec.Word2Vec.load_word2vec_format()
    # model.save_word2vec_format("data/vectors.txt",binary=False)
    max_length, longest_element = max([(len(x), x) for x in sentences])
    print("train word 2 vector")
    rows = len(sentences)
    print("max_length=", max_length, "longest_element=", longest_element)
    print("word to vector , mapping...")
    X=[]
    for sentence in sentences:
        line = []
        for word in sentence:
            line.append(model[word])
        X.append(line)
    return X,Y


model =word2vec.Word2Vec.load("mymodel")

trainX,trainY = dataPrepare("data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv")
print("type(trainX)=",type(trainX)," type(trainY)=",type(trainY))
print("　　　　　　　　　x:",trainX[0])
print("　　　　　　　　　y:",trainY[0])

valiX,valiY = dataPrepare("data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
print("type(valiX)=",type(valiX)," type(valiY)=",type(valiY))
print("　　　　　　　　　x:",valiX[0])
print("　　　　　　　　　y:",valiY[0])

# for sentence in sentences:
# print(model['大众'])
# print( model.most_similar("大众"))
x = tf.placeholder(tf.float32, [None,max_length*100 ])
W = tf.Variable(tf.zeros([max_length*100,4]))
b = tf.Variable(tf.zeros([4]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,4])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#梯度下降法；

init = tf.global_variables_initializer()#初始化变量；
sess = tf.Session()
sess.run(init)

for i in range(rows//100):
    index = random.randint(0, rows-1)
    start = index
    if( index+100 > rows):
        start -=100
    end = start+100
    sess.run(train_step, feed_dict={x: trainX[start:end], y_: trainY[start:end]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: valiX, y_: valiY}))

TP = tf.count_nonzero(y * y_, 0)
print (sess.run(TP, feed_dict={x: valiX, y_: valiY}))
FP = tf.count_nonzero(y * (y_ - 1), 0)
print (sess.run(FP, feed_dict={x: valiX, y_: valiY}))
FN = tf.count_nonzero((y - 1) * y_, 0)
print (sess.run(FN, feed_dict={x: valiX, y_: valiY}))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print("precision=",precision, "   recall=",recall, "  f1=",f1)
