import tensorflow as tf
import numpy as np
import math
from numpy import linalg as LA

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], 'builder_test7_msrparaphrase')
    weights=tf.trainable_variables()
    F1,Fb1=sess.run(weights[0]),sess.run(weights[1])                          # F1=100X1X1X100  Fb1=1X1X1X100
    F2,Fb2=sess.run(weights[2]),sess.run(weights[3])                          # F2=100X2X1X100  Fb2=1X1X1X100
    F3,Fb3=sess.run(weights[4]),sess.run(weights[5])                          # F3=100X3X1X100  Fb3=1X1X1X100
    dense1,dense_bias1=sess.run(weights[6]),sess.run(weights[7])              # dense1=600X84   dense_bias1=84X1
    dense2,dense_bias2=sess.run(weights[8]),sess.run(weights[9])              # dense2=84X1     dense_bias2=1

address='glove.6B.50d.txt'
file=open(address,'r',encoding='utf8')
word2vec={}

for line in file:
            spli = line.split()
            word = spli[0]
            representation = spli[1:]
            representation = np.array([float(val) for val in representation])
            word2vec[word]=representation/LA.norm(representation)

def sentence_to_wordvec_list(sentence):
    sen_to_words=sentence.split()
    word_vec_list=[]
    
    for word in sen_to_words:
        try:
            word_vec=word2vec[word]
            word_vec=np.resize(word_vec,(50,1))
        except KeyError:
            word_vec=word2vec['<unk>']
            word_vec=np.resize(word_vec,(50,1))
        word_vec_list.append(word_vec)
        
    return word_vec_list

def semantic_match_vec(s_vec,t_list):
    #Calculating Global Semantic Matching Vector
    cosine_sim_list=[]
    s_vec_norm=LA.norm(s_vec)
    
    for i in range(len(t_list)):
        t_vec=t_list[i]
        t_vec_norm=LA.norm(t_vec)
        cosine_sim=np.float(np.dot(s_vec.T,t_vec))/s_vec_norm*t_vec_norm
        cosine_sim_list.append(cosine_sim)
    
    numerator=0
    for i in range(len(t_list)):
        numerator+=cosine_sim_list[i]*t_list[i]
    sem_match_vec=numerator/sum(cosine_sim_list)
    
    return sem_match_vec

def decompose_sentence(S,T):
    s_list=sentence_to_wordvec_list(S)
    t_list=sentence_to_wordvec_list(T)
    s_plus=[]
    s_minus=[]
    
    for i in range(len(s_list)):
        sem_match_vec=semantic_match_vec(s_list[i],t_list)
        alpha=float(np.dot(s_list[i].T,sem_match_vec))/np.dot(sem_match_vec.T,sem_match_vec)
        plus_vec=alpha*sem_match_vec
        minus_vec=s_list[i]-plus_vec
        s_plus.append(plus_vec)
        s_minus.append(minus_vec)
    
    sp=np.concatenate((s_plus),axis=1)
    sm=np.concatenate((s_minus),axis=1)
    spm=np.concatenate((sp,sm),axis=0)
    
    return spm

embedding_size=50
num_filters=200

def CnnModel(spm,tpm,ngram,F,Fb,max_len=200):     #dim(decomp_sent)=no. of words*emd_size*2,2 for S_plus and S_minus
    
    Cs=tf.nn.conv2d(spm,F,[1,1,1,1],'VALID')+Fb
    Cs=tf.tanh(Cs)
    Cs=tf.reduce_max(Cs,axis=2)
    Cs=tf.reshape(Cs,[-1,1,1,num_filters])
    
    Ct=tf.nn.conv2d(tpm,F,[1,1,1,1],'VALID')+Fb
    Ct=tf.tanh(Ct)
    Ct=tf.reduce_max(Ct,axis=2)
    Ct=tf.reshape(Ct,[-1,1,1,num_filters])
    
    return Cs,Ct

def similarity_score(spm,tpm,dense1=dense1,dense_bias1=dense_bias1,dense2=dense2,dense_bias2=dense_bias2):   

    S_uni,T_uni=CnnModel(spm,tpm,1,F1,Fb1)
    S_bi,T_bi=CnnModel(spm,tpm,2,F2,Fb2)
    S_tri,T_tri=CnnModel(spm,tpm,3,F3,Fb3)

    uni=tf.concat([S_uni,T_uni],-1)
    bi=tf.concat([S_bi,T_bi],-1)
    tri=tf.concat([S_tri,T_tri],-1)
    
    feats=tf.concat([uni,bi,tri],-1)
    feats=tf.Session().run(feats)
    feats=feats.reshape(-1,1)
    
    bias1=dense_bias1.reshape(-1,1)
    out1=dense1.T.dot(feats)+bias1
    dense_bias2=dense_bias2.reshape(-1,1)
    out2=dense2.T.dot(out1)+dense_bias2
    out2=float(out2)
    
    if out2>=0.50:
        out2=1
    
    else:
        out2=0
    
    return out2

def checkaccuracy(file_address):
    file=open(file_address,'r',encoding='utf8')
    labels=[]
    pred=[]
    k=1
    for line in file:
        spli=line.split('\t')
        if len(spli[0])<=200 and len(spli[1])<=200 and k<=1000:
            S,T,label=spli[3],spli[4],float(spli[0][len(spli[0])-1])
            labels.append(label)
            spm=decompose_sentence(S,T)
            tpm=decompose_sentence(T,S)
            zeros_req_s=200-spm.shape[1]
            zeros_req_t=200-tpm.shape[1]
            s_c=np.zeros(shape=(2*embedding_size,zeros_req_s))
            t_c=np.zeros(shape=(2*embedding_size,zeros_req_t))
            spm=np.concatenate((spm,s_c),axis=1)
            tpm=np.concatenate((tpm,t_c),axis=1)
            h,w=spm.shape
            spm=np.reshape(spm,(-1,h,w,1))
            h,w=tpm.shape
            tpm=np.reshape(tpm,(-1,h,w,1))
            score=similarity_score(spm,tpm)
            pred.append(score)
            k+=1
        else:
          break
    acc=0
    
    for i in range(len(labels)):
        if labels[i]==pred[i]:
            acc+=1
        
    return float(acc)/k

print(checkaccuracy('msr_paraphrase_train.txt'))

