import tensorflow as tf
import numpy as np
import math
from numpy import linalg as LA

address='glove.6B.50d.txt'
file=open(address,'r',encoding='utf8')
word2vec={}

for line in file:
    spli=line.split()
    word=spli[0]
    representation=spli[1:]
    representation=np.array([float(val) for val in representation])
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
num_filters=100

def CnnModel(decomp_S,decomp_T,max_len,ngram):     #dim(decomp_sent)=no. of words*emd_size*2,2 for S_plus and S_minus
    
    name='F'+str(ngram)
    F=tf.get_variable(name,[2*embedding_size,ngram,1,num_filters],initializer=tf.contrib.layers.xavier_initializer())

    nameb='Fb'+str(ngram)
    Fb=tf.get_variable(nameb,[1,1,1,num_filters],initializer=tf.contrib.layers.xavier_initializer())
    #print(decomp_S.shape)
    Cs=tf.nn.conv2d(decomp_S,F,[1,1,1,1],'VALID')+Fb
    Cs=tf.tanh(Cs)
   # print(Cs.shape)
    Cs= tf.nn.max_pool(Cs,[1,1,max_len-ngram+1,1],[1,1,1,1],'VALID')
    #print(Cs.shape)
    Ct=tf.nn.conv2d(decomp_T,F,[1,1,1,1],'VALID')+Fb
    Ct=tf.tanh(Ct)
    Ct= tf.nn.max_pool(Ct,[1,1,max_len-ngram+1,1],[1,1,1,1],'VALID')
    
    return Cs,Ct

def similarity_score(S_decomp,T_decomp,max_len):   
    
    S_uni,T_uni=CnnModel(S_decomp,T_decomp,max_len,1)
    S_bi,T_bi=CnnModel(S_decomp,T_decomp,max_len,2)
    S_tri,T_tri=CnnModel(S_decomp,T_decomp,max_len,3)
    
    uni=tf.concat([S_uni,T_uni],-1)
    #print(S_uni.shape)
    bi=tf.concat([S_bi,T_bi],-1)
    #print(bi.shape)
    tri=tf.concat([S_tri,T_tri],-1)
    #print(tri.shape)
    feats=tf.concat([uni,bi,tri],-1)
    #print(feats.shape)
    dense1=tf.layers.dense(inputs=feats, units=84,use_bias=True,bias_initializer=tf.zeros_initializer())
    dense2=tf.layers.dense(inputs=dense1,units=1,use_bias=True,bias_initializer=tf.zeros_initializer())
    
    dense2=tf.reshape(dense2,[-1,1])
    pred = dense2
    
    return pred


def random_mini_batch(X1,X2,Y,mini_batch_size=64,seed=0):
    np.random.seed(seed)
    m=int(X1.shape[0])
    mini_batches=[]
    p=list(np.random.permutation(m))
    
    X1_list=[]
    X2_list=[]
    Y_list=[]
    
    for i in range(m):
        X1_list.append(X1[i])
        X2_list.append(X2[i])
        Y_list.append(Y[i])
    
    X1_list=np.array(X1_list)
    X2_list=np.array(X2_list)
    Y_list=np.array(Y_list)
    
    shuffled_X1=X1_list[p]
    shuffled_X2=X2_list[p]
    shuffled_Y=Y_list[p]
    
    num_complete_minibatches=math.floor(m/mini_batch_size)
    for i in range(0,num_complete_minibatches):
        mini_batch_X1=shuffled_X1[i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_X2=shuffled_X2[i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y=shuffled_Y[i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch=(mini_batch_X1,mini_batch_X2,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m%mini_batch_size!=0:
        mini_batch_X1=shuffled_X1[num_complete_minibatches*mini_batch_size:]
        mini_batch_X2=shuffled_X2[num_complete_minibatches*mini_batch_size:]
        mini_batch_Y=shuffled_Y[num_complete_minibatches*mini_batch_size:]
        mini_batch=(mini_batch_X1,mini_batch_X2,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def make_data(file_address):
    file=open(file_address,'r',encoding='utf8')
    max_len=0
    data_set=[]
    labels=[]
    
    for line in file:
        spli=line.split('\t')
        if len(spli)==3 and spli[0]!=None and spli[1]!=None and spli[2]!=None:
            if len(spli[0])<=150 and len(spli[1])<=150:
                S,T,label=spli[0],spli[1],float(spli[2])
                if max(len(S),len(T))>max_len:     
                    max_len=max(len(S),len(T))
                tup=(S,T,label) 
                data_set.append(tup)
                labels.append(label)
    Xs=[]
    Xt=[]
    Y=labels
    #print(len(data_set))
    for t in data_set:
        spm=decompose_sentence(t[0],t[1])
        tpm=decompose_sentence(t[1],t[0])
        
        zeros_req_s=max_len-spm.shape[1]
        zeros_req_t=max_len-tpm.shape[1]
        
        s_c=np.zeros(shape=(2*embedding_size,zeros_req_s))
        t_c=np.zeros(shape=(2*embedding_size,zeros_req_t))
        
        spm=np.concatenate((spm,s_c),axis=1)
        tpm=np.concatenate((tpm,t_c),axis=1)
        
        Xs.append(spm)
        Xt.append(tpm)

    Xs=np.array(Xs)
    m,h,w=Xs.shape
    Xs=np.reshape(Xs,(m,h,w,1))
    
    Xt=np.array(Xt)
    m,h,w=Xt.shape
    Xt=np.reshape(Xt,(m,h,w,1))
    
    Y=np.array(Y)

    return Xs,Xt,Y,max_len,Y.shape[0]


S_decomp,T_decomp,label,max_len,m=make_data('WikiQA-train.txt')

tf.set_random_seed(1)
seed=3

num_epochs=500
minibatch_size=64

tf.reset_default_graph() 

Xs=tf.placeholder(tf.float32,[None,2*embedding_size,max_len,1],name='Xs')
Xt=tf.placeholder(tf.float32,[None,2*embedding_size,max_len,1],name='Xt')
Ys=tf.placeholder(tf.float32,[None,1],name='labels')

scores=similarity_score(Xs,Xt,max_len)
costs=tf.nn.sigmoid_cross_entropy_with_logits(labels=Ys,logits=scores)
cost=tf.reduce_mean(costs)
optimizer =tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()
saver1=tf.train.Saver(max_to_keep=1)
saver2=tf.saved_model.builder.SavedModelBuilder('./model_test2/')
#sess=tf.Session()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        epochcost=0.0
        num_minibatches = int(m//minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batch(S_decomp,T_decomp,label,minibatch_size, seed)

        for minibatch in minibatches:
            (minibatch_X1,minibatch_X2,minibatch_Y) = minibatch
            minibatch_Y=np.reshape(minibatch_Y,(minibatch_Y.shape[0],1))
            _ , temp_cost =sess.run([optimizer,cost],{Xs:minibatch_X1,Xt:minibatch_X2,Ys:minibatch_Y})
            epochcost+=temp_cost
        print('epoch='+str(epoch+1)+' cost='+str(epochcost))
    saver2.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=None,
                                       assets_collection=None)
    saver1.save(sess,"./model_finaltest")
saver2.save()