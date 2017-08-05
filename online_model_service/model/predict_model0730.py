import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf


def get_median(x):
    l = len(x[0])
    if(l>0):
        return x[0][l/2]
    if(l<=0):
        return -1

# process features
def read_features(path):   
    feature_list = []
    for i in range(16):
        data_path = os.path.join(path,'featurestest%d.csv'%(i))
        with open(data_path, 'rb') as f: 
            reader = csv.reader(f)
            for row in reader:
                feature_list.append(row)
                
    features = [[float(i) for i in j] for j in feature_list[0:]]
    features = np.array(features)
    features = features.astype(np.float32)
    
    return features
    
def read_labels(path):
    label_list = []
    for i in range(16):
        label_path = os.path.abspath('labels_ordered/label_test%d.csv'%(i))
        with open(label_path, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                label_list.append(row)

    labels = [[int(i) for i in j] for j in label_list[0:]]
    labels = np.array(labels)
    labels = labels.astype(np.float32)
    
    return labels

def one_hot_encode(x,dim):
    y = np.zeros([x.shape[0],dim])
    for i in range(x.shape[0]):
        y[i,int(x[i])] = 1
    return y

def encode_scores(scores):
    encoded_scores = np.array(scores)
    scores[np.where(scores[:,0]>120),0] = -1
    scores[np.where(scores[:,1]>120),1] = -1
    scores[np.where(scores[:,2]>350),2] = -1
    scores[np.where(scores[:,3]>800),3] = -1
    
    gpa = scores[np.where(scores[:,0]>0),0][0]
    tofel = scores[np.where(scores[:,1]>0),1][0]
    gre = scores[np.where(scores[:,2]>0),2][0]
    gmat = scores[np.where(scores[:,3]>0),3][0]

    gpa_mean = gpa.mean()
    tofel_mean = tofel.mean()
    gre_mean = gre.mean()
    gmat_mean = gmat.mean()

    gpa_std = gpa.std()
    tofel_std = tofel.std()
    gre_std = gre.std()
    gmat_std = gmat.std()
    
    print('gpa_mean = %f'%gpa_mean)
    print('tofel_mean = %f'%tofel_mean)
    print('gre_mean = %f'%gre_mean)
    print('gmat_mean = %f'%gmat_mean)
    print('gpa_std = %f'%gpa_std)
    print('tofel_std = %f'%tofel_std)
    print('gre_std = %f'%gre_std)
    print('gmat_std = %f'%gmat_std)
    with open("scores_param.txt","w") as f:
        f.write('gpa_mean = %.6f, gpa_std = %.6f\n'%(gpa_mean,gpa_std))
        f.write('tofel_mean = %.6f, tofel_std = %.6f\n'%(tofel_mean,tofel_std))
        f.write('gre_mean = %.6f, gre_std = %.6f\n'%(gre_mean,gre_std))
        f.write('gmat_mean = %.6f, gmat_std = %.6f\n'%(gmat_mean,gmat_std))

    encoded_scores[:,0] = (scores[:,0] - gpa_mean)/gpa_std
    encoded_scores[:,1] = (scores[:,1] - tofel_mean)/tofel_std
    encoded_scores[:,2] = (scores[:,2] - gre_mean)/gre_std
    encoded_scores[:,3] = (scores[:,3] - gmat_mean)/gmat_std
    encoded_scores[np.where(encoded_scores<-3)] = -1
    
    return encoded_scores
    

def encode_features(features):
    encoded_features = np.zeros([features.shape[0],30],dtype = np.float32)
    encoded_features[:,0:12] = features[:,0:12]
    encoded_features[:,12:20] = one_hot_encode(features[:,12],8)
    encoded_features[:,20:24] = one_hot_encode(features[:,13],4)
    encoded_features[:,24] = features[:,14]
    encoded_features[:,25:29] = encode_scores(features[:,15:19])
    encoded_features[:,29] = features[:,19]
    
    return encoded_features

def encode_labels(labels):
    encoded_labels = np.zeros(labels.shape,dtype=np.float32)
    for i in range (labels.shape[0]):
        temp = np.where(labels[i]>0)[0]
        if temp.size==0:
            continue
        mean = temp.mean()
        if temp.size==1:
            std = 30
        else:
            std = temp.std()
        for j in range(labels.shape[1]):
            encoded_labels[i,j] = norm.cdf((j-mean)/std)
            if np.isnan(encoded_labels[i,j]):
                print(i,j,mean,std)
    return encoded_labels


# ------------------------------------------------------------------------------
# ---------------------------- preparation ----------------------------
print('loading features...')
features = read_features(os.path.join('..', 'data', 'feature0730'))
print('loading labels...')
labels = read_labels(os.path.join('..', 'data', 'labels_ordered'))

print('encoding features...')
encoded_features = encode_features(features)
print('encoding labels... Will take a long time, please wait.')
encoded_labels = encode_labels(labels)

# split train and test set
train_feature = encoded_features[:6500]
train_label = encoded_labels[:6500]
test_feature = encoded_features[6500:]
test_label = encoded_labels[6500:]


# ---------------------------- train model ----------------------------
print('training...')
sess = tf.Session()

# keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
x = tf.placeholder(tf.float32, shape=(None, train_feature.shape[1]))
dense1 = tf.layers.dense(x,128,activation=tf.nn.relu)
# dense1 = tf.nn.dropout(dense1,keep_prob)
dense2 = tf.layers.dense(dense1,128,activation=tf.nn.relu)
# dense2 = tf.nn.dropout(dense2,keep_prob)
y_ = tf.layers.dense(dense2,train_label.shape[1],activation=tf.nn.sigmoid)

y = tf.placeholder(tf.float32, shape=(None, train_label.shape[1]))

loss = tf.reduce_mean(tf.losses.mean_squared_error(y,y_))
train_step = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

with sess.as_default():
    epoch = 1000
    for i in range(100 * epoch):
        start = (i % 100) * 50
        end = start + 50
        train_feature_batch = train_feature[start:end]
        train_label_batch = train_label[start:end]
        sess.run(train_step,feed_dict={x: train_feature_batch,
                                       y: train_label_batch})
                                      # keep_prob: 0.75})
        
        if i % 1000==0:
            mse = sess.run(loss, feed_dict={x: train_feature_batch, y: train_label_batch})
            	# , keep_prob: 1.0})
            test_mse = sess.run(loss, feed_dict={x: test_feature, y: test_label})
            	# , keep_prob: 1.0})
            print("epoch " + str(i/100) + ", Minibatch Loss= " + "{:.6f}".format(mse) 
                  + ", test set mse= " + "{:.6f}".format(test_mse))


# ---------------------------- save model ----------------------------  
from tensorflow.contrib.session_bundle import exporter
saver = tf.train.Saver()
model_dir = os.path.abspath('./model')
model_exporter = exporter.Exporter(saver)
model_version = 1
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'feature': x}),
        'outputs': exporter.generic_signature({'score': y_})})
model_exporter.export(model_dir,         
                      tf.constant(model_version),
                      sess)