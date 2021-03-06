# coding: utf-8

import os
import csv
from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

class ModelClient(object):

    def __init__(self):
        feature_path = os.path.join('data', 'feature0730')
        features = self.read_features(feature_path)
        self.encoded_features = self.encode_features(features)
        with open(os.path.join('model', 'Universities.txt'), 'r') as fr:
            universities = fr.readlines()
        self.universities = [university for university in universities if university and university.strip()]

        host = 'localhost'
        port = 9000
        channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # ----------------------------process features--------------------------
    @staticmethod
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

    @staticmethod
    def one_hot_encode(x,dim):
        y = np.zeros([x.shape[0],dim])
        for i in range(x.shape[0]):
            y[i,int(x[i])] = 1
        return y

    @staticmethod
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
        

    @staticmethod
    def encode_features(features):
        encoded_features = np.zeros([features.shape[0],30],dtype = np.float32)
        encoded_features[:,0:12] = features[:,0:12]
        encoded_features[:,12:20] = ModelClient.one_hot_encode(features[:,12],8)
        encoded_features[:,20:24] = ModelClient.one_hot_encode(features[:,13],4)
        encoded_features[:,24] = features[:,14]
        encoded_features[:,25:29] = ModelClient.encode_scores(features[:,15:19])
        encoded_features[:,29] = features[:,19]
        return encoded_features

    def predict(self, features=None):
        if not features:
            features = self.encoded_features[0:1]
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'test'
        request.inputs['feature'].CopyFrom(tf.contrib.util.make_tensor_proto(features, shape=[1, 30], dtype=np.float32))
        result = self.stub.Predict(request, 10.0)  # 10 secs timeout
        result = result.outputs['score']
        ret = []
        for i in range(len(result.float_val)):
            ret.append(result.float_val[i])
        assert len(ret) == len(self.universities)
        ret = zip(self.universities, ret)
        return ret
