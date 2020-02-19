#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import zmq
import zmq.decorators as zmqd
from zmq.utils import jsonapi
import threading
import numpy as np
import cPickle as pickle
from bert_serving.client import BertClient
import gensim 
from segmenter import segmenter

def set_logger(name,verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s-%(levelname)-.1s:' + name + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s',
        datefmt='%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    # console_handler = logging.FileHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger

def cosin_dist(vec,matrix):
    # compute dist vec with each row of matrix
    # dist = np.reshape(np.dot(vec,matrix.transpose()),(-1))/(np.linalg.norm(vec)*np.linalg.norm(matrix,axis=1,keepdims=True))  
    n = len(matrix)
    dist_list = np.zeros(shape=(n),dtype=np.float32)
    for i in xrange(n):
        norm1 = np.linalg.norm(vec)
        norm2 = np.linalg.norm(matrix[i])
        if norm1 == 0 or norm2 == 0:
            dist_list[i] = 0
        else:
            dist_list[i] = np.dot(vec,matrix[i])/(np.linalg.norm(vec)*np.linalg.norm(matrix[i]))
    return dist_list

class BertData(object):
    def __init__(self,query_set_file,bert_vec_file):
        # self.logger = set_logger("test",verbose=True)
        self.query_list,self.vec_list = self.loadQueryVec(query_set_file,bert_vec_file)
        self.vec_list = np.asarray(self.vec_list,dtype=np.float32)
        # self.logger.info("query_list size:%d,query_vec size:%s"%(len(self.query_list),str(self.vec_list.shape)))

    def loadQueryVec(self,query_set_file,bert_vec_file):
        query_list = []
        vec_list = []
        with open(query_set_file) as fp:
            lines = fp.readlines()
            for line in lines:
                query_list.append(line.strip('\n'))

        # with open(bert_vec_file) as fp:
            # lines = fp.readlines()
            # for line in lines:
                # vec_list.append(np.asarray(line.strip().split('[')[1][0:-1].split(','),dtype=np.float32))
        with open(bert_vec_file,'rb') as fp:
            vec_list = pickle.load(fp)
        return query_list,vec_list

    def knn(self,query_vec,k):
        query_vec = np.asarray(query_vec,dtype=np.float32)
        # dist = np.reshape(np.dot(query_vec,self.vec_list.transpose()),(-1))/(np.linalg.norm(query_vec)*np.linalg.norm(self.vec_list,axis=1,keepdims=True))  
        dists = cosin_dist(query_vec,self.vec_list)
        sorted_indexs = np.argsort(dists)  
        knn_sen = []
        for i in range(0,k):
            knn_sen.append(self.query_list[sorted_indexs[-i-1]])
        return knn_sen
            
class W2VData(object):
    def __init__(self,w2v_file):
        # self.logger = set_logger("test",verbose=True)
        self.w2v = gensim.models.Word2Vec.load(w2v_file)
        # self.logger.info("query_list size:%d,query_vec size:%s"%(len(self.query_list),str(self.vec_list.shape)))
    
    def get_maxpooling_vec(self,str):
        items = segmenter.segment(str)
        vectors = [] 
        for item in items:
            if item in self.w2v:
                vectors.append(self.w2v[item])
        if len(vectors) == 0:
            return np.zeros([200,])
        vectors = np.asarray(vectors)
        res = vectors.max(axis=0)
        return res

    def calCosinDist(self,target,sentence):
        target_vec = self.get_maxpooling_vec(target.encode('gb18030'))
        sentence_vec = self.get_maxpooling_vec(sentence.encode('gb18030'))
        norm1 = np.linalg.norm(target_vec)
        norm2 = np.linalg.norm(sentence_vec)
        if norm1 != 0 and norm2 != 0:
            cos_res = np.dot(target_vec,sentence_vec)/(np.linalg.norm(target_vec)*np.linalg.norm(sentence_vec))
        else:
            cos_res = 0.0
        return cos_res



class SenSimServer(threading.Thread):
    def __init__(self, args):
        super(SenSimServer,self).__init__()
        self.logger = set_logger("test",verbose=True)
        self.port = args["port"]
        # self.bertdata = args["bertdata"]
        self.w2vdata = args["w2vdata"]
        self.args =args

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.process:
            p.close()
        self.join()

    def run(self):
        self._run()


    @zmqd.context()
    @zmqd.socket(zmq.REP)
    def _run(self,_,backend_sock):
        backend_sock.bind('tcp://127.0.0.1:%d' % self.port)
        self.logger.info("bind server socket:%s:%d"%("tcp://127.0.0.1",self.port))
        bc = BertClient(ip='127.0.0.1')
        while True:
            try:
                message = backend_sock.recv_json()
            except Exception:
                self.logger.error("recv message error")
                continue
            else:
                target = message["target"]
                sentences = message["sentences"]
                self.logger.info("target:%s"%(target))
                if target == "stop":
                    self.logger.info("stop server")
                    break
                if target == "":
                    continue
                index = {}
                bert_seq = [target]
                k = 0
                for i,item in enumerate(sentences):
                    if item == "":
                        continue
                    else:
                        bert_seq.append(item)
                        index[k] = i
                        k += 1
                bert_vecs = bc.encode(bert_seq)  
                bert_dist = cosin_dist(bert_vecs[0],bert_vecs[1:])
                bert_res = [0] * 5 
                for i,d in enumerate(bert_dist):
                    bert_res[index[i]] = str(d)

                w2v_res = []
                for sen in sentences:
                    d = self.w2vdata.calCosinDist(target,sen)
                    w2v_res.append(str(d))
                backend_sock.send_json(jsonapi.dumps({"sim_res":[bert_res,w2v_res]}))

        backend_sock.close()
        self.logger.info("stop")

def startService():
    args = {"port":5679}
    w2v_file = "/search/odin/workspace/querySemantic/data/wordRepresentation/click2vec/retrain_word2click_vectors_v2"
    # args["bertdata"] = BertData(query_set_file,query_vec_file_bert)
    args["w2vdata"] = W2VData(w2v_file)
    server = SenSimServer(args)
    server.start()
    server.join()
        

if __name__ == "__main__":
    startService()
    # query_set_file = "/search/odin/workspace/querySemantic/data/show/query_set"
    # query_vec_file = "/search/odin/workspace/querySemantic/data/show/query_vec"
    # bertdata = BertData(query_set_file,query_vec_file)
    # query_set_file = "/search/odin/workspace/querySemantic/data/show/query_set"
    # query_vec_file = "/search/odin/workspace/querySemantic/data/show/query_vec_maxpooling"
    # w2v_file = "/search/odin/workspace/querySemantic/data/wordRepresentation/click2vec/retrain_word2click_vectors_v2"
    # w2vdata = W2VData(query_set_file,query_vec_file,w2v_file)

