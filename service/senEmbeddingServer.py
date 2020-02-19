#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import zmq
import zmq.decorators as zmqd
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
    def __init__(self,query_set_file,query_vec_file,w2v_file):
        # self.logger = set_logger("test",verbose=True)
        self.query_list,self.vec_list = self.loadQueryVec(query_set_file,query_vec_file)
        self.vec_list = np.asarray(self.vec_list,dtype=np.float32)
        self.w2v = gensim.models.Word2Vec.load(w2v_file)
        # self.logger.info("query_list size:%d,query_vec size:%s"%(len(self.query_list),str(self.vec_list.shape)))

    def loadQueryVec(self,query_set_file,query_vec_file):
        query_list = []
        vec_list = []
        with open(query_set_file) as fp:
            lines = fp.readlines()
            for line in lines:
                query_list.append(line.strip('\n'))

        with open(query_vec_file,'rb') as fp:
            vec_list = pickle.load(fp)
        return query_list,vec_list
    
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

    def knn(self,query,k):
        query_vec = self.get_maxpooling_vec(query)
        # dist = np.reshape(np.dot(query_vec,self.vec_list.transpose()),(-1))/(np.linalg.norm(query_vec)*np.linalg.norm(self.vec_list,axis=1,keepdims=True))  
        dists = cosin_dist(query_vec,self.vec_list)
        sorted_indexs = np.argsort(dists)  
        knn_sen = []
        for i in range(0,k):
            knn_sen.append(self.query_list[sorted_indexs[-i-1]])
        return knn_sen

class SenEmbServer(threading.Thread):
    def __init__(self, args):
        super(SenEmbServer,self).__init__()
        self.logger = set_logger("test",verbose=True)
        # self.num_concurrent_socket = max(8, args.num_worker * 2)
        self.port = args["port"]
        self.bertdata = args["bertdata"]
        self.w2vdata = args["w2vdata"]
        self.args =args
        self.k_neighbor = args["k_neighbor"]
        # self.logger.info()

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
                sentences = [message["request"]]
                self.logger.info("sentence:%s"%(sentences[0]))
                if sentences[0] == "stop":
                    self.logger.info("stop server")
                    break
                if sentences[0] == "":
                    continue
                sentence_vecs = bc.encode(sentences)
                knn_sentences_bert = self.bertdata.knn(sentence_vecs,self.k_neighbor)
                knn_sentences_w2v = self.w2vdata.knn(str(sentences[0].encode("gb18030")),self.k_neighbor)
                # knn_sentences_w2v = knn_w2v(sentences[0])
                # backend_sock.send_json({"reply":"get message "+ str(sentence_vecs[0])})
                backend_sock.send_json({"reply":[knn_sentences_bert,knn_sentences_w2v]})
        backend_sock.close()
        self.logger.info("stop")

def startService():
    args = {"port":5678}
    query_set_file = "/search/odin/workspace/querySemantic/data/show/query_set_final"
    query_vec_file_bert = "/search/odin/workspace/querySemantic/data/show/query_vec_bert2"
    query_vec_file_w2v = "/search/odin/workspace/querySemantic/data/show/query_vec_maxpooling2"
    w2v_file = "/search/odin/workspace/querySemantic/data/wordRepresentation/click2vec/retrain_word2click_vectors_v2"
    # w2vdata = W2VData(query_set_file,query_vec_file,w2v_file)
    args["bertdata"] = BertData(query_set_file,query_vec_file_bert)
    args["w2vdata"] = W2VData(query_set_file,query_vec_file_w2v,w2v_file)
    args["k_neighbor"] = 10
    server = SenEmbServer(args)
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

