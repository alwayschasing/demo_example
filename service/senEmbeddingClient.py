#!/usr/bin/env python
# -*- coding: utf-8 -*-

import zmq
import zmq.decorators as zmqd


context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5678")
print "connect"

socket.send_json({"request":"故宫"})
print "send"
response = socket.recv_json()
print "recv"
[bert_querys,w2v_querys]= response["reply"]
for q in bert_querys:
    print q
print "---------------"
for q in w2v_querys:
    print q
# print "response:"+str(response["reply"])
