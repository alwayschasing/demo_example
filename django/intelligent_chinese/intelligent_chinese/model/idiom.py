#!/usr/bin/env python
# -*- coding: utf-8 -*-

from django.shortcuts import render
import zmq
# from zmq.utils import jsonapi
import logging

logger = logging.getLogger('idiom')

def mainhtml(request):
    return render(request,"idiom.html")

def search_by_describe(request):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:1991") 
    request.encoding = 'utf-8' 
    if "describe" in request.GET:
        desc = request.GET['describe'].encode('utf-8')   
        logger.info("query:%s"%(desc))
        try:
            socket.send_json({"request":desc})
        except Exception:
            return render(request, "idiom.html", {"error":"send error"})
        response = socket.recv_json()
        idioms = response["reply"]
        return render(request, "idiom.html",{"idioms":idioms})
    else:
        logger.info("error:" + desc)
        return render(request, "idiom.html", {"error":"error"})
    socket.close()


