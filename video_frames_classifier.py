# -*- coding: utf-8 -*-
"""
Modified on Monday November 14 09:45:13 2016

VFC is an example of using a pretrained neural network to check frames from video for presence of specific classes of objects. It was designed to be used on multiple camera-enabled systems operating on a common network. A neural network is loaded into memory (in this case the resnet_50_weights for theano as in this Keras example: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py), then threads are spawned to listen for UDP multicast messages on one or more ports as defined in "hostnames_ports_listen.csv" (this system's hostname must preceed one or more valid port numbers). Valid messages may contain two fields; the first is the command. If the command is "predict" then the second field is taken to be the file path of the video to analyze. The video frames are then input sequentially to the neural network which outputs probabilities for each of the 1000 classes in ImageNet. These class probabilities are used determine the likelihood that a vehicle (as defined in "vehicle_words.csv") is present in each image. A vehicle (car, truck, motorcycle, etc.) is considered present if a class in "vehicle_words.csv" is present in the top 20 predicted classes and has a probability > 0.09.

The code is run from the command-line like:
  $ python video_frames_classifier.py '224.1.1.1'

The code to send the UDP messages is "send_UDP_messages_to_listeners_multicast.py". This should be run from a system on the same network (or even another terminal on the same system) as follows:
  $ vpy34 send_UDP_message_to_listeners_multicast.py predict outputTest.avi
 where "predict" is the commnd and "outputTest.avi" is the videofile to be analyzed.

Requirements:
  python 3.x
  theano
  keras
  skimage
  imageio

also requires that the model weights "resnet50_weights_th_dim_ordering_th_kernels.h5" be in the Keras directory ~/.keras/models AND that the two CSV files "vehicle_words.csv" and "hostnames_ports_listen.csv" be in the VFC directory.

@author: foresterd
"""

import os
import numpy as np
import imageio
import resnet50
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input, load_vehicle_words
from skimage.transform import resize
import threading
import struct


# INITIALIZATIONS and SETTINGS
model = resnet50.ResNet50(include_top=True, weights='imagenet')
vWordList = load_vehicle_words()
doexit = False
global p_num, m_grp

# make an initial prediction to load model to memory
xf = np.zeros((1, 3, 224, 224),dtype=np.uint8)
print('loading neural network...')
model.predict(xf)
print('network model loaded.')

# read command-line arguments
from sys import argv
args = [x for x in argv]
m_grp = args[1] # first arguemnt is multicast group

# get this system's hostname
import socket
hostname = socket.gethostname()

# read csv file to learn which on ports this system should listen
import csv
portslist = []
with open('hostnames_ports_listen.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] == hostname:
            portslist = row[1:]
if (len(portslist)) > 0:
    portslist = [int(x) for x in portslist]
else:
    print('WARNING: THIS HOSTNAME NOT FOUND IN HOST_PORTS FILEn/')

#-----------------------------------------------------------------------------

def UDP_multicast_listener():
    global predict_now, filename, doexit
    MCAST_GRP = m_grp
    MCAST_PORT = p_num
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))  # use MCAST_GRP instead of '' to listen only
                                 # to MCAST_GRP, not all groups on MCAST_PORT
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)

    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        msg = sock.recv(10240).decode()
        print("Received message: ", msg)
        cmd, fn = msg.split('_')
        if cmd == "predict":
            predict_now = True
            filename = fn
        elif cmd == 'exit':
            doexit = True
        else:
            predict_now = False
            return(predict_now)

#-----------------------------------------------------------------------------

for ip in range(len(portslist)):
    p_num = portslist[ip]
    predict_now = threading.Thread(target=UDP_multicast_listener).start() # listen on ports

while True:
    if predict_now:
        vid = imageio.get_reader(filename,  'ffmpeg')
        for i, im in enumerate(vid):
            print('Frame #:', i)
            rows,cols,channels = im.shape
            im = im[:,cols//2-rows//2:cols//2+rows//2,:] # crop image so that cols = rows)
            im = resize(im, (224,224)) # resize to 224 x 224, the network input size
            x = image.img_to_array(im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            #print('Input image shape:', x.shape)
            preds = model.predict(x)
            decoded = decode_predictions(preds)[0]
            predList = [x[0] for x in decoded]
            nameList = [x[1] for x in decoded]
            probList = [x[2] for x in decoded]
            detectedList = [[nameList[jj], probList[jj]] for jj in range(len(predList))
                             if predList[jj] in vWordList and probList[jj] > 0.09]
            #print(nameList)
            if len(detectedList) > 0:
                print('VEHICLE PRESENT')
                #print(detectedList)
        predict_now = False

    if doexit:
        print('Exiting...')
        os._exit(0)
