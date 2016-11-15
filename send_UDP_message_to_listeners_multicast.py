# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:15:05 2016

@author: foresterd
"""

#Sending Messages ↓
#________________________________________________
import os
import datetime as dt
import socket
import struct
from sys import argv

# read command-line arguments
args = [x for x in argv]
command = args[1]
thearg = args[2]

MCAST_GRP = '224.1.1.1' # target system's multicast address
MCAST_PORT = 5007

dataDict = {0:'save', 1:'exit', 2:'predict'}
now = str(dt.datetime.now())[:-4]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
#MMSG = dataDict[1] +'_'+ now
#MMSG = dataDict[2] + '_' + 'outputTest.avi'
MMSG = command +'_'+ thearg

sock.sendto(MMSG.encode(), (MCAST_GRP, MCAST_PORT))
