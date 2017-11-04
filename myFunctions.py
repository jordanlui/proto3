from __future__ import print_function
import decodeBytes27

packetlist = []

def printStats(mode,packets,datalist):
    data=[]
    packetsSaved = 0
    packetsDropped = 0
    if mode == 0:    
        for i in range(0,len(datalist)):
            decodeBytes27.decode(datalist[i])
            if len(datalist[i].encode('hex')) == 128:
                packetsSaved += 1
        packetsDropped = len(datalist) - packetsSaved 
    else:
        packetsDropped = packets - len(packetlist)               
    
       
    
    print("Packages Received: ", end=" ")
    print(packets)
    print("Packages Dropped: ", end=" ")
    print(packetsDropped)
    
    #print(packetlist)
 