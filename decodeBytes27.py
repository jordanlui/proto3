from __future__ import print_function
import struct
#import time
import myFunctions


numBytes = 33 * 4 + 8


def decode(dataIn, printOut = 0):
    data = dataIn.encode('hex')

    index = 2
    #each index of data is one string so a byte has to be two indices to get the hex value
    startBytes = data[index:index+4]
    if startBytes != '3c3c':
        #only error checking implemented so far, first line is usually garbage
        print("bad start")
        print(startBytes)
        return 0
    if len(data) != 128:
        print("bad length")
        return 0

    
    index += 4

    packetHex = str(data[index:index+4])
    packet = int(packetHex,16)
    index += 4
    myFunctions.packetlist.append(packet)
    runtimeHex = str(data[index:index+8])
    runtime = long(runtimeHex,16)
    index += 8
    DL1Hex = str(data[index:index+4])
    DL1 = int(DL1Hex,16)
    index += 4
    DS1Hex = str(data[index:index+4])
    DS1 = int(DS1Hex,16)
    index += 4
    DL2Hex = str(data[index:index+4])
    DL2 = int(DL2Hex,16)
    index += 4
    DS2Hex = str(data[index:index+4])
    DS2 = int(DS2Hex,16)
    index += 4
    AcXHex = str(data[index:index+8])
    AcX = struct.unpack('<f', AcXHex.decode('hex'))[0]
    index += 8
    AcYHex = str(data[index:index+8])
    AcY = struct.unpack('<f', AcYHex.decode('hex'))[0]
    index += 8
    AcZHex = str(data[index:index+8])
    AcZ = struct.unpack('<f', AcZHex.decode('hex'))[0]
    index += 8
    GyXHex = str(data[index:index+8])
    GyX = struct.unpack('<f', GyXHex.decode('hex'))[0]
    index += 8
    GyYHex = str(data[index:index+8])
    GyY = struct.unpack('<f', GyYHex.decode('hex'))[0]
    index += 8
    GyZHex = str(data[index:index+8])
    GyZ = struct.unpack('<f', GyZHex.decode('hex'))[0]
    index += 8
#    O8to1_1Hex = str(data[index:index+4])
#    O8to1_1 = int(O8to1_1Hex,16)
#    index += 4
#    O8to1_2Hex = str(data[index:index+4])
#    O8to1_2 = int(O8to1_2Hex,16)
#    index += 4
#    O8to1_3Hex = str(data[index:index+4])
#    O8to1_3 = int(O8to1_3Hex,16)
#    index += 4
#    O8to1_4Hex = str(data[index:index+4])
#    O8to1_4 = int(O8to1_4Hex,16)
#    index += 4
#    O8to1_5Hex = str(data[index:index+4])
#    O8to1_5 = int(O8to1_5Hex,16)
#    index += 4
#    O8to1_6Hex = str(data[index:index+4])
#    O8to1_6 = int(O8to1_6Hex,16)
#    index += 4
#    O8to1_7Hex = str(data[index:index+4])
#    O8to1_7 = int(O8to1_7Hex,16)
#    index += 4
#    O8to1_8Hex = str(data[index:index+4])
#    O8to1_8 = int(O8to1_8Hex,16)
#    index += 4
    O16to1_1Hex = str(data[index:index+2])
    O16to1_1 = int(O16to1_1Hex,16)
    index += 2
    O16to1_2Hex = str(data[index:index+2])
    O16to1_2 = int(O16to1_2Hex,16)
    index += 2
    O16to1_3Hex = str(data[index:index+2])
    O16to1_3 = int(O16to1_3Hex,16)
    index += 2
    O16to1_4Hex = str(data[index:index+2])
    O16to1_4 = int(O16to1_4Hex,16)
    index += 2
    O16to1_5Hex = str(data[index:index+2])
    O16to1_5 = int(O16to1_5Hex,16)
    index += 2
    O16to1_6Hex = str(data[index:index+2])
    O16to1_6 = int(O16to1_6Hex,16)
    index += 2
    O16to1_7Hex = str(data[index:index+2])
    O16to1_7 = int(O16to1_7Hex,16)
    index += 2
    O16to1_8Hex = str(data[index:index+2])
    O16to1_8 = int(O16to1_8Hex,16)
    index += 2
    O16to1_9Hex = str(data[index:index+2])
    O16to1_9 = int(O16to1_9Hex,16)
    index += 2
    O16to1_10Hex = str(data[index:index+2])
    O16to1_10 = int(O16to1_10Hex,16)
    index += 2
    O16to1_11Hex = str(data[index:index+2])
    O16to1_11 = int(O16to1_11Hex,16)
    index += 2
    O16to1_12Hex = str(data[index:index+2])
    O16to1_12 = int(O16to1_12Hex,16)
    index += 2
    O16to1_13Hex = str(data[index:index+2])
    O16to1_13 = int(O16to1_13Hex,16)
    index += 2
    O16to1_14Hex = str(data[index:index+2])
    O16to1_14 = int(O16to1_14Hex,16)
    index += 2
    O16to1_15Hex = str(data[index:index+2])
    O16to1_15 = int(O16to1_15Hex,16)
    index += 2
    O16to1_16Hex = str(data[index:index+2])
    O16to1_16 = int(O16to1_16Hex,16)
    
    O16 = [O16to1_1,O16to1_2,O16to1_3,O16to1_4,O16to1_5,O16to1_6,O16to1_7,O16to1_8,O16to1_9,O16to1_10,O16to1_11,O16to1_12,O16to1_13,O16to1_14,O16to1_15,O16to1_16]

    if printOut == 1:
        print(packet, end="   ")
        print(runtime, end="   ")
        print(DL1, end="   ")
        print(DS1, end="   ")
        print(DL2, end=" ")
        print(DS2, end=" ")
        print(AcX, end=" ")
        print(AcY, end=" ")
        print(AcZ, end=" ")
        print(GyX, end=" ")
        print(GyY, end=" ")
        print(GyZ, end=" ")
    #    print(O8to1_1, end=" ")
    #    print(O8to1_2, end=" ")
    #    print(O8to1_3, end=" ")
    #    print(O8to1_4, end=" ")
    #    print(O8to1_5, end=" ")
    #    print(O8to1_6, end=" ")
    #    print(O8to1_7, end=" ")
    #    print(O8to1_8, end=" ")
        print(O16to1_1, end=" ")
        print(O16to1_2, end=" ")
        print(O16to1_3, end=" ")
        print(O16to1_4, end=" ")
        print(O16to1_5, end=" ")
        print(O16to1_6, end=" ")
        print(O16to1_7, end=" ")
        print(O16to1_8, end=" ")
        print(O16to1_9, end=" ")
        print(O16to1_10, end=" ")
        print(O16to1_11, end=" ")
        print(O16to1_12, end=" ")
        print(O16to1_13, end=" ")
        print(O16to1_14, end=" ")
        print(O16to1_15, end=" ")
        print(O16to1_16, end="\n")
    
    dataOut = [packet, runtime, DL1, DS1, DL2, DS2, AcX, AcY, AcZ, GyX, GyY, GyZ, O16]
    return dataOut






