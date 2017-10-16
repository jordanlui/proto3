from __future__ import print_function
import struct
#import time
import myFunctions

numBytes = 33 * 4 + 8

def decode(data):
    #print(len(data))
    
    index = 0
    #each index of data is one string so a byte has to be two indices to get the hex value
    startBytes = data[index:index+4]
    if startBytes != b'3C3C':
        #only error checking implemented so far, first line is usually garbage
        return 0
    if len(data) != 180:
#        print(data)
#        print(len(data))
#        time.sleep(2)
        return 0

    
    index += 4

    packetHex = str(data[index:index+4])
    packet = int(packetHex,16)
    index += 4
    runtimeHex = str(data[index:index+8])
    runtime = long(runtimeHex,16)
    index += 8
    DL1Hex = str(data[index:index+8])
    DL1 = struct.unpack('<f', DL1Hex.decode('hex'))[0]
    index += 8
    DS1Hex = str(data[index:index+8])
    DS1 = struct.unpack('<f', DS1Hex.decode('hex'))[0]
    index += 8
    DL2Hex = str(data[index:index+8])
    DL2 = struct.unpack('<f', DL2Hex.decode('hex'))[0]
    index += 8
    DS2Hex = str(data[index:index+8])
    DS2 = struct.unpack('<f', DS2Hex.decode('hex'))[0]
    index += 8
    AcXHex = str(data[index:index+8])
    AcX = struct.unpack('<f', AcXHex.decode('hex'))[0]
    index += 8
    AcYHex = str(data[index:index+8])
    AcY = struct.unpack('<f', AcYHex.decode('hex'))[0]
    index += 8
    AcZHex = str(data[index:index+8])
    AcZ = struct.unpack('<f', AcZHex.decode('hex'))[0]
    index += 8
    O8to1_1Hex = str(data[index:index+4])
    O8to1_1 = int(O8to1_1Hex,16)
    index += 4
    O8to1_2Hex = str(data[index:index+4])
    O8to1_2 = int(O8to1_2Hex,16)
    index += 4
    O8to1_3Hex = str(data[index:index+4])
    O8to1_3 = int(O8to1_3Hex,16)
    index += 4
    O8to1_4Hex = str(data[index:index+4])
    O8to1_4 = int(O8to1_4Hex,16)
    index += 4
    O8to1_5Hex = str(data[index:index+4])
    O8to1_5 = int(O8to1_5Hex,16)
    index += 4
    O8to1_6Hex = str(data[index:index+4])
    O8to1_6 = int(O8to1_6Hex,16)
    index += 4
    O8to1_7Hex = str(data[index:index+4])
    O8to1_7 = int(O8to1_7Hex,16)
    index += 4
    O8to1_8Hex = str(data[index:index+4])
    O8to1_8 = int(O8to1_8Hex,16)
    index += 4
    O16to1_1Hex = str(data[index:index+4])
    O16to1_1 = int(O16to1_1Hex,16)
    index += 4
    O16to1_2Hex = str(data[index:index+4])
    O16to1_2 = int(O16to1_2Hex,16)
    index += 4
    O16to1_3Hex = str(data[index:index+4])
    O16to1_3 = int(O16to1_3Hex,16)
    index += 4
    O16to1_4Hex = str(data[index:index+4])
    O16to1_4 = int(O16to1_4Hex,16)
    index += 4
    O16to1_5Hex = str(data[index:index+4])
    O16to1_5 = int(O16to1_5Hex,16)
    index += 4
    O16to1_6Hex = str(data[index:index+4])
    O16to1_6 = int(O16to1_6Hex,16)
    index += 4
    O16to1_7Hex = str(data[index:index+4])
    O16to1_7 = int(O16to1_7Hex,16)
    index += 4
    O16to1_8Hex = str(data[index:index+4])
    O16to1_8 = int(O16to1_8Hex,16)
    index += 4
    O16to1_9Hex = str(data[index:index+4])
    O16to1_9 = int(O16to1_9Hex,16)
    index += 4
    O16to1_10Hex = str(data[index:index+4])
    O16to1_10 = int(O16to1_10Hex,16)
    index += 4
    O16to1_11Hex = str(data[index:index+4])
    O16to1_11 = int(O16to1_11Hex,16)
    index += 4
    O16to1_12Hex = str(data[index:index+4])
    O16to1_12 = int(O16to1_12Hex,16)
    index += 4
    O16to1_13Hex = str(data[index:index+4])
    O16to1_13 = int(O16to1_13Hex,16)
    index += 4
    O16to1_14Hex = str(data[index:index+4])
    O16to1_14 = int(O16to1_14Hex,16)
    index += 4
    O16to1_15Hex = str(data[index:index+4])
    O16to1_15 = int(O16to1_15Hex,16)
    index += 4
    O16to1_16Hex = str(data[index:index+4])
    O16to1_16 = int(O16to1_16Hex,16)

    print(packet, end="   ")
    print(runtime, end="   ")
    print(DL1, end="   ")
    print(DS1, end="   ")
    print(DL2, end=" ")
    print(DS2, end=" ")
    print(AcX, end=" ")
    print(AcY, end=" ")
    print(AcZ, end=" ")
    print(O8to1_1, end=" ")
    print(O8to1_2, end=" ")
    print(O8to1_3, end=" ")
    print(O8to1_4, end=" ")
    print(O8to1_5, end=" ")
    print(O8to1_6, end=" ")
    print(O8to1_7, end=" ")
    print(O8to1_8, end=" ")
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







