import serial
import time
import traceback
import numpy as np
from key_check import *


def check_range(x,lim):
    if(x>lim):
        return lim
    elif(x<-lim):
        return -lim
    return x

# this is if you are using the hacky arduino+TxRx based controller (Script included)
class remote_control():
    def __init__(self):
        self.st = 0.0
        self.th = 0.0
        self.autonomous = False
        self.ser =  serial.Serial('COM4', 250000, timeout=0) # put in your controller's com port number here. 
        if self.ser.isOpen():
            self.ser.close()
        self.ser.open()

    def read_loop(self):
        while 1:
            try:
                while(self.ser.inWaiting()<10):
                    time.sleep(0.01)

                buf = np.frombuffer(self.ser.read(10),dtype='int16')

                if len(buf)==5:
                    steering = buf[0]#ord(self.ser.read())<<8|ord(self.ser.read())
                    throttle = buf[1]#ord(self.ser.read())<<8|ord(self.ser.read())
                    right_stick = buf[2]#ord(self.ser.read())<<8|ord(self.ser.read())
                    A = buf[3]#ord(self.ser.read())<<8|ord(self.ser.read())
                    B = buf[4]#ord(self.ser.read())<<8|ord(self.ser.read())
                    while self.ser.inWaiting():
                        self.ser.read()
                    self.st = (steering - 1500)*0.002
                    self.th = (throttle - 1500)*0.0018
                    AxisRx = (right_stick-1500)*0.002 
                    #prevent steering and throttle from being outside the usable range.
                    if(B>1500):
                        self.autonomous = True
                    else:
                        self.autonomous = False 
                    self.st = check_range(self.st,1.0)

                    self.th = check_range(self.th,1.0)
                    # print(steering,throttle)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                # pass
                print(traceback.format_exc())
            except:
                pass

# This one is used by default
class keyboard_control():
    def __init__(self):
        self.st = 0.0
        self.th = 0.0
        self.autonomous = False
        reset_mouse_pos()

    def read_loop(self):
        while 1:
            try:
                time.sleep(0.01)
                self.st,self.th = get_mouse_pos()
                print(key_press())
                if(key_press() == ['A']):
                    self.autonomous = True
                if(key_press() == ['M']):
                    self.autonomous = False
                self.st = check_range(self.st,1.0)
                self.th = check_range(self.th,1.0)
                # print(steering,throttle)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                # pass
                print(traceback.format_exc())
            except:
                pass

if __name__ == "__main__": 
    obj = keyboard_control()
    obj.read_loop()