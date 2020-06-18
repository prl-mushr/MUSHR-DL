import time
import win32api as wapi
import win32con
import numpy as np

KeyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    KeyList.append(char)
    
def key_press():
    keys = []
    for key in KeyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys 

def reset_mouse_pos():
    screen_width = wapi.GetSystemMetrics(0)
    screen_height = wapi.GetSystemMetrics(1)
    wapi.SetCursorPos((screen_width//2,screen_height//2))

def get_mouse_pos():
    screen_width = wapi.GetSystemMetrics(0)
    screen_height = wapi.GetSystemMetrics(1)
    x,y = wapi.GetCursorPos()
    x -= screen_width//2
    x /= (screen_width)*0.25
    y -= screen_height//2
    y /= -(screen_height)*0.25
    return x,y

def key_check():
    keys = np.array([0,0,0,0])
    if wapi.GetAsyncKeyState(win32con.VK_UP):
        keys = np.sum([keys,[0,1,0,0]],axis = 0)
    if wapi.GetAsyncKeyState(win32con.VK_LEFT):
        keys = np.sum([keys,[1,0,0,0]],axis = 0)
    if wapi.GetAsyncKeyState(win32con.VK_RIGHT):
        keys = np.sum([keys,[0,0,1,0]],axis = 0)
    if wapi.GetAsyncKeyState(win32con.VK_DOWN):
        keys = np.sum([keys,[0,0,0,1]],axis = 0) 
    return keys 

def key_check_alert():
    keys = np.array([0,0,0])
    if wapi.GetAsyncKeyState(win32con.VK_UP):
        keys = np.sum([keys,[0,1,0]],axis = 0)
    if wapi.GetAsyncKeyState(win32con.VK_LEFT):
        keys = np.sum([keys,[1,0,0]],axis = 0)
    if wapi.GetAsyncKeyState(win32con.VK_RIGHT):
        keys = np.sum([keys,[0,0,1]],axis = 0)
    return keys 
