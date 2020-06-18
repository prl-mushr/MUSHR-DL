import time
import win32api as wapi
import win32con
import numpy as np
from pynput import keyboard, mouse
from pynput.mouse import Controller
from screeninfo import get_monitors

comp_mouse = Controller()

m = get_monitors()
m = m[0]

screen_width = m.width
screen_height = m.height

mouse_x = screen_width//2
mouse_y = screen_height//2

pressed_key=''

def on_press(key):
    global pressed_key
    try:
        pressed_key = key.char
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    global pressed_key
    pressed_key = ''
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
listener = keyboard.Listener(on_press=on_press,on_release=on_release)
listener.start()


def on_move(x, y):
    global mouse_x
    global mouse_y
    mouse_x = x
    mouse_y = y

listener_mouse = mouse.Listener(on_move=on_move)
listener_mouse.start()

KeyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    KeyList.append(char)
    
def key_press():
    # keys = []
    # for key in KeyList:
    #     if wapi.GetAsyncKeyState(ord(key)):
    #         keys.append(key)
    # return keys 
    global pressed_key
    return [str(pressed_key).upper()]

def reset_mouse_pos():
    global screen_width
    global screen_height
    # screen_width = wapi.GetSystemMetrics(0)
    # screen_height = wapi.GetSystemMetrics(1)
    # wapi.SetCursorPos((screen_width//2,screen_height//2))
    comp_mouse.position = (screen_width//2,screen_height//2)

def get_mouse_pos():
    global screen_width
    global screen_height
    # screen_width = wapi.GetSystemMetrics(0)
    # screen_height = wapi.GetSystemMetrics(1)
    global mouse_x
    global mouse_y
    # x,y = wapi.GetCursorPos()
    x,y = mouse_x, mouse_y
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
