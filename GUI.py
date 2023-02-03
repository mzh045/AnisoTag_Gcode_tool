#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter.ttk import *
import ctypes
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import re
from AnisoTag_gen import print_obj

LOG_LINE_NUM = 0

ctypes.windll.shcore.SetProcessDpiAwareness(1)
ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0)
#root.tk.call('tk', 'scaling', ScaleFactor/75)


def encode(s):
    return ' '.join([bin(ord(c)).replace('0b', '') for c in s]) # Used ' ' as the split
 
def decode(s):
    return ''.join([chr(i) for i in [int(b, 2) for b in s.split(' ')]]) # Can directly input a array equal to s.split(' '), for example, every 7 bits as one element.

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name
        self.init_window_name.call('tk', 'scaling', ScaleFactor/75)
        self.init_window_name.iconbitmap('AnisoTag_logo.ico')
        self.figure = Figure(figsize=(3.85, 2.5), dpi=100) # 3.86=2.5/55*85
        self.figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.flag = 0

    #Window
    def set_init_window(self):
        self.init_window_name.title("AnisoTag")#ID-card Gcode generator
        self.init_window_name.geometry('610x280+10+10')
        self.init_window_name.resizable(0,0)
        #Label
        self.init_data_label = Label(self.init_window_name, text="Print settings", anchor='w', width=30)
        self.init_data_label.grid(row=0, column=0, columnspan=2)
        self.log_label = Label(self.init_window_name, text="ID", anchor='w', width=21, background='cyan')
        self.log_label.grid(row=2, column=0)
        self.result_data_label = Label(self.init_window_name, text="G-code", anchor='w', width=50)
        self.result_data_label.grid(row=0, column=2)
        #Text
        self.init_data_Text = Text(self.init_window_name, width=30, height=15)  #Printing Setting
        self.init_data_Text.grid(row=1, column=0, rowspan=1, columnspan=2)
        self.init_data_Text.insert(INSERT, '[Default]\nmachine_width=230\nmachine_depth=190\nmaterial_diameter=2.85\nextruder_diameter=0.4\nline_width=0.4\n')
        self.init_data_Text.insert(INSERT, '\n')
        self.init_data_Text.insert(INSERT, 'tag_length=21.34\ntag_width=19.35\nx_offset=83.24\ny_offset=89.93\n')
        self.init_data_Text.insert(INSERT, '\n')
        self.init_data_Text.insert(INSERT, 'region_num=4\nangle_encoding_bits=4\n')
        
        self.log_data_Text = Text(self.init_window_name, width=21, height=1)  #ID
        self.log_data_Text.grid(row=3, column=0, columnspan=1)
        self.log_data_Text.insert(INSERT, '19127')
        
        self.str_trans_to_md5_button = Button(self.init_window_name, text="Gen", width=5,command=self.gcode_gen)
        self.str_trans_to_md5_button.grid(row=3, column=1)

    def gcode_gen(self):
        parameters = self.parameter_read()
        Obj = print_obj(parameters)
        Obj.line_width = parameters['line_width'] # * factor for better line illustration
        # Data sequence genration
        src = self.log_data_Text.get(1.0,END).strip()
        id = bin(int(src, 10)) # decode: str(int('0b100000000', 2))
        id = id[2:]
        print(id)
        print(len(id))
        data_size = Obj.region_num * Obj.angle_encoding_bits
        data = np.zeros(data_size, dtype = 'int')
        for i in range(len(id)):
            data[i] = int(id[i])
        points = Obj.points_gen_horizontal(data)
        Obj.gcode_write(points, 'AnisoTag.gcode')
        # Draw
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(points[:,0]+self.flag, points[:,1])
        ax.axis('equal')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master = self.init_window_name)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=2, rowspan=3, columnspan=1)
        
    def parameter_read(self):
        src = self.init_data_Text.get(1.0,END).strip()
        src = src + '\n'
        parameters = {'x_offset':83.24, 'y_offset':89.93, 'tag_width':19.35, 'tag_length':21.34, 'machine_width':230, 'machine_depth':190, 'material_diameter':2.85, \
                    'extruder_diameter':0.4, 'line_width':0.4, 'region_num':4, 'angle_encoding_bits':4}
        for key in parameters.keys():
            temp = re.search('(?<='+key+'=).*?(?=\n)',src)
            if temp is not None:
                parameters[key] = float(temp.group())
        return parameters
        
    def button_test(self):
        self.flag = self.flag + 10
        self.figure.clear()
        
        x = np.arange(0, 2*math.pi, 0.001)
        print(self.flag)
        y = np.cos(x + self.flag)
        xy = np.zeros([len(x)*2])
        xy[::2] = x
        xy[1::2] = y
        ax = self.figure.add_subplot(111)
        ax.plot(points[:,0]+self.flag, points[:,1])
        
        self.canvas = FigureCanvasTkAgg(self.figure, master = self.init_window_name)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=2, rowspan=3, columnspan=1)
    

def gui_start():
    init_window = Tk()
    ZMJ_PORTAL = MY_GUI(init_window)
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()

if __name__ == '__main__':
    gui_start()
