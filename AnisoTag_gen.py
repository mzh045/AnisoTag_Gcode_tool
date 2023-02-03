# Generating Gcode file directly from message
import math
import numpy as np
import re
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime

# Draw AnisoTag
def draw_lines(points):
    if type(points) == list:
        points = points[-1]
    plt.figure()
    plt.plot(points[:,0], points[:,1])
    plt.axis('equal')
    plt.show()

# The Gray code functions
def flip_num(my_nu):
   return '1' if(my_nu == '0') else '0'

def binary_to_gray_op(n):
   n = int(n, 2)
   n ^= (n >> 1)
   return bin(n)[2:]

def gray_to_binary(gray):
   binary_code = ""
   binary_code += gray[0]
   for i in range(1, len(gray)):
      if (gray[i] == '0'):
         binary_code += binary_code[i - 1]
      else:
         binary_code += flip_num(binary_code[i - 1])
   return binary_code

def dist(p1, p2):
    # distance of two points
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def timelabel():
    # Auto timelabel generation for filename
    T = datetime.datetime.now()
    labels = [str(T.month),str(T.day),str(T.hour),str(T.minute)]
    t = ''
    for i in range(len(labels)):
        if len(labels[i]) == 1:
            t = t + '0' + str(labels[i])
        else:
            t = t + str(labels[i])
    return t[0:-4] + '_' + t[-4:]

def rel2abs(rel):
    # Convert Gcode from relative mode to absluate mode
    rel_Mcode = re.search('M83.*\n', rel).span()
    header = rel[0:rel_Mcode[0]]
    body_footer = rel[rel_Mcode[1]:-1]
    abs_Gcode_footer = re.search('M82.*\n', body_footer).span()
    body = body_footer[0:abs_Gcode_footer[0]]
    footer = body_footer[abs_Gcode_footer[0]:-1]
    
    iter = re.finditer('(?<=\d E).*?(?= |\n|;)', body)
    prior_footer = 0
    body1 = ''
    E_abs = 0
    for E in iter:
        E_abs = E_abs + float(E.group())
        E_abs = int(E_abs*100000)/100000
        span = E.span()
        body1 = body1 + body[prior_footer:span[0]] + str(E_abs)
        prior_footer = span[1]
    body1 = body1 + body[prior_footer:-1]
    
    abs_gcode = header + body1 + footer
    return abs_gcode

class print_obj():
    def __init__(self, parameters):
        # tag setting
        self.Len = parameters['tag_length'] # mm
        self.Wid = parameters['tag_width'] #
        self.F = np.array([700, 700]) # Speed mm/min
        # 3D printer setting
        self.machine_width = parameters['machine_width'] # X
        self.machine_depth = parameters['machine_depth'] # Y
        material_diameter = parameters['material_diameter']
        extruder_diameter = parameters['extruder_diameter']
        self.line_width = parameters['line_width'] #default 0.4
        self.layer_thickness = 0.2 # default 0.2
        self.top_layer_thickness = 0.4 # The top layer with SCS microstructure should have a higher layer thickness
        self.extruding_scale_factor = 0.85 # adjust the extruding filament amount
        # For dual-extruder case
        extruder_index = 1
        extruder1_offset = 0 #-22
        # Capacity Setting
        self.region_num = int(parameters['region_num'])
        self.angle_encoding_bits = int(parameters['angle_encoding_bits'])
        # Calculated setting
        self.extrusion_speed = math.pi*(extruder_diameter/2)*(self.line_width/2)/(math.pi*(material_diameter/2)**2) * self.extruding_scale_factor
        # The position of tag region
        # In most cases, x and y_offset are calculated automatically with the goal of printing in the center of the 3D printer
        # Yet if you want to generate AnisoTag on a 3D printed object, x and y_offset, determined by the Gcode of the 3D printed object, should be used as input.
        if 'x_offset' in parameters.keys():
            self.x_offset = parameters['x_offset']
        else:
            self.x_offset = (self.machine_width-self.Wid)/2
        if 'y_offset' in parameters.keys():
            self.y_offset = parameters['y_offset']
        else:
            self.y_offset = (self.machine_depth-self.Len)/2
        # switch the extruder
        if extruder_index == 1:
            self.x_offset = self.x_offset + extruder1_offset
        # mapping function
        # Note that an effective nonlinear mapping should be calculated from the actual parameters of the detection prototype.
        temp = sio.loadmat('map_inverse.mat')
        self.mapping = temp['Mi']/math.pi
        # Gcode file header and footer
        # To meet the needs of different 3D printers for headers and footers of Gcode file
        if 'printer_type' in parameters.keys():
            with open('header_creality.txt', 'r') as f:
                self.g_header = f.read()
            with open('footer_creality.txt', 'r') as f:
                self.g_footer = f.read()
        else:
            if extruder_index == 0:
                with open('header.txt', 'r') as f:
                    self.g_header = f.read()
            else:
                with open('header2.txt', 'r') as f:
                    self.g_header = f.read()
            with open('footer.txt', 'r') as f:
                self.g_footer = f.read()
        # skirt + Boundary line
        self.adhesion = self.adhesion_gen(self.x_offset, self.y_offset, self.Len, self.Wid, 3, 3) + self.adhesion_gen(self.x_offset, self.y_offset, self.Len, self.Wid, 0, 3)
        
    def adhesion_gen(self, x_off, y_off, L, W, d, n):
        # Adhesion generation, rectangular infilling lines as skirt
        # d: The distance between skirt and AnisoTag
        # n: Number of skirt lines
        # default start = 1
        skirt = ''
        for i in range(n-1,-1,-1):
            skirt = skirt + 'G0' + ' F' + str(self.F[0]) + ' X' + str(round(x_off-d-self.line_width*i,2)) + ' Y' + str(round(y_off-d-self.line_width*i,2)) + '\n'
            skirt = skirt + 'G1' + ' F' + str(self.F[1]) + ' X' + str(round(x_off+W+d+self.line_width*i,2)) + ' Y' + str(round(y_off-d-self.line_width*i,2)) + ' E' + str(round((W+2*d)*self.extrusion_speed ,5)) + '\n'
            skirt = skirt + 'G1' + ' F' + str(self.F[1]) + ' X' + str(round(x_off+W+d+self.line_width*i,2)) + ' Y' + str(round(y_off+L+d+self.line_width*i,2)) + ' E' + str(round((L+2*d)*self.extrusion_speed ,5)) + '\n'
            skirt = skirt + 'G1' + ' F' + str(self.F[1]) + ' X' + str(round(x_off-d-self.line_width*i,2)) + ' Y' + str(round(y_off+L+d+self.line_width*i,2)) + ' E' + str(round((W+2*d)*self.extrusion_speed ,5)) + '\n'
            skirt = skirt + 'G1' + ' F' + str(self.F[1]) + ' X' + str(round(x_off-d-self.line_width*i,2)) + ' Y' + str(round(y_off-d-self.line_width*i,2)) + ' E' + str(round((L+2*d)*self.extrusion_speed ,5)) + '\n'
        return skirt
    
    def block_lines(self, L, W, angle):
        # Generate infilling line (points array) in one 2D block
        if angle > 90:
            angle = 180 - angle
            ang_flag = 1
        else:
            ang_flag = 0
        max_num = (W*W + L*L)/(math.cos(angle/180*math.pi)*W + math.sin(angle/180*math.pi)*L)/self.line_width
        max_num = int(math.ceil(max_num)) # max lines number
        points = np.empty((max_num*2 ,2), dtype='float') # (x, y) ~ (W, L), One line needs two points
        dy = self.line_width / math.cos(angle/180*math.pi)
        dx = self.line_width / math.sin(angle/180*math.pi)
        Dy = dy / 2
        Dx = dx / 2
        zigzag_flag = 1
        for i in range(max_num):
            if Dy < L:
                p1 = [0, Dy]
            else:
                W1 = (Dy-L)/math.tan(angle/180*math.pi)
                if W1 <= W:
                    p1 = [W1 ,L]
                else:
                    break
            Dy = Dy + dy
            
            if Dx < W:
                p2 = [Dx, 0]
            else:
                L2 = (Dx-W)*math.tan(angle/180*math.pi)
                if L2 <= L:
                    p2 = [W, L2]
                else:
                    break
            Dx = Dx + dx
            
            if zigzag_flag == 1:
                points[i*2,:] = p1
                points[i*2+1,:] = p2
            else:
                points[i*2,:] = p2
                points[i*2+1,:] = p1
            zigzag_flag = 1 - zigzag_flag
        points_c = points[0:i*2,:]
        if ang_flag == 1:
            points_c[:,0] = W - points_c[:,0]
        return points_c
    
    def gcode_convert(self, x_off, y_off, points):
        # Convert a points array to Gcode command with offset
        g_temp = ''
        G_flag = 0
        for i in range(points.shape[0]):
            if G_flag == 0:
                if dist(points[i,:], points[i-1,:]) > self.line_width*1.5:
                    g_temp = g_temp + 'G1 F1500 E-2\n'
                    retraction_flag = 1
                else:
                    retraction_flag = 0
                g_temp = g_temp + 'G' + str(G_flag) + ' F' + str(self.F[G_flag]) + ' X' + str(round(x_off+points[i,0],2)) + ' Y' + str(round(y_off+points[i,1],2)) + '\n'
            else:
                if retraction_flag == 1:
                    g_temp = g_temp + 'G1 F1500 E2\n'
                p0 = points[i,:]
                p1 = points[i-1,:] # 1 -> 0
                q = 0.1 # The edge has less extruding amount, q is the edge perscent
                eq = 1 # The extruding percent, 0.5 default 
                p00 = p0 + (p1-p0)*q
                p11 = p1 + (p0-p1)*q
                g_temp = g_temp + 'G' + str(G_flag) + ' F' + str(self.F[G_flag]) + ' X' + str(round(x_off+p11[0],2)) + ' Y' + str(round(y_off+p11[1],2))
                g_temp = g_temp + ' E' + str(round(dist(p11,p1)*self.extrusion_speed*eq,5)) + '\n'
                g_temp = g_temp + 'G' + str(G_flag) + ' F' + str(self.F[G_flag]) + ' X' + str(round(x_off+p00[0],2)) + ' Y' + str(round(y_off+p00[1],2))
                g_temp = g_temp + ' E' + str(round(dist(p00,p11)*self.extrusion_speed,5)) + '\n'
                g_temp = g_temp + 'G' + str(G_flag) + ' F' + str(self.F[G_flag]) + ' X' + str(round(x_off+p0[0],2)) + ' Y' + str(round(y_off+p0[1],2))
                g_temp = g_temp + ' E' + str(round(dist(p0,p00)*self.extrusion_speed*eq,5)) + '\n'
            G_flag = 1 - G_flag
        return g_temp
    
    def points_gen_vertical(self, data):
        # Divide the tag region vertically to generate encoding regions
        # The infilling angel of each encoding region is determined by data
        block_wid = self.Wid/self.region_num
        delta_wid = 0.0 # Set 0.1 to avoid material accumulation between regions
        for i in range(self.region_num):
            data_temp = data[i*self.angle_encoding_bits:(i+1)*self.angle_encoding_bits]
            strdata = ''
            # Gray code deconding
            for p in range(len(data_temp)):
                strdata = strdata + str(data_temp[p])
            data_bin = gray_to_binary(strdata)
            for p in range(len(data_temp)):
                data_temp[p] = int(data_bin[p])
            angle_step = 180/(2**self.angle_encoding_bits)
            bit_sum = 0
            for p in range(self.angle_encoding_bits):
                bit_sum = bit_sum + (2**p)*data_temp[self.angle_encoding_bits-1 - p]
            
            angle = np.argmin(abs(bit_sum/(2**self.angle_encoding_bits)-self.mapping)) + angle_step/1000
            #print(angle)
            if i == 0 or i == self.region_num:
                temp = self.block_lines(self.Len, block_wid - delta_wid, angle)
            else:
                temp = self.block_lines(self.Len, block_wid - delta_wid*2, angle)
            if angle > 90:# because angle > 90, change to 4->3, left to right
                temp = temp[::-1,:]
            if i == 0:
                points = temp
            else:
                temp[:,0] = temp[:,0] + i*block_wid + delta_wid
                points = np.vstack((points,temp))
        return points
    
    def points_gen_horizontal(self, data):
        # Horizontal
        block_len = self.Len/self.region_num
        delta_len = 0.0  
        for i in range(self.region_num):
            data_temp = data[i*self.angle_encoding_bits:(i+1)*self.angle_encoding_bits]
            strdata = ''
            for p in range(len(data_temp)):
                strdata = strdata + str(data_temp[p])
            data_bin = gray_to_binary(strdata)
            for p in range(len(data_temp)):
                data_temp[p] = int(data_bin[p])
            angle_step = 180/(2**self.angle_encoding_bits)
            bit_sum = 0
            for p in range(self.angle_encoding_bits):
                bit_sum = bit_sum + (2**p)*data_temp[self.angle_encoding_bits-1 - p]
            angle = np.argmin(abs(bit_sum/(2**self.angle_encoding_bits)-self.mapping)) + angle_step/1000
            if i == 0 or i == self.region_num:
                temp = self.block_lines(block_len - delta_len, self.Wid, angle)
            else:
                temp = self.block_lines(block_len - delta_len*2, self.Wid, angle)
            # starting at 1 or 2, need not reverse like points_gen_vertical
            if i == 0:
                points = temp
            else:
                temp[:,1] = temp[:,1] + i*block_len + delta_len
                points = np.vstack((points,temp))
        return points
    
    def gcode_write(self, points, save_name):
        # Generate the Gcode file from a set of points
        if type(points) == list:
            lines = ''
            layer_height = 0
            for i in range(len(points)):
                # if the input points is a list, representing that there are multiple layers. So add Z-axis from 0.2
                if i == (len(points) - 1):
                    layer_height = layer_height + self.top_layer_thickness
                else:
                    layer_height = layer_height + self.layer_thickness
                lines = lines + 'G1 F600 Z' + str(layer_height) + '\n' + self.gcode_convert(self.x_offset, self.y_offset, points[i]) + '\n'
        else:
            lines = self.gcode_convert(self.x_offset, self.y_offset, points)
        g = rel2abs(self.g_header + self.adhesion + '\n' + lines + self.g_footer)
        with open(save_name, 'w') as f:
            f.write(g)

    def gcode_intermedia(self, points, save_name):
        # Generate Gcode of AnisoTag on relative mode
        if type(points) == list:
            lines = ''
            layer_height = 0
            for i in range(len(points)):
                if i == (len(points) - 1):
                    layer_height = layer_height + self.top_layer_thickness
                else:
                    layer_height = layer_height + self.layer_thickness
                lines = lines + 'G1 F600 Z' + str(layer_height) + '\n' + self.gcode_convert(self.x_offset, self.y_offset, points[i]) + '\n'
        else:
            lines = self.gcode_convert(self.x_offset, self.y_offset, points)
        with open(save_name, 'w') as f:
            f.write(lines)

if __name__ == '__main__':
    plt.close('all')
    #parameters = {'tag_length':54, 'tag_width':86, 'machine_width':230, 'machine_depth':190, 'material_diameter':2.85, \
    #              'extruder_diameter':0.4, 'line_width':0.4, 'region_num':17, 'angle_encoding_bits':4}
    
    #parameters = {'tag_length':54, 'tag_width':86, 'machine_width':220, 'machine_depth':220, 'material_diameter':1.75, \
    #              'extruder_diameter':0.4, 'line_width':0.4, 'region_num':17, 'angle_encoding_bits':4, 'printer_type': 1}
    
    parameters = {'x_offset':83.24, 'y_offset':89.93, 'tag_width':19.35, 'tag_length':21.34, 'machine_width':230, 'machine_depth':190, 'material_diameter':2.85, \
                  'extruder_diameter':0.4, 'line_width':0.4, 'region_num':4, 'angle_encoding_bits':4}

    C = print_obj(parameters)
    print('top_layer_thickness: %f'% C.top_layer_thickness) # should be slightly high than C.layer_thickness

    # data generation
    data_size = C.region_num * C.angle_encoding_bits
    np.random.seed(0)
    data = np.random.randint(0,2,(1,data_size))[0]
    '''
    data = []
    words = 'CHI2023'
    for i in range(len(words)):
        word = words[i]
        temp = bin(ord(word))[2::]
        temp = '0'*(7-len(temp)) + temp
        for j in range(len(temp)):
            if temp[j] == '1':
                data.append(1)
            else:
                data.append(0)
    data = [0]*(data_size-len(data)) + data
    data = np.array(data)
    '''
    
    points = C.points_gen_horizontal(data)
    
    # test multilayer
    '''
    parameters1 = parameters
    parameters1['region_num'] = 1
    C1 = print_obj(parameters1)
    points_bottom_layers = []
    for i in range(3): # layer_count
        if i%2 == 0:
            data = np.array([0,1,0,0]) # differnt angle for adjacent layers
        else:
            data = np.array([0,0,1,1])
        temp = C1.points_gen_horizontal(data)
        points_bottom_layers.append(temp)
    points_bottom_layers.append(points)
    points = points_bottom_layers
    '''
    # test Grid for bottom print

    # Lines show
    draw_lines(points)

    C.gcode_write(points, 'test'+timelabel()+'.gcode')
    #C.gcode_intermedia(points, 'Aniso'+timelabel()+'.gcode')