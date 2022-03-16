# MIT License
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#basic modules
import time
import sys
import os
import numpy as np

#matplotlib 
import matplotlib as mpl
mpl.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#basemap
from mpl_toolkits.basemap import Basemap


class CamVisualizer(object):
    
    def __init__(self):
        
        # Create figre
        lats = np.linspace(-90,90,768)
        longs = np.linspace(-180,180,1152)
        self.my_map = Basemap(projection='gall', llcrnrlat=min(lats), 
                             llcrnrlon=min(longs), urcrnrlat=max(lats), 
                             urcrnrlon=max(longs), resolution = 'i')
        
        xx, yy = np.meshgrid(longs, lats)
        self.x_map, self.y_map = self.my_map(xx,yy)

       
        # Create new colormap
        colors_1 = [(252-32*i,252-32*i,252-32*i,i*1/16) for i in np.linspace(0, 1, 32)]
        colors_2 = [(220-60*i,220-60*i,220,i*1/16+1/16) for i in np.linspace(0, 1, 32)]
        colors_3 = [(160-20*i,160+30*i,220,i*3/8+1/8) for i in np.linspace(0, 1, 96)]
        colors_4 = [(140+80*i,190+60*i,220+30*i,i*4/8+4/8) for i in np.linspace(0, 1, 96)]
        colors = colors_1 + colors_2 + colors_3 + colors_4

        colors = list(map(lambda c: (c[0]/256,c[1]/256,c[2]/256,c[3]), colors))
        self.my_cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', colors, N=64)

        #print once so that everything is set up
        self.my_map.bluemarble()
        self.my_map.drawcoastlines()
        
    
    def plot(self, filename, title_prefix, data, label, year, month, day, hour):
        
        # Get data
        tstart = time.time()
        data = np.roll(data,[0,int(1152/2)])
        
        # Get labels
        label = np.roll(label, [0,int(1152/2)])
        l1 = (label == 1)
        l2 = (label == 2)
        print("extract data: {}".format(time.time() - tstart))

        #pdf
        #with PdfPages(filename+'.pdf') as pdf:
        
        #get figure
        fig = plt.figure(figsize=(100,20), dpi=100)
        
        #draw stuff
        tstart = time.time()
        self.my_map.bluemarble()
        self.my_map.drawcoastlines()
        print("draw background: {}".format(time.time() - tstart))
        
        # Plot data
        tstart = time.time()
        self.my_map.contourf(self.x_map, self.y_map, data, 128, vmin=0, vmax=89, cmap=self.my_cmap, levels=np.arange(0,89,2))
        print("draw data: {}".format(time.time() - tstart))
        
        # Plot colorbar
        tstart = time.time()
        cbar = self.my_map.colorbar(ticks=np.arange(0,89,11))
        cbar.ax.set_ylabel('Integrated Water Vapor kg $m^{-2}$',size=32)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=28)
        print("draw colorbar: {}".format(time.time() - tstart))
        
        # Draw Tropical Cyclones & Atmospheric Rivers
        tstart = time.time()
        tc_contour = self.my_map.contour(self.x_map, self.y_map, l1, [0.5], linewidths=3, colors='orange')
        ar_contour = self.my_map.contour(self.x_map, self.y_map, l2, [0.5], linewidths=3, colors='magenta')
        print("draw contours: {}".format(time.time() - tstart))
        
        tstart = time.time()
        self.my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
        self.my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
        print("draw meridians: {}".format(time.time() - tstart))
    
        # Plot legend and title
        tstart = time.time()
        lines = [tc_contour.collections[0], ar_contour.collections[0]]
        labels = ['Tropical Cyclone', "Atmospheric River"]
        plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='38')
        plt.title("{} Extreme Weather Patterns {:04d}-{:02d}-{:02d}".format(title_prefix, int(year), int(month), int(day)), fontdict={'fontsize': 44})
        print("draw legend/title: {}".format(time.time() - tstart))
        
        tstart = time.time()
        #pdf.savefig(bbox_inches='tight')
        #mask_ex = plt.gcf()
        #mask_ex.savefig(filename, bbox_inches='tight')
        plt.gcf().savefig(filename, format="PNG", bbox_inches='tight')
        plt.clf()
        print("save plot: {}".format(time.time() - tstart))
        
