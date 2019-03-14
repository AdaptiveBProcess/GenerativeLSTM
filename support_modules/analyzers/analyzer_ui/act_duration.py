# -*- coding: utf-8 -*-
import support as sup
import matplotlib as mpl
#mpl.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

#import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import ttk

LARGE_FONT= ("Verdana", 12)

class ActDuration(tk.Frame):

    def __init__(self, parent, controller, task_duration):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Activity level", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame('StartPage'))
        button1.pack()

        #----graph----
        #----- Constants -----
        sources =  sorted(list(set([x['source'] for x in task_duration])))
        width = 0.8/len(sources)         # the width of the bars
        bars_distances = sup.create_symetric_list(width, len(sources))
        bars_colors = ['darkgrey','royalblue','darkseagreen','rosybrown','green']
        labels = [y['alias'] for y in list(filter(lambda x: x['source'] == 'log', task_duration))]
        #-----figure 1: Processing time ----
        #----- Data -----
        series, confidence = dict(), dict()
        for source in sources:
            series[source] = [y['processing_time'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
            if source != 'log':
                confidence[source] = [y['pmci'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
        #----- figure ----
        fig = mpl.figure.Figure(figsize=(9, 5), dpi=100)
        ax = fig.add_subplot(211)
        ind = np.arange(len(series['log']))    # the x locations for the groups
        bars_series = dict(log=ax.bar(ind + bars_distances[0], series['log'], width, color='k', bottom=0))
        series_sources = sorted(list(set([x['source'] for x in task_duration])))
        series_sources.remove('log')
        for i in range(0 , len(series_sources)):
            source = series_sources[i]
            bars_series[source] = ax.bar(ind + bars_distances[i+1], series[source], width, color=bars_colors[i], bottom=0)
                #yerr=confidence[source], ecolor='r', capsize=3)
        ax.set_title('Processing time')
        ax.set_xticks(ind)
        ax.set_xticklabels(labels)
        ax.set_yscale('symlog')
        rectangles, series_names = list(), list()
        for source in sources:
            rectangles.append(bars_series[source][0])
            series_names.append(source)
        ax.legend(rectangles, series_names, loc=1, fontsize='xx-small')
        ax.autoscale_view()

        #-----figure 2: Waiting time ----
        #----- Data -----
        series, confidence = dict(), dict()
        for source in sources:
            series[source] = [y['waiting_time'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
            if source != 'log':
                confidence[source] = [y['wmci'] for y in list(filter(lambda x: x['source'] == source, task_duration))]
        #----- figure ----
        ax2 = fig.add_subplot(212)
        ind = np.arange(len(series['log']))    # the x locations for the groups
        bars_series = dict(log=ax2.bar(ind + bars_distances[0], series['log'], width, color='k', bottom=0))
        series_sources = sorted(list(set([x['source'] for x in task_duration])))
        series_sources.remove('log')
        for i in range(0 , len(series_sources)):
            source = series_sources[i]
            bars_series[source] = ax2.bar(ind + bars_distances[i+1], series[source], width, color=bars_colors[i], bottom=0)
               # yerr=confidence[source], ecolor='r', capsize=3)
        ax2.set_title('Waiting time')
        ax2.set_xticks(ind)
        ax2.set_xticklabels(labels)
        ax2.set_yscale('symlog')
        rectangles, series_names = list(), list()
        for source in sources:
            rectangles.append(bars_series[source][0])
            series_names.append(source)
        ax2.legend(rectangles, series_names, loc=1, fontsize='xx-small')
        ax2.autoscale_view()

        fig.subplots_adjust(hspace=0.5)

        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.
            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """
            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.5, 'left': 0.5}  # x_txt = x + w*off
            for rect in rects:
                height = rect.get_height()
                if height < 0:
                    ypost = 1.8*height
                else:
                    ypost = 1.01*height
                ax.text(rect.get_x() + rect.get_width()*offset[xpos], ypost,
                        '{}'.format(height)+'%', ha=ha[xpos], va='bottom', color=rects.patches[0].get_facecolor())
        #
        # #autolabel(p1, "center")
        # #autolabel(p2, "center")

        #----drawing----
        canvas = FigureCanvasTkAgg(fig, self)

        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
