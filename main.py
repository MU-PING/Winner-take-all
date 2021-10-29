# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 01:10:56 2020
@author: MU-PING
"""

import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

def gen_data():
    global data
    data=[]
    plt.clf()
    plt.title("Data")
    for _ in range(neurons.get()): #群數
        center_x = np.random.randint(-31, 31)
        center_y = np.random.randint(-31, 31)
        for _ in range(np.random.randint(100, 150)): #一群的點數
            new_x = center_x + np.random.uniform(-30, 30)
            new_y = center_y + np.random.uniform(-30, 30)
            data.append([new_x, new_y])
            plt.plot(new_x, new_y, 'o', ms=3 , color = 'gray', alpha=.8) #畫圖 ms：折點大小
    data = np.array(data)
    canvas1.draw()
    
def Euclid():
    weight = np.random.rand(2, neurons.get())
    for _ in range(epoch.get()):
        for i in data:
            ans=[]
            for j in range(neurons.get()):
                ans.append(((i[0]-weight.T[j][0])**2 + (i[1]-weight.T[j][1])**2)**0.5)
            winner = np.argmin(np.array(ans))
            weight.T[winner] = weight.T[winner] + lr.get()*(i-weight.T[winner])

    #劃出結果
    plt.clf()
    plt.title("Data")
    for i in data:
        ans=[]
        for j in range(neurons.get()):
            ans.append(((i[0]-weight.T[j][0])**2 + (i[1]-weight.T[j][1])**2)**0.5)
        winner = np.argmin(np.array(ans))
        plt.plot(i[0], i[1], 'o', ms=3 , color = color[winner], alpha=.8) #畫圖 ms：折點大小
    canvas1.draw()
    
def Cosθ():
    weight = np.random.rand(2, neurons.get())
    for _ in range(epoch.get()):
        for i in data:
            ans=[]
            for j in range(neurons.get()):
                ans.append(i.dot(weight.T[j])/(i[0]**2+i[1]**2)**0.5/(weight.T[j][0]**2+weight.T[j][1]**2)**0.5)
            winner = np.argmax(np.array(ans))
            weight.T[winner] = weight.T[winner] + lr.get()*(i-weight.T[winner])
    #劃出結果
    plt.clf()
    plt.title("Data")
    for i in data:
        ans=[]
        for j in range(neurons.get()):
            ans.append(i.dot(weight.T[j])/(i[0]**2+i[1]**2)**0.5/(weight.T[j][0]**2+weight.T[j][1]**2)**0.5)
        winner = np.argmax(np.array(ans))
        plt.plot(i[0], i[1], 'o', ms=3 , color = color[winner], alpha=.8) #畫圖 ms：折點大小
    canvas1.draw()
    
def Dot():
    weight = np.random.rand(2, neurons.get())
    for _ in range(epoch.get()):
        for i in data:
            winner = np.argmax(i.dot(weight))
            weight.T[winner] = weight.T[winner] + lr.get()*(i-weight.T[winner])
    #劃出結果
    plt.clf()
    plt.title("Data")
    for i in data:
        plt.plot(i[0], i[1], 'o', ms=3 , color = color[np.argmax(i.dot(weight))], alpha=.8) #畫圖 ms：折點大小
    canvas1.draw()
    
def start():
    if(data_combobox.current()==0):
        Euclid()
        
    elif(data_combobox.current()==1):
        Cosθ()
        
    else:
        Dot()
        
window = tk.Tk()
window.geometry("520x390")
window.resizable(False, False)
window.title("競爭式學習法 - 單層神經網路")

#全域變數
lr = tk.IntVar()#學習率
lr.set(1)
neurons = tk.IntVar()#神經元個數
neurons.set(3)
epoch = tk.IntVar()#訓練次數
epoch.set(10)
data = []
color = ["#FF0000", "#0000E3", "#FFD306", "#9F4D95", "#00DB00", "#5CADAD", "#FF8000", "#FF0080"]

classification = ["歐基里德距離", "cosθ值", "內積"]
setting1 = tk.Frame(window)
setting1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.N)
tk.Label(setting1, font=("微軟正黑體", 12, "bold"), text="學習率：").grid(row=0, sticky=tk.W, pady=5)
tk.Entry(setting1, width=10, textvariable=lr).grid(row=1, sticky=tk.W)
tk.Label(setting1, font=("微軟正黑體", 12, "bold"), text="神經元個數：").grid(row=2, sticky=tk.W, pady=5)
tk.Entry(setting1, width=10, textvariable=neurons).grid(row=3, sticky=tk.W)
tk.Label(setting1, font=("微軟正黑體", 12, "bold"), text="訓練次數").grid(row=4, sticky=tk.W, pady=5)
tk.Entry(setting1, width=10, textvariable=epoch).grid(row=5, sticky=tk.W)
tk.Label(setting1, font=("微軟正黑體", 12, "bold"), text="分類標準").grid(row=6, sticky=tk.W, pady=4)
data_combobox = ttk.Combobox(setting1, value = classification, state="readonly", width=12) #readonly為只可讀狀態
data_combobox.current(0)
data_combobox.grid(row=7, sticky=tk.W)
btn = tk.Button(setting1, text='隨機產生資料', command = gen_data)
btn.grid(row=8, sticky=tk.W, pady=20)
btn = tk.Button(setting1, text='開始訓練', command = start)
btn.grid(row=9, sticky=tk.W)

setting2 = tk.Frame(window)
setting2.grid(row=0, column=1, pady=10)
fig = plt.figure(figsize=(5,5))
plt.title("Data")
canvas1 = FigureCanvasTkAgg(fig, setting2)  # A tk.DrawingArea.
canvas1.get_tk_widget().grid()

window.mainloop()
