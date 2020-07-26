import tkinter as tk
from tkinter import ttk
from tkinter import *

import pickle as pickle

from tkinter import filedialog
class import_node():
    def __init__(self, mainfram, project_info):
        self.root2=mainfram
        self.node_name = None
        self.node_type = None
        self.node_save_path = None
        self.node_current_save_path=None
        self.project_path = project_info[project_info['模块类型'] == 'project']['保存地址'][0]
        self.exist_data = list(project_info['模块名字'])
        self.exist_add = list(project_info['保存地址'])
        self.master = mainfram
        self.load_node()
        self.save = 'N'
    def load_node(self):
            width = 500
            height = 250
            screenwidth = self.root2.winfo_screenwidth()
            screenheight = self.root2.winfo_screenheight()
            alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
            self.root2.geometry(alignstr)

            def selectExcelfile():
                sfname = filedialog.askopenfilename(title='选择模块文件', filetypes=[('IGN', '*.IGN'),('DATASET', '*.dataset'),('Sampling', '*.sample'),('Spliting', '*.spliting'),('Model','model')])
                self.node_current_save_path=sfname
                self.nodeimport_E1.delete(0, 'end')
                self.nodeimport_E1.insert(INSERT, sfname)
                try:
                    fr = open(sfname, 'rb')
                    node_info = pickle.load(fr)
                    self.node_data = node_info
                    fr.close()
                    try:
                        self.node_name=node_info[0]['node_name']
                        self.node_type=node_info[0]['node_type']
                        self.use_node = node_info[0]['use_node']
                        self.node_time=node_info[0]['time']
                        self.node_save_path=node_info[0]['node_save_path']
                        self.nodeimport_E2.delete(0, 'end')
                        self.nodeimport_E2.insert(INSERT, self.node_name)
                        self.label_str.set(self.node_type)
                    except Exception as e:
                        tk.messagebox.showwarning('错误', e)
                except Exception as e:
                    tk.messagebox.showwarning('错误', e)
            L1 = Label(self.root2, text="模块路径")
            L1.grid(column=0, row=0, columnspan=2, sticky=(W))
            self.nodeimport_E1 = Entry(self.root2, width=50,  bd=1)
            self.nodeimport_E1.grid(column=1, row=0, sticky=(W))
            button1 = ttk.Button(self.root2, text='浏览', width=8, command=selectExcelfile)
            button1.grid(column=2, row=0, sticky=(W))

            L1 = Label(self.root2, text="模块名称")
            L1.grid(column=0, row=1, columnspan=2, sticky=(W))
            self.nodeimport_E2 = Entry(self.root2,  width=23, bd=1)
            self.nodeimport_E2.grid(column=1, row=1, sticky=(W))

            L3 = Label(self.root2, text="模块类型")
            L3.grid(column=0, row=2, sticky=(W))
            self.label_str = StringVar()
            warning = Label(self.root2, textvariable=self.label_str)
            warning.grid(column=1, row=2, sticky=(W))


            test_button4 = ttk.Button(self.root2, text='确定')
            test_button4.grid(column=1, row=5, sticky=(W))
            test_button4.bind("<Button-1>", self.save_node)

    def save_node(self,event):
        flag_error=0
        if self.nodeimport_E1.get()!=self.node_current_save_path:
            self.node_current_save_path=self.nodeimport_E1.get()
            try:
                fr = open(self.nodeimport_E1.get(), 'rb')
                node_info = pickle.load(fr)
                self.node_data=node_info
                fr.close()
                try:
                    self.node_name = node_info[0]['node_name']
                    self.node_type = node_info[0]['node_type']
                    self.use_node = node_info[0]['use_node']
                    self.node_time = node_info[0]['time']
                    # self.node_save_path = node_info[0]['node_save_path']
                    self.nodeimport_E2.delete(0, 'end')
                    self.nodeimport_E2.insert(INSERT, self.node_name)
                    self.label_str.set(self.node_type)
                except Exception as e:
                    flag_error = 1
                    tk.messagebox.showwarning('错误', e)
            except Exception as e:
                flag_error = 1
                tk.messagebox.showwarning('错误', e)
        if flag_error==0:
            if self.nodeimport_E2.get() in self.exist_data:
                tk.messagebox.showwarning('错误', '该名称已经在project中，请改名')
            elif self.node_current_save_path in self.exist_add:
                tk.messagebox.showwarning('错误', '该地址已经在project中，请勿重复导入')
            elif self.nodeimport_E2.get() != self.node_name:
                self.node_name= self.nodeimport_E2.get()
            else:
                self.node_setting={'node_type':self.node_type,'node_name':self.node_name,'use_node':self.use_node,'node_save_path':self.node_current_save_path,'time':self.node_time}
                self.save = 'Y'
                self.root2.destroy()
