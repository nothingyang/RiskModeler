import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
from pandastable import Table
import pickle as pickle
import os
from .inputdata import inputdata
from .split import spliting
from .sampling import sample
from .interactive_grouping import IGN
from .model_ui import model
from .score_ui import  scoreing
from .load_node import import_node
from tkinter import filedialog
from os.path import exists
import datetime
class scorecard():
    def __init__(self):
        self.row = 0
        self.col = 0
        self.project_name = None
        self.project_path = None

        self.project_seting = {}
        self.project_detail = pd.DataFrame(columns=['模块类型', '模块名字', '引用模块', '保存地址', '状态','创建时间'])
        self.root = Tk()

        self.Start_UI()
        self.root.withdraw()
        self.root.mainloop()
    def Start_UI(self):
        self.start_window_base = Toplevel(self.root)
        self.start_window_base.title('项目')
        self.start_window = LabelFrame(self.start_window_base, text='创建新项目')
        name = tk.StringVar(value='scorecard1')
        width = 500
        height = 200

        def selectExcelfold():
            sfname = filedialog.askdirectory()
            self.project_path_E.insert(INSERT, sfname)

        screenwidth = self.start_window.winfo_screenwidth()
        screenheight = self.start_window.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.start_window_base.geometry(alignstr)

        L1 = Label(self.start_window, text="项目路径")
        L1.grid(column=0, row=0, sticky=(W))
        self.project_path_E = Entry(self.start_window, width=50, bd=1)
        self.project_path_E.grid(column=1, row=0, sticky=(W))
        button1 = ttk.Button(self.start_window, text='浏览', width=8, command=selectExcelfold)
        button1.grid(column=2, row=0, sticky=(W))

        L2 = Label(self.start_window, text="项目名称")
        L2.grid(column=0, row=1, sticky=(W))
        self.project_name_E = Entry(self.start_window, textvariable=name, bd=1)
        self.project_name_E.grid(column=1, row=1, sticky=(W))

        test_button4 = ttk.Button(self.start_window, text='确定')
        test_button4.grid(column=1, row=2, sticky=(W))
        test_button4.bind("<Button-1>", self.new_project)
        self.start_window.grid(column=0, row=0, columnspan=2, rowspan=3)

        def selectExcelfile():
            sfname = filedialog.askopenfilename(title='选择project文件', filetypes=[('project', '*.project')])
            self.project_path_Ex.insert(INSERT, sfname)

        self.start_window_ex = LabelFrame(self.start_window_base, text='导入现有项目')

        L5 = Label(self.start_window_ex, text="项目路径")
        L5.grid(column=0, row=4, sticky=(W))
        self.project_path_Ex = Entry(self.start_window_ex, width=50, bd=1)
        self.project_path_Ex.grid(column=1, row=4, sticky=(W))
        button1 = ttk.Button(self.start_window_ex, text='浏览', width=8, command=selectExcelfile)
        button1.grid(column=2, row=4, sticky=(W))

        test_button5 = ttk.Button(self.start_window_ex, text='导入')
        test_button5.grid(column=1, row=5, sticky=(W))
        test_button5.bind("<Button-1>", self.load_project)
        self.start_window_ex.grid()

    def load_project(self, event):
        try:
            project_add = self.project_path_Ex.get()
            fr = open(project_add, 'rb')
            project_info = pickle.load(fr)
            fr.close()
            self.project_detail = project_info
            self.project_path = project_add
            self.project_name = self.project_detail[self.project_detail['模块类型'] == 'project']['模块名字'][0]
            self.project_detail['保存地址'][self.project_detail['模块类型'] == 'project']=self.project_path
            self.start_window_base.destroy()
            self.base_UI()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)

    def new_project(self, event):
        self.project_name = self.project_name_E.get()
        self.project_path = self.project_path_E.get()+ '/' + '%s.project' % self.project_name
        if exists(self.project_path)==False:
            self.project_seting = {'project_name': self.project_name, 'project_path': self.project_path}
            tt = [{'模块类型': 'project',
                   '模块名字': self.project_name,
                   '引用模块': [],
                   '保存地址': self.project_path,
                   '状态': 'Good',
                   '创建时间':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') }]
            mm = pd.DataFrame(tt)
            self.project_detail = self.project_detail.append(mm)

            try:
                self.save_project()
            except Exception as e:
                tk.messagebox.showwarning('错误', e)
                self.start_window_base.destroy()
                self.root.destroy()
                self.__init__()
            self.start_window_base.destroy()
            self.base_UI()
        else:
            tk.messagebox.showwarning('错误', '在文件夹下有同名项目')

    def base_UI(self):

        self.root.update()
        self.root.deiconify()
        width = 1000
        height = 600
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)

        self.root.title(self.project_name)
        menubar = Menu(self.root)
        sysmenu_inputdata = Menu(menubar, tearoff=False)
        sysmenu_save = Menu(menubar, tearoff=False)
        sysmenu_data_deal = Menu(menubar, tearoff=False)
        sysmenu_IGN = Menu(menubar, tearoff=False)
        sysmenu_model = Menu(menubar, tearoff=False)
        sysmenu_load_node = Menu(menubar, tearoff=False)

        menubar.add_cascade(label='保存/刷新', menu=sysmenu_save)
        sysmenu_save.add_command(label='保存项目', command=lambda: self.save_project())
        sysmenu_save.add_command(label='刷新', command=lambda: self.refresh_df(self.root, self.project_detail))

        menubar.add_cascade(label='导入模块', menu=sysmenu_load_node)
        sysmenu_load_node.add_command(label='导入', command=lambda: self.func_menu('load_node'))

        menubar.add_cascade(label='导入数据集', menu=sysmenu_inputdata)
        sysmenu_inputdata.add_command(label='添加', command=lambda: self.func_menu('importdata'))

        menubar.add_cascade(label='数据集处理', menu=sysmenu_data_deal)
        sysmenu_data_deal.add_command(label='分区', command=lambda: self.func_menu('split'))
        sysmenu_data_deal.add_command(label='抽样', command=lambda: self.func_menu('sampling'))

        menubar.add_cascade(label='交互分组', menu=sysmenu_IGN)
        sysmenu_IGN.add_command(label='单变量分组', command=lambda: self.func_menu('IGN'))

        menubar.add_cascade(label='评分卡', menu=sysmenu_model)
        sysmenu_model.add_command(label='训练模型', command=lambda: self.func_menu('model'))
        sysmenu_model.add_command(label='数据集打分', command=lambda: self.func_menu('Scoring'))

        self.root.grid()
        self.root.config(menu=menubar)
        self.refresh_df(self.root, self.project_detail)
        self.root.update()

    def save_project(self):

        filename = self.project_path
        fw = open(filename, 'wb')
        pickle.dump(self.project_detail, fw, 1)
        fw.close()

    def func_menu(self, func):
        try:
            if self.root2.state() == 'normal':
                tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
        except:
            self.root2 = Toplevel(self.root)
            if func == 'importdata':
                self.root2.title('导入数据集')
                new_node = inputdata(self.root2, self.project_detail)
                new_node.newdatainput()
                tip = '导入数据集'
            elif func == 'split':
                self.root2.title('数据集分区')
                new_node = spliting(self.root2, self.project_detail)
                new_node.ui_start()
                tip = '数据集分区'
            elif func == 'sampling':
                self.root2.title('数据集抽样')
                new_node = sample(self.root2, self.project_detail)
                new_node.ui_start()
                tip = '数据集抽样'
            elif func == 'IGN':
                self.root2.title('交互式分组')
                new_node = IGN(self.root2, self.project_detail)
                new_node.Start_UI()
                new_node.adjustsetting()
                tip = '交互式分组'
            elif func == 'model':
                self.root2.title('评分卡模型')
                new_node = model(self.root2, self.project_detail)
                new_node.Start_UI()
                new_node.adjustsetting()
                tip = '评分卡模型'
            elif func == 'Scoring':
                self.root2.title('数据集打分')
                new_node = scoreing(self.root2, self.project_detail)
                new_node.Start_UI()
                new_node.adjustsetting()
                tip = '数据集打分'
            elif func == 'load_node':
                new_node = import_node(self.root2, self.project_detail)
                tip = '导入模块'
            self.root.wait_window(self.root2)

            if new_node.save != 'Y':
                tk.messagebox.showwarning('错误', "%s未完成" % tip)
            else:
                try:
                    print(new_node.save)
                    tt = [{'模块类型': new_node.node_setting['node_type'],
                           '模块名字': new_node.node_setting['node_name'],
                           '引用模块': new_node.node_setting['use_node'],
                           '保存地址': new_node.node_setting['node_save_path'],
                           '创建时间': new_node.node_setting['time'],
                           '状态': 'Good'}]

                    mm = pd.DataFrame(tt)
                    print(mm)
                    self.project_detail = self.project_detail.append(mm)
                    # del new_node
                    self.refresh_df(self.root, self.project_detail)
                except Exception as e:
                    tk.messagebox.showwarning('错误', "%s未完成%s" % (tip, e))
    def refresh_check(self, node_save_path):

        p2 = ttk.Label(self.root, text='checking.... \n wait.....')

        p2.grid(row=0, column=0)
        self.root.update()
        try:
            fr = open(node_save_path, 'rb')

            fr.close()

            p2.destroy()
            return 'Good'

        except Exception as e:

            p2.destroy()
            return 'error'

    def refresh_df(self, mianfram, df):
        try:
            self.save_project()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)
        df['状态'] = df.apply(lambda x: 'Good' if x['模块类型'] == 'project' else self.refresh_check(x['保存地址']), axis=1)
        df = df[['模块类型', '模块名字', '引用模块', '保存地址', '状态','创建时间']]
        f = Frame(mianfram)
        f.grid(column=0, row=1, rowspan=1,
               columnspan=5, sticky=(E, W))
        screen_width = f.winfo_screenwidth() * 0.8
        screen_height = f.winfo_screenheight() * 0.8
        self.table = self.ptm = Table(f, dataframe=df, height=screen_height, width=screen_width)
        self.ptm.show()
        self.table.grid()
        self.table.bind("<Button-3>", self.right_click_menu)
        self.table.bind("<Button-2>", self.right_click_menu)
        # self.table.bind("<Button-1>", self.right_click_menu)
        self.table.bind("<Double-Button-3>", self.right_click_menu)
        self.table.bind("<Double-Button-1>", self.right_click_menu)
        self.table.bind("<Double-Button-2>", self.right_click_menu)
        self.table.bind("<Triple-Button-3>", self.right_click_menu)
        self.table.bind("<Triple-Button-1>", self.right_click_menu)
        self.table.bind("<Triple-Button-2>", self.right_click_menu)
    def right_click_menu(self, event):
        rowclicked = self.ptm.get_row_clicked(event)
        colclicked = self.ptm.get_col_clicked(event)
        menu = Menu(self.root)
        sysmenu_inputdata = Menu(menu, tearoff=False)
        menu.add_command(label="设置", command=lambda: self.setting(rowclicked, colclicked))
        menu.add_separator()
        menu.add_command(label="结果", command=lambda: self.result(rowclicked, colclicked))
        menu.add_separator()
        menu.add_command(label="删除", command=lambda: self.delet(rowclicked, colclicked))

        menu.post(event.x_root, event.y_root)
        # self.root.update()

    def setting(self, rowclicked, colclicked):

        # data_variable_setting.iloc[self.rowclicked,self.colclicked]
        node_type = self.project_detail.iloc[rowclicked]['模块类型']
        node_name = self.project_detail.iloc[rowclicked]['模块名字']
        node_save_path = self.project_detail.iloc[rowclicked]['保存地址']
        try:
            fr = open(node_save_path, 'rb')
            node_info = pickle.load(fr)
            fr.close()
            flag_error = 0
        except Exception as e:
            flag_error = 1
            tk.messagebox.showwarning('错误', e)
        if flag_error != 1:
            try:
                if self.root2.state() == 'normal':
                    tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
            except:
                self.root2 = Toplevel(self.root)
                self.root2.title(node_name)
                # try:
                if node_type == 'DATA':
                    new_node = inputdata(self.root2, self.project_detail)
                    new_node.load(node_info)
                    new_node.variable_seting_ui()
                elif node_type == 'SPLIT':
                    new_node = spliting(self.root2, self.project_detail)
                    new_node.load_node(node_data=node_info, ac='setting')
                elif node_type == 'SAMPLE':
                    new_node = sample(self.root2, self.project_detail)
                    new_node.load_node(node_data=node_info, ac='setting')
                elif node_type == 'IGN':
                    new_node = IGN(self.root2, self.project_detail)
                    new_node.load_node(node_info,ac='setting')
                elif node_type == 'SCR':
                    new_node = model(self.root2, self.project_detail)
                    new_node.import_node( node_info,ac='setting')
                elif node_type == 'Scoring':
                    new_node = scoreing(self.root2, self.project_detail)
                    new_node.load_node(node_info,ac='setting')

                self.root.wait_window(self.root2)
                try:
                    tt = [{'模块类型': new_node.node_setting['node_type'],
                           '模块名字': new_node.node_setting['node_name'],
                           '引用模块': new_node.node_setting['use_node'],
                           '保存地址': new_node.node_setting['node_save_path'],
                           '创建时间': new_node.node_setting['time'],
                           '状态': 'Good'}]

                    mm = pd.DataFrame(tt)
                    print(mm)
                    self.project_detail = self.project_detail[self.project_detail['模块名字'] != node_name]
                    self.project_detail = self.project_detail.append(mm)
                    self.refresh_df(self.root, self.project_detail)
                except:
                    pass
                # except Exception as e:
                #     tk.messagebox.showwarning('错误', e)

    def result(self, rowclicked, colclicked):
        node_type = self.project_detail.iloc[rowclicked]['模块类型']
        node_name = self.project_detail.iloc[rowclicked]['模块名字']
        node_save_path = self.project_detail.iloc[rowclicked]['保存地址']
        try:
            fr = open(node_save_path, 'rb')
            node_info = pickle.load(fr)
            fr.close()
            flag_error = 0
        except Exception as e:
            flag_error = 1
            tk.messagebox.showwarning('错误', e)
        if flag_error != 1:
            try:
                if self.root2.state() == 'normal':
                    tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
            except:
                self.root2 = Toplevel(self.root)
                self.root2.title(node_name)
                if node_type == 'SPLIT':
                    new_node = spliting(self.root2, self.project_detail)
                    new_node.load_node(node_data=node_info, ac='result')
                elif node_type == 'DATA':
                    new_node = inputdata(self.root2, self.project_detail)
                    new_node.load(node_info)
                    new_node.variable_seting_ui()
                elif node_type == 'SAMPLE':
                    new_node = sample(self.root2, self.project_detail)
                    new_node.load_node(node_data=node_info, ac='result')
                elif node_type == 'IGN':
                    new_node = IGN(self.root2, self.project_detail)
                    new_node.load_node(node_info, ac='result')
                    new_node.result()
                elif node_type=='SCR':
                    new_node = model(self.root2, self.project_detail)
                    new_node.import_node(node_info,ac='result')
                    new_node.reult_show_only(self.root2)
                elif node_type == 'Scoring':
                    new_node = scoreing(self.root2, self.project_detail)
                    new_node.load_node( node_info,ac='result')
                self.root.wait_window(self.root2)
                try:
                    tt = [{'模块类型': new_node.node_setting['node_type'],
                           '模块名字': new_node.node_setting['node_name'],
                           '引用模块': new_node.node_setting['use_node'],
                           '保存地址': new_node.node_setting['node_save_path'],
                           '创建时间': new_node.node_setting['time'],
                           '状态': 'Good'}]
                    mm = pd.DataFrame(tt)
                    print(mm)
                    self.project_detail = self.project_detail[self.project_detail['模块名字'] != node_name]
                    self.project_detail = self.project_detail.append(mm)
                    self.refresh_df(self.root, self.project_detail)
                except:
                    pass

    def delet(self, rowclicked, colclicked):

        node_name = self.project_detail.iloc[rowclicked]['模块名字']
        node_save_path = self.project_detail.iloc[rowclicked]['保存地址']
        try:
            os.remove(node_save_path)
            self.project_detail = self.project_detail[self.project_detail['模块名字'] != node_name]
            self.refresh_df(self.root, self.project_detail)
        except  Exception as e:
            tk.messagebox.showwarning('错误', e)
            self.project_detail = self.project_detail[self.project_detail['模块名字'] != node_name]
            self.refresh_df(self.root, self.project_detail)


# %%


