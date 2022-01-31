import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
from pandastable import Table
import pickle as pickle
from .func import binning
import math
from .IGN_UI import UserInterfacea
import datetime
from tkinter import filedialog
import os
binning = binning()
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
class PLC():
    def __init__(self, mianframe, project_info):
        self.master = mianframe
        # project参数
        self.save = 'N'
        self.project_info = project_info
        self.project_path = os.path.split(project_info[project_info['模块类型'] == 'project']['保存地址'][0])[0]
        self.node_name='policy'
        self.exist_data = list(project_info['模块名字'])
        self.load='N'
        self.finsh='N'
        # 数据参数

        self.par_train_data = pd.DataFrame()
        self.par_train_dataname = None
        self.par_train_dataname_time = None
        self.par_traindatavariable_setting = pd.DataFrame()
        self.grouped_train_data = pd.DataFrame()

        self.par_valid_data = pd.DataFrame()
        self.par_valid_dataname = None
        self.grouped_valid_data = pd.DataFrame()

        self.par_oot_data = pd.DataFrame()
        self.par_oot_dataname = None
        self.par_oot_dataname_time = None
        self.par_ootdatavariable_setting = pd.DataFrame()
        self.grouped_oot_data = pd.DataFrame()

        self.par_reject_data = pd.DataFrame()
        self.par_reject_dataname = None
        self.par_reject_dataname_time = None
        self.par_rejectdatavariable_setting = pd.DataFrame()
        self.grouped_reject_data = pd.DataFrame()

        # 分组参数
        self.par_use_freezing_flag = '否'
        self.par_import_groupdataname=None

        self.par_use_specialcode_flag = '否'
        self.par_specialcode_data = pd.DataFrame()
        self.par_sepcialcode_dataname = None
        self.par_char_restric_flag = '不限制'
        self.par_char_restric_num = 500
        self.par_char_restric_pct = 0.005
        self.par_tree_criterion = 'entropy'
        self.par_num_f_group = 8
        self.par_min_num_group = 500
        self.par_min_pct_group = 0.01
        self.par_variable_reject_flag = 'no'
        self.par_variable_reject_iv = 0.05
        # policy 参数
        self.par_min_lift = 1.4
        self.par_min_badnum = 5
        self.par_min_badrate = 0.1
        self.par_num_s_bin = 20
        # 分组过程参数
        self.IGNvariable_setting = pd.DataFrame()
        self.target_train = None
        self.target_reject = None
        self.target_oot = None
        self.flag_timeid_oot = False
        self.timeid_oot = None
        self.flag_timeid_reject = False
        self.timeid_reject = None
        self.flag_timeid_train = False
        self.timeid_train = None
        self.varnum = []
        self.varchar = []
        self.varnum_reject = []
        self.varchar_reject = []
        self.varnum_oot = []
        self.varchar_oot = []
        self.groupingdata = pd.DataFrame()

        self.pre_data()
   # 模块参数
        self.previous_train_check_change =[]
        self.previous_train_node_usedlist =[]

        self.previous_reject_check_change =[]
        self.previous_reject_node_usedlist =[]

        self.previous_oot_check_change =[]
        self.previous_oot_node_usedlist =[]
    def load_data(self, event, datatype):
        try:
            if datatype == 'train':
                path = self.project_info[self.project_info['模块名字'] == self.comboxlist_train_data.get()]['保存地址'][0]
                fr = open(path, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                self.par_traindatavariable_setting = node_info[0]['data_variable_setting']
                self.par_train_dataname = node_info[0]['node_name']
                self.par_train_dataname_time=node_info[0]['time']
                self.par_train_data = node_info[1]
                # self.previous_train_check_change = node_info[0]['check_change']
                self.previous_train_node_usedlist = node_info[0]['use_node']
                if node_info[0]['node_type'] == 'SPLIT':
                    self.par_valid_data = node_info[2]
                else:
                    self.par_valid_data = pd.DataFrame()
                    self.par_valid_dataname = None
                    self.grouped_valid_data = pd.DataFrame()
                self.IGNvariable_setting = self.par_traindatavariable_setting.copy()
                if self.par_reject_data.empty != True:
                    reject = self.par_rejectdatavariable_setting.copy()
                    reject['变量名称_拒绝样本'] = reject['变量名称']
                    reject['变量类型_拒绝样本'] = reject['变量类型']
                    reject['是否使用_拒绝样本'] = reject['是否使用']
                    reject['变量角色_拒绝样本'] = reject['变量角色']
                    reject['备注_拒绝样本'] = reject['备注']
                    reject = reject.drop(columns=['备注', '变量名称', '是否使用', '变量角色', '变量类型'])
                    self.IGNvariable_setting = pd.merge(self.IGNvariable_setting, reject,
                                                        how='outer', left_on='变量名称', right_on='变量名称_拒绝样本')
                if self.par_oot_data.empty != True:
                    oot = self.par_ootdatavariable_setting.copy()
                    oot['变量名称_时间外样本'] = oot['变量名称']
                    oot['变量类型_时间外样本'] = oot['变量类型']
                    oot['是否使用_时间外样本'] = oot['是否使用']
                    oot['变量角色_时间外样本'] = oot['变量角色']
                    oot['备注_时间外样本'] = oot['备注']
                    oot = oot.drop(
                        columns=['备注', '变量名称', '是否使用', '变量角色', '变量类型'])
                    self.IGNvariable_setting = pd.merge(self.IGNvariable_setting, oot,
                                                        how='outer', left_on='变量名称', right_on='变量名称_时间外样本')
            elif (datatype == 'reject')&(str(self.comboxlist_reject_data.get())!='NO'):
                path = self.project_info[self.project_info['模块名字'] == self.comboxlist_reject_data.get()]['保存地址'][0]
                fr = open(path, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                self.par_rejectdatavariable_setting = node_info[0]['data_variable_setting']
                self.par_reject_dataname = node_info[0]['node_name']
                self.par_reject_dataname_time=node_info[0]['time']
                self.par_reject_data = node_info[1]
                # self.previous_reject_check_change = node_info[0]['check_change']
                self.previous_reject_node_usedlist = node_info[0]['use_node']
                try:
                    self.IGNvariable_setting = self.IGNvariable_setting.drop(
                        columns=['变量名称_拒绝样本', '是否使用_拒绝样本', '变量角色_拒绝样本'])
                except:
                    pass
                reject = self.par_rejectdatavariable_setting.copy()
                reject['变量名称_拒绝样本'] = reject['变量名称']
                reject['是否使用_拒绝样本'] = reject['是否使用']
                reject['变量角色_拒绝样本'] = reject['变量角色']
                reject['变量类型_拒绝样本'] = reject['变量类型']
                reject = reject.drop(columns=['备注', '变量名称', '是否使用', '变量角色', '变量类型'])
                if self.IGNvariable_setting.empty != True:
                    self.IGNvariable_setting = pd.merge(self.IGNvariable_setting, reject,
                                                        how='outer', left_on='变量名称', right_on='变量名称_拒绝样本')
            elif (datatype == 'oot')&(str(self.comboxlist_oot_data.get())!='NO'):
                path = self.project_info[self.project_info['模块名字'] == self.comboxlist_oot_data.get()]['保存地址'][0]
                fr = open(path, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                self.par_ootdatavariable_setting = node_info[0]['data_variable_setting']
                self.par_oot_dataname = node_info[0]['node_name']
                self.par_oot_dataname_time=node_info[0]['time']
                self.par_oot_data = node_info[1]
                # self.previous_oot_check_change = node_info[0]['check_change']

                self.previous_oot_node_usedlist =node_info[0]['use_node']

                try:
                    self.IGNvariable_setting = self.IGNvariable_setting.drop(columns=
                                                                             ['变量名称_时间外样本', '变量角色_时间外样本', '是否使用_时间外样本'])
                except Exception as e:
                    pass
                oot = self.par_ootdatavariable_setting.copy()
                oot['变量名称_时间外样本'] = oot['变量名称']
                oot['是否使用_时间外样本'] = oot['是否使用']
                oot['变量角色_时间外样本'] = oot['变量角色']
                oot['变量类型_时间外样本'] = oot['变量类型']
                oot = oot.drop(
                    columns=['备注', '变量名称', '是否使用', '变量角色', '变量类型'])
                if self.IGNvariable_setting.empty != True:
                    self.IGNvariable_setting = pd.merge(self.IGNvariable_setting, oot,
                                                        how='outer', left_on='变量名称', right_on='变量名称_时间外样本')
            else:
                if datatype == 'reject':
                    self.par_reject_data = pd.DataFrame()
                    self.par_reject_dataname = None
                    self.par_rejectdatavariable_setting = pd.DataFrame()
                    self.grouped_reject_data = pd.DataFrame()
                    # self.previous_reject_check_change = []
                    self.previous_reject_node_usedlist = []
                    try:
                        self.IGNvariable_setting = self.IGNvariable_setting.drop(
                            columns=['变量名称_拒绝样本', '是否使用_拒绝样本', '变量角色_拒绝样本'])
                    except:
                        pass
                elif datatype == 'oot':
                    self.par_oot_data = pd.DataFrame()
                    self.par_oot_dataname = None
                    self.par_ootdatavariable_setting = pd.DataFrame()
                    self.grouped_oot_data = pd.DataFrame()
                    # self.previous_oot_check_change = []
                    self.previous_oot_node_usedlist = []
                    try:
                        self.IGNvariable_setting = self.IGNvariable_setting.drop(columns=
                                                                             ['变量名称_时间外样本', '变量角色_时间外样本', '是否使用_时间外样本'])
                    except:
                        pass
                else:
                    pass
        except  Exception as e:
            tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (datatype, e))
    def pre_data(self):
        dd = list((self.project_info[(self.project_info['模块类型'] == 'DATA') |
                                     (self.project_info['模块类型'] == 'SPLIT') |
                                     (self.project_info['模块类型'] == 'SAMPLE')])['保存地址'])
        self.train_data_list = []
        self.reject_data_list = ['NO']
        self.oot_data_list = ['NO']

        for add in dd:
            try:
                fr = open(add, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                data_role = node_info[0]['data_role']
                node_name = node_info[0]['node_name']
                if data_role == 'Training model':
                    self.train_data_list.append(node_name)
                else:
                    pass
            except Exception as e:
                tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (add, e))
    def Start_UI(self):
        self.start_window_base = self.master
        width = max(self.master.winfo_screenwidth() * 0.18,400)
        height = self.master.winfo_screenheight() * 0.8
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()
        self.start_window_base.geometry(
            '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
        self.start_window_base.title('规则集参数设置')
        print(height,width)
    def adjustsetting(self):
        # 导入数据
        self.node_intro=LabelFrame(self.start_window_base, text='模块名称:')
        L8 = Label(self.node_intro, width=20, text="模块名称:")
        L8.grid(column=0, row=0, sticky=(W))
        if (self.load=='N')&(self.finsh=='N'):
            node_name=tk.StringVar(value=self.node_name)
            self.entry_node_name= Entry(self.node_intro, textvariable=node_name, bd=1, width=18)
            self.entry_node_name.grid(column=1, row=0, sticky=(W))
        else:
            L88 = Label(self.node_intro, width=20, text="%s" %self.node_name)
            L88.grid(column=1, row=0, sticky=(W))
        self.node_intro.grid(columnspan=3, sticky=(W), padx=10, pady=10)


        self.start_window_data = LabelFrame(self.start_window_base, text='导入数据:')
        L1 = Label(self.start_window_data, width=20, text="训练样本:")
        L1.grid(column=0, row=0, sticky=(W))
        self.comboxlist_train_data = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_train_data["value"] = self.train_data_list
        if self.par_train_data.empty != True:
            for i in range(len(self.train_data_list)):
                if self.train_data_list[i] == self.par_train_dataname:
                    self.comboxlist_train_data.current(i)
        self.comboxlist_train_data.bind("<<ComboboxSelected>>", lambda event: self.load_data(event, datatype='train'))
        self.comboxlist_train_data.grid(column=1, row=0, sticky=(W))

        self.start_window_data.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        # 数值变量设置

        self.start_window_numeric_setting = LabelFrame(self.start_window_base, text='数值型变量分组设置:')
        L6 = Label(self.start_window_numeric_setting, width=20, text="细分箱方法:")
        L6.grid(column=0, row=5, sticky=(W))
        L7 = Label(self.start_window_numeric_setting, width=20, text="分位数:", bd=1)
        L7.grid(column=1, row=5, sticky=(W))
        s_num_bin = tk.StringVar(value=self.par_num_s_bin)
        L8 = Label(self.start_window_numeric_setting, width=20, text="细分箱个数(整数):")
        L8.grid(column=0, row=6, sticky=(W))
        self.entry_s_bin_num = Entry(self.start_window_numeric_setting, textvariable=s_num_bin, bd=1, width=18)
        self.entry_s_bin_num.grid(column=1, row=6, sticky=(W))
        self.entry_s_bin_num.bind('<Return>', lambda event: self.int_num_check(event, 'entry_s_bin_num', 'int'))

        L9 = Label(self.start_window_numeric_setting, width=20, text="是否使用特殊值:")
        L9.grid(column=0, row=7, sticky=(W))
        self.comboxlist_special_code_list = ttk.Combobox(self.start_window_numeric_setting, width=15)
        self.comboxlist_special_code_list["value"] = ['是', '否']
        if self.par_use_specialcode_flag=='是':
            self.comboxlist_special_code_list.current(0)
        else:
            self.comboxlist_special_code_list.current(1)
        self.comboxlist_special_code_list.grid(column=1, row=7, sticky=(W))

        L10 = Label(self.start_window_numeric_setting, width=20, text="导入特殊值数据集:")
        L10.grid(column=0, row=8, sticky=(W))
        L11 = Label(self.start_window_numeric_setting, width=20, text=self.par_sepcialcode_dataname)
        L11.grid(column=1, row=8, sticky=(W))

        self.button_special_code_data = ttk.Button(self.start_window_numeric_setting, text='导入:')
        self.button_special_code_data.grid(column=1, row=9, sticky=(W))
        self.button_special_code_data.bind("<Button-1>", self.specialcode)

        self.button_special_code_data = ttk.Button(self.start_window_numeric_setting, text='样例:')
        self.button_special_code_data.grid(column=0, row=9)
        self.button_special_code_data.bind("<Button-1>", self.specialcode_example)

        self.start_window_numeric_setting.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        # 字符变量设置

        self.start_window_char_setting = LabelFrame(self.start_window_base, text='字符型变量分组设置:')
        L15 = Label(self.start_window_char_setting, width=20, text="限制标准:")
        L15.grid(column=0, row=4, sticky=(W))
        self.comboxlist_char_limitrule_list = ttk.Combobox(self.start_window_char_setting, width=15)
        self.comboxlist_char_limitrule_list["value"] = ['不限制', '样本数', '百分比']
        if self.par_char_restric_flag=='不限制':
            self.comboxlist_char_limitrule_list.current(0)
        elif self.par_char_restric_flag=='样本数':
            self.comboxlist_char_limitrule_list.current(1)
        else:
            self.comboxlist_char_limitrule_list.current(2)
        self.comboxlist_char_limitrule_list.grid(column=1, row=4, sticky=(W))
        char_minnum_bin = tk.StringVar(value=self.par_char_restric_num)
        L13 = Label(self.start_window_char_setting, width=20, text="不分组最小样本数(整数):")
        L13.grid(column=0, row=5, sticky=(W))
        self.entry_min_num_char = Entry(self.start_window_char_setting, textvariable=char_minnum_bin, bd=1, width=18)
        self.entry_min_num_char.grid(column=1, row=5, sticky=(W))
        self.entry_min_num_char.bind('<Return>', lambda event: self.int_num_check(event, 'entry_min_num_char', 'int'))
        char_minpct_bin = tk.StringVar(value=self.par_char_restric_pct)
        L14 = Label(self.start_window_char_setting, width=20, text="不分组最小占比(0-1):")
        L14.grid(column=0, row=6, sticky=(W))
        self.entry_min_pct_char = Entry(self.start_window_char_setting, bd=1, textvariable=char_minpct_bin, width=18)
        self.entry_min_pct_char.grid(column=1, row=6, sticky=(W))
        self.entry_min_pct_char.bind('<Return>', lambda event: self.int_num_check(event, 'entry_min_pct_char', 'pct'))

        self.start_window_char_setting.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        # 粗分箱参数设置
        self.start_window_tree_setting = LabelFrame(self.start_window_base, text='规则集参数设置:')
        L10 = Label(self.start_window_tree_setting, width=20, text="规则最小样本数:")
        L10.grid(column=0, row=10, sticky=(W))
        f_num_bin = tk.StringVar(value=self.par_min_badnum)
        self.entry_min_badnum = Entry(self.start_window_tree_setting, textvariable=f_num_bin, width=18, bd=1)
        self.entry_min_badnum.grid(column=1, row=10, sticky=(W))
        self.entry_min_badnum.bind('<Return>', lambda event: self.int_num_check(event, 'entry_min_badnum', 'int'))

        L11 = Label(self.start_window_tree_setting, width=20, text="规则最小提升率:")
        L11.grid(column=0, row=11, sticky=(W))
        f_num_sample_bin = tk.StringVar(value=self.par_min_lift)
        self.entry_min_lift = Entry(self.start_window_tree_setting, textvariable=f_num_sample_bin, width=18, bd=1)
        self.entry_min_lift.grid(column=1, row=11, sticky=(W))
        self.entry_min_lift.bind('<Return>',
                                       lambda event: self.int_num_check(event, 'entry_min_lift', 'num'))

        L12 = Label(self.start_window_tree_setting, width=20, text="规则最小坏账率:")
        L12.grid(column=0, row=12, sticky=(W))
        f_pct_sample_bin = tk.StringVar(value=self.par_min_badrate)
        self.entry_min_badrate = Entry(self.start_window_tree_setting, textvariable=f_pct_sample_bin, width=18, bd=1)
        self.entry_min_badrate.grid(column=1, row=12, sticky=(W))
        self.entry_min_badrate.bind('<Return>',
                                       lambda event: self.int_num_check(event, 'entry_min_badrate', 'pct'))

        self.start_window_tree_setting.grid(columnspan=3, sticky=(W), padx=10, pady=10)




        self.button_setting_save = ttk.Button(self.start_window_base, text='退出')
        self.button_setting_save.grid(column=0, row=7, sticky=(W), padx=10, pady=10)
        self.button_setting_save.bind("<Button-1>", self.save_project)
        if (self.load == 'Y') | (self.finsh == 'Y'):
            self.check_result = ttk.Button(self.start_window_base, text='查看结果')
            self.check_result.grid(column=1, row=7, sticky=(W), padx=10, pady=10)
            self.check_result.bind("<Button-1>", self.show_result)
        if (self.load == 'N') & (self.finsh == 'N'):
            self.button_setting_run = ttk.Button(self.start_window_base, text='应用')
            self.button_setting_run.grid(column=2, row=7, sticky=(W))
            self.button_setting_run.bind("<Button-1>", self.interactive_grouping)
        else:
            self.button_refresh_run = ttk.Button(self.start_window_base, text='刷新结果')
            self.button_refresh_run.grid(column=2, row=7, sticky=(W))
            self.button_refresh_run.bind("<Button-1>", self.interactive_grouping)
    # 检查所有变量参数是否正确
    def loading_grouping_data(self,event):
        # name = tk.StringVar(value='IGN')
        # path = tk.StringVar(value='D:\\SynologyDrive\\IGN2.IGN')
        self.Group_ui = Toplevel(self.master)
        self.Group_ui.title('分组数据集导入（IGN)')
        width = 500
        height = 250
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()

        self.Group_ui.geometry(
            '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

        def selectExcelfile():
            sfname = filedialog.askopenfilename(title='选择IGN文件', filetypes=[('PLC', '*.PLC')])
            self.E111.delete(0, 'end')
            self.E111.insert(INSERT, sfname)
        L1 = Label(self.Group_ui, text="数据集路径（IGN)")
        L1.grid(column=0, row=0, columnspan=2, sticky=(W))
        self.E111 = Entry(self.Group_ui, width=50,  bd=1)
        self.E111.grid(column=1, row=0, sticky=(W))
        button1 = ttk.Button(self.Group_ui, text='浏览', width=8, command=selectExcelfile)
        button1.grid(column=2, row=0, sticky=(W))

        test_button4 = ttk.Button(self.Group_ui, text='确定')
        test_button4.grid(column=1, row=5, sticky=(W))
        test_button4.bind("<Button-1>", self.grouping_dataim)
    def grouping_dataim(self,event):
        try:
            fr = open(self.E111.get(), 'rb')
            node_info = pickle.load(fr)
            fr.close()
            self.groupingdata = node_info[1][0]
            self.par_import_groupdataname=node_info[0]['node_name']
            self.get_par()
            try:
                self.Group_ui.destroy()
            except:
                pass
            for child in self.master.winfo_children():
                child.destroy()

            self.adjustsetting()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)
    def load_node(self,node_data,ac):
        self.import_node=node_data
        self.load='Y'
        self.node_setting=node_data[0]

        self.par_min_lift=node_data[0]['par_min_lift' ]
        self.par_min_badnum=node_data[0]['par_min_badnum' ]
        self.par_min_badrate=node_data[0]['par_min_badrate' ]
        self.par_num_s_bin=node_data[0]['par_num_s_bin' ]
        self.node_name = node_data[0]['node_name']
        self.par_train_dataname=node_data[0]['previous_node_name'][0]
        self.par_reject_dataname= node_data[0]['previous_node_name'][1]
        self.par_oot_dataname= node_data[0]['previous_node_name'][2]
        self.par_use_freezing_flag=node_data[0]['par_use_freezing_flag']
        self.par_import_groupdataname=node_data[0]['par_import_groupdataname']
        self.par_num_s_bin=node_data[0]['par_num_s_bin']
        self.par_use_specialcode_flag=node_data[0]['par_use_specialcode_flag']
        self.par_specialcode_data=node_data[0]['par_specialcode_data']
        self.par_sepcialcode_dataname=node_data[0]['par_sepcialcode_dataname']
        self.par_char_restric_flag=node_data[0]['par_char_restric_flag']
        self.par_char_restric_num=node_data[0]['par_char_restric_num']
        self.par_char_restric_pct=node_data[0]['par_char_restric_pct']
        self.par_tree_criterion=node_data[0]['par_tree_criterion']
        self.par_num_f_group=node_data[0]['par_num_f_group']
        self.par_min_num_group=node_data[0]['par_min_num_group']
        self.par_min_pct_group=node_data[0]['par_min_pct_group']
        self.par_variable_reject_flag=node_data[0]['par_variable_reject_flag']
        self.par_variable_reject_iv=node_data[0]['par_variable_reject_iv']
        self.IGNvariable_setting=node_data[0]['IGNvariable_setting']
        self.groupingdata=node_data[1][0]
        self.f_group_report=node_data[1][1]
        self.s_group_report=node_data[1][2]
        self.not_use=node_data[1][3]
        self.final_rule =node_data[1][4]
        self.grouped_train_data=node_data[2]
        self.grouped_valid_data=node_data[3]
        self.grouped_reject_data=node_data[4]
        self.grouped_oot_data=node_data[5]
        self.aprior_data =node_data[6]
        self.par_traindatavariable_setting=node_data[0]['data_variable_setting']
        self.par_rejectdatavariable_setting=node_data[0]['reject_data_variable_setting']
        self.par_ootdatavariable_setting=node_data[0]['oot_data_variable_setting']
        previous_node_name=node_data[0]['previous_node_name']
        previous_node_time=node_data[0]['previous_node_time']
        self.par_train_dataname_time=node_data[0]['previous_node_time'][0]
        self.par_reject_dataname_time=node_data[0]['previous_node_time'][1]
        self.par_oot_dataname_time=node_data[0]['previous_node_time'][2]


        if ac == 'setting':
            error_list=[]
            for i in range(0,3):
                if previous_node_name[i]!=None:
                    path_list = self.project_info[self.project_info['创建时间'] == previous_node_time[i]]['保存地址']
                    if len(path_list) == 0:
                        print(previous_node_time)
                        print({'name':previous_node_name[i],'time':previous_node_time[i]})
                        error_list=error_list+[{'name':previous_node_name[i],'time':previous_node_time[i]}]
            def continu(event):
                for child in self.master.winfo_children():
                    child.destroy()
                #以前数据集更新了就重新更新结果
                self.load = 'N'
                self.Start_UI()
                self.adjustsetting()

            def back(event):
                self.master.destroy()

            if len(error_list) > 0:
                self.master.title('提示')
                L00 = Label(self.master, width=80, text="该模块引用的%s 模块 没有在项目中找到，\n可能该模块已经更新，删除，"
                                                        "或未导入\n继续设置可能会导致以前结果丢失" % (error_list))
                L00.grid(column=0, row=0, columnspan=3, sticky=(W))
                button_contin = ttk.Button(self.master, text='继续设置')
                button_contin.grid(column=0, row=1, sticky=(W), padx=10, pady=10)
                button_contin.bind("<Button-1>", continu)
                button_back = ttk.Button(self.master, text='返回')
                button_back.grid(column=2, row=1, sticky=(W), padx=10, pady=10)
                button_back.bind("<Button-1>", back)
            else:
                try:
                    path = self.project_info[self.project_info['创建时间'] == self.par_train_dataname_time]['保存地址'][0]
                    fr = open(path, 'rb')
                    node_info = pickle.load(fr)
                    fr.close()
                    self.par_traindatavariable_setting = node_info[0]['data_variable_setting']
                    self.par_train_dataname = node_info[0]['node_name']
                    self.previous_train_node_usedlist = node_info[0]['use_node']
                    # self.previous_train_check_change = node_info[0]['check_change']
                    self.par_train_data = node_info[1]
                    if node_info[0]['node_type'] == 'SPLIT':
                        self.par_valid_data = node_info[2]
                        print(node_info[0]['node_type'])

                    if self.par_reject_dataname_time != None:
                        path = self.project_info[self.project_info['创建时间'] == self.par_reject_dataname_time]['保存地址'][0]
                        fr = open(path, 'rb')
                        node_info = pickle.load(fr)
                        fr.close()
                        self.par_rejectdatavariable_setting = node_info[0]['data_variable_setting']
                        self.par_reject_dataname = node_info[0]['node_name']
                        self.par_reject_dataname_time=node_info[0]['time']
                        self.previous_reject_node_usedlist = node_info[0]['use_node']
                        # self.previous_reject_check_change = node_info[0]['check_change']
                        self.par_reject_data = node_info[1]

                    if self.par_oot_dataname_time != None:
                        path = self.project_info[self.project_info['创建时间'] == self.par_oot_dataname_time]['保存地址'][0]
                        fr = open(path, 'rb')
                        node_info = pickle.load(fr)
                        fr.close()
                        self.par_ootdatavariable_setting = node_info[0]['data_variable_setting']
                        self.par_oot_dataname = node_info[0]['node_name']
                        self.par_oot_dataname_time=node_info[0]['time']
                        self.previous_oot_node_usedlist = node_info[0]['use_node']
                        # self.previous_oot_check_change = node_info[0]['check_change']
                        self.par_oot_data = node_info[1]
                    self.Start_UI()
                    self.adjustsetting()
                except Exception as e:
                    self.master.title('提示')
                    L00 = Label(self.master, width=80, text="导入%s （创建于 %s)模块 发生错误，\n可能该模块已经被破坏或删除，"
                                                            "\n%s" % (previous_node_name, previous_node_time, e))
                    L00.grid(column=0, row=0, columnspan=3, sticky=(W))
                    button_contin = ttk.Button(self.master, text='继续设置')
                    button_contin.grid(column=0, row=1, sticky=(W), padx=10, pady=10)
                    button_contin.bind("<Button-1>", continu)
                    button_back = ttk.Button(self.master, text='返回')
                    button_back.grid(column=2, row=1, sticky=(W), padx=10, pady=10)
                    button_back.bind("<Button-1>", back)

        self.target_train = \
            list(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
        if len(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == 'TimeID']) == 1:
            self.flag_timeid_train = True
            self.timeid_train = \
                self.par_traindatavariable_setting.loc[self.par_traindatavariable_setting['变量角色'] == 'TimeID'][
                    '变量名称'].values[0]
        if self.grouped_reject_data.empty != True:
            # 拒绝集变量
            try:
                self.target_reject = \
                    list(self.par_rejectdatavariable_setting[self.par_rejectdatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
            except:
                pass
            self.varchar_reject = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                                (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                                (self.IGNvariable_setting['变量类型'] == '字符型')&
                                                                (self.IGNvariable_setting['变量类型_拒绝样本'] == '字符型')]['变量名称_拒绝样本'])
            self.varnum_reject = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                               (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                               (self.IGNvariable_setting['变量类型'] == '数值型')&
                                                               (self.IGNvariable_setting['变量类型_拒绝样本'] == '数值型')]['变量名称_拒绝样本'])

            if len(self.par_rejectdatavariable_setting[self.par_rejectdatavariable_setting['变量角色'] == 'TimeID']) == 1:
                self.flag_timeid_reject = True
                self.timeid_reject = \
                    self.par_rejectdatavariable_setting.loc[self.par_rejectdatavariable_setting['变量角色'] == 'TimeID'][
                        '变量名称'].values[0]

        if self.grouped_oot_data.empty != True:
            # oot变量
            try:
                self.target_oot = \
                    list(self.par_ootdatavariable_setting[self.par_ootdatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
            except:
                pass
            self.varchar_oot = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                             (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                             (self.IGNvariable_setting['变量类型'] == '字符型')&
                                                             (self.IGNvariable_setting['变量类型_时间外样本'] == '字符型')]['变量名称_时间外样本'])
            self.varnum_oot = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                            (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                            (self.IGNvariable_setting['变量类型'] == '数值型')&
                                                            (self.IGNvariable_setting['变量类型_时间外样本'] == '数值型')]['变量名称_时间外样本'])

            if len(self.par_ootdatavariable_setting[self.par_ootdatavariable_setting['变量角色'] == 'TimeID']) == 1:
                self.flag_timeid_oot = True
                self.timeid_oot = \
                    self.par_ootdatavariable_setting.loc[self.par_ootdatavariable_setting['变量角色'] == 'TimeID'][
                        '变量名称'].values[0]

        # 训练集变量
        self.varchar = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                     (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                     (self.IGNvariable_setting['变量类型'] == '字符型')
                                                     ]['变量名称'])
        self.varnum = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                    (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                    (self.IGNvariable_setting['变量类型'] == '数值型')]['变量名称'])
    def result(self):
        self.vaild_flag = self.grouped_valid_data.empty != True
        self.reject_flag = self.grouped_reject_data.empty == False
        self.oot_flag = self.grouped_oot_data.empty == False
        UserInterfacea(mainfram=self.master, f_data=self.f_group_report, s_data=self.s_group_report,
                                    group_info=self.groupingdata,
                                    varchar=self.varchar, varnum=self.varnum,
                                    varchar_reject=self.varchar_reject, varnum_reject=self.varnum_reject,
                                    varchar_oot=self.varchar_oot, varnum_oot=self.varnum_oot,
                                    target=self.target_train, target_reject=self.target_reject,
                                    target_oot=self.target_oot,
                                    grouped_train_data=self.grouped_train_data,
                                    grouped_valid_data=self.grouped_valid_data,
                                    grouped_reject_data=self.grouped_reject_data,
                                    grouped_oot_data=self.grouped_oot_data,

                                    train_data=self.grouped_train_data, valid_data=self.grouped_valid_data,
                                    reject_data=self.grouped_reject_data, oot_data=self.grouped_oot_data,

                                    vaild_flag=self.vaild_flag, reject_flag=self.reject_flag, oot_flag=self.oot_flag,
                                    flag_timeid_oot=self.flag_timeid_oot,
                                    timeid_oot=self.timeid_oot,
                                    flag_timeid_reject=self.flag_timeid_reject,
                                    timeid_reject=self.timeid_reject,
                                    flag_timeid_train=self.flag_timeid_train,
                                    timeid_train=self.timeid_train,
                                    not_use=self.not_use,
                                    project_path=self.project_path,
                                    node_name=self.node_name,result_show=True)
    def check_all_setting(self, event):
        self.get_par()
        mm=0




        if (self.node_name in self.exist_data) & (self.load == 'N'):
            mm = mm + 1
            tk.messagebox.showwarning('错误', "该名称已经被占用，请更改")
        if self.par_train_data.empty == True:
            mm= mm + 1
            tk.messagebox.showwarning('错误', "错误：训练样本为空")
        else:
            total = ['entry_min_badrate', 'entry_min_badnum', 'entry_min_lift',
                     'entry_min_num_char', 'entry_min_pct_char', 'entry_s_bin_num']
            if self.par_char_restric_flag == '不限制':
                total.remove('entry_min_pct_char')
                total.remove('entry_min_num_char')
            for p in total:
                if p in ['entry_min_badnum', 'entry_min_num_char', 'entry_s_bin_num']:
                    flag = 'int'
                    entry_p = p
                    pp=self.int_num_check(event, entry_p, flag)
                    mm=mm+pp
                elif p in [ 'entry_min_pct_char', 'entry_min_badrate']:
                    flag = 'pct'
                    entry_p = p
                    pp=self.int_num_check(event, entry_p, flag)
                    mm = mm + pp
                else:
                    flag = 'g'
                    entry_p = p
                    pp=self.int_num_check(event, entry_p, flag)
                    mm = mm + pp
        return mm
    def get_par(self):

        self.par_min_lift = float(self.entry_min_lift.get())
        self.par_min_badnum = int(self.entry_min_badnum.get())
        self.par_min_badrate = float(self.entry_min_badrate.get())

        self.par_char_restric_flag = self.comboxlist_char_limitrule_list.get()
        self.par_char_restric_num = int(self.entry_min_num_char.get())
        self.par_char_restric_pct = float(self.entry_min_pct_char.get())

        self.par_num_s_bin = int(self.entry_s_bin_num.get())
        self.par_use_specialcode_flag = self.comboxlist_special_code_list.get()
        self.par_specialcode_data = self.par_specialcode_data
        self.par_sepcialcode_dataname = self.par_sepcialcode_dataname


        self.par_import_groupdataname = self.par_import_groupdataname
        if (self.finsh=='N')&(self.load=='N'):
            self.node_name=self.entry_node_name.get()
    def int_num_check(self, event, entry_p, flag):
        a=0

        if entry_p == 'entry_min_badrate':
            inputnum = self.entry_min_badrate.get()
            tip = '规则最小坏账率'
        elif entry_p == 'entry_min_badnum':
            inputnum = self.entry_min_badnum.get()
            tip = '规则最小坏样本数'
        elif entry_p == 'entry_min_lift':
            inputnum = self.entry_min_lift.get()
            tip = '规则最小提升率'
        elif entry_p == 'entry_min_num_char':
            inputnum = self.entry_min_num_char.get()
            tip = '字符型变量不分组最小样本数'
        elif entry_p == 'entry_min_pct_char':
            inputnum = self.entry_min_pct_char.get()
            tip = '字符型变量不分组最小占比'
        elif entry_p == 'entry_s_bin_num':
            inputnum = self.entry_s_bin_num.get()
            tip = '细分箱个数'
        else:
            tip = entry_p

        try:
            if float(inputnum) < 0:
                a = a + 1
                tk.messagebox.showwarning('错误', '%s:输入值不能小于0' % tip)
            else:
                if flag == 'int':
                    try:
                        int(inputnum)
                    except Exception as e:
                        a = a + 1
                        tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
                elif flag == 'pct':
                    try:
                        num = float(inputnum)
                        if num > 1:
                            a = a + 1
                            tk.messagebox.showwarning('错误', '%s:输入值不能大于1' % tip)
                        else:
                            pass
                    except Exception as e:
                        a = a + 1
                        tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
                else:
                    try:
                        num = float(inputnum)
                    except Exception as e:
                        a = a + 1
                        tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
        except Exception as e:
            a = a + 1
            tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
        return a
    # 导入特殊值数据集
    def specialcode(self, event):
        # name = tk.StringVar(value='data1')
        # path = tk.StringVar(value='D:\\SynologyDrive\\specialcode.csv')
        self.special_code_ui = Toplevel(self.master)
        self.special_code_ui.title('特殊值数据集导入')
        width = 500
        height = 250
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()

        self.special_code_ui.geometry(
            '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

        def selectExcelfile():
            sfname = filedialog.askopenfilename(title='选择CSV文件', filetypes=[('CSV', '*.csv')])
            self.E1.delete(0, 'end')
            self.E1.insert(INSERT, sfname)
            entry_name = os.path.basename(sfname).replace('.csv','')
            self.E2.delete(0, 'end')
            self.E2.insert(INSERT, entry_name)
        L1 = Label(self.special_code_ui, text="数据集路径（CSV)")
        L1.grid(column=0, row=0, columnspan=2, sticky=(W))
        self.E1 = Entry(self.special_code_ui, width=50,  bd=1)
        self.E1.grid(column=1, row=0, sticky=(W))
        button1 = ttk.Button(self.special_code_ui, text='浏览', width=8, command=selectExcelfile)
        button1.grid(column=2, row=0, sticky=(W))

        L1 = Label(self.special_code_ui, text="数据集名称")
        L1.grid(column=0, row=1, columnspan=2, sticky=(W))
        self.E2 = Entry(self.special_code_ui, width=23, bd=1)
        self.E2.grid(column=1, row=1, sticky=(W))

        L3 = Label(self.special_code_ui, text="数据集编码")
        L3.grid(column=0, row=2, sticky=(W))
        self.E3 = ttk.Combobox(self.special_code_ui)
        self.E3["value"] = ['utf-8', 'gbk']
        self.E3.current(0)
        self.E3.grid(column=1, row=2, sticky=(W))

        test_button4 = ttk.Button(self.special_code_ui, text='确定')
        test_button4.grid(column=1, row=5, sticky=(W))
        test_button4.bind("<Button-1>", self.readdata)
    def readdata(self, event):
        path = self.E1.get()
        name = self.E2.get()
        coding = self.E3.get()
        try:
            data = pd.read_csv(r'%s' % path, encoding='%s' % coding, low_memory=False)
            if data.empty == True:
                tk.messagebox.showwarning('错误', "错误：数据集为空")
            elif( ('value' in data.columns) == False)|(('variable' in data.columns) == False):
                tk.messagebox.showwarning('错误', "错误：数据集格式错误请看样例")
            else:
                self.tt1 = Toplevel(self.special_code_ui)
                self.tt1.title(name)
                screenwidth = self.tt1.winfo_screenwidth()
                screenheight = self.tt1.winfo_screenheight()
                width = self.tt1.winfo_screenwidth() * 0.2
                height = self.tt1.winfo_screenheight() * 0.4
                self.tt1.geometry(
                    '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
                df = data[0:100]
                test_button4 = ttk.Button(self.tt1, text='确定', command=lambda: self.save_spcecialcode_data(data, name))
                test_button4.grid(column=0, row=0, sticky=(W))

                f = Frame(self.tt1)
                f.grid(column=0, row=1, sticky=(E, W))
                screen_width = self.tt1.winfo_screenwidth() * 0.2
                screen_height = self.tt1.winfo_screenheight() * 0.4
                table = ptm = Table(f, dataframe=df, height=screen_height, width=screen_width)
                ptm.show()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)
    def specialcode_example(self, event):
        dd = pd.DataFrame([{'variable': 'VAR1', 'value': '999'}, {'variable': 'VAR1', 'value': '-1'},
                           {'variable': 'VAR2', 'value': '-1'}])
        tt = Toplevel()
        tt.title('样例')
        screenwidth = tt.winfo_screenwidth()
        screenheight = tt.winfo_screenheight()
        width = tt.winfo_screenwidth() * 0.2
        height = tt.winfo_screenheight() * 0.4
        tt.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
        f = Frame(tt)

        f.grid(column=0, row=1, sticky=(E, W))
        screen_width = f.winfo_screenwidth() * 0.2
        screen_height = f.winfo_screenheight() * 0.4
        table = ptm = Table(f, dataframe=dd, height=screen_height, width=screen_width)
        ptm.show()
    def save_spcecialcode_data(self, data, name):
        self.par_specialcode_data = data
        self.par_sepcialcode_dataname = name
        self.get_par()
        self.tt1.destroy()
        self.special_code_ui.destroy()
        for child in self.master.winfo_children():
            child.destroy()

        self.adjustsetting()
    # 变量设置
    def modify_variable_role(self, event):
        try:
            self.comboxlist_modify_f_group.destroy()

        except:
            pass
        self.rowclicked = self.ptm.get_row_clicked(event)
        self.colclicked = self.ptm.get_col_clicked(event)

        if list(self.IGNvariable_setting.columns)[self.colclicked] == '是否使用':
            try:
                self.comboxlist_modify_f_group = ttk.Combobox(self.data_variable_set_ui)

                self.comboxlist_modify_f_group["value"] = ['使用', '不使用']

                self.comboxlist_modify_f_group.place(x=event.x_root - self.data_variable_set_ui.winfo_rootx(),
                                                     y=event.y_root - self.data_variable_set_ui.winfo_rooty())
                self.comboxlist_modify_f_group.bind("<<ComboboxSelected>>", self.variable_role_update)

            except:
                pass

        else:
            pass
    def show_variabledetail(self, event):
        self.data_variable_set_ui = Toplevel(self.master)
        self.data_variable_set_ui.title('变量设置')
        self.refresh_datavariable_df()
    def refresh_datavariable_df(self):
        f = Frame(self.data_variable_set_ui)
        f.grid(column=0, row=1,
               columnspan=6, sticky=(E, W))
        screen_width = f.winfo_screenwidth() * 0.7
        screen_height = f.winfo_screenheight() * 0.9
        self.table = self.ptm = Table(f, dataframe=self.IGNvariable_setting, colspan=7,
                                      height=screen_height, width=screen_width)
        self.ptm.show()
        self.table.bind("<Button-3>", self.modify_variable_role)
        self.table.bind("<Button-2>", self.modify_variable_role)

        self.table.bind("<Double-Button-3>", self.modify_variable_role)
        self.table.bind("<Double-Button-1>", self.modify_variable_role)
        self.table.bind("<Double-Button-2>", self.modify_variable_role)
        self.table.bind("<Triple-Button-3>", self.modify_variable_role)
        self.table.bind("<Triple-Button-1>", self.modify_variable_role)
        self.table.bind("<Triple-Button-2>", self.modify_variable_role)
    def variable_role_update(self, event):
        value = self.comboxlist_modify_f_group.get()
        self.IGNvariable_setting.iloc[self.rowclicked, self.colclicked] = value
        self.comboxlist_modify_f_group.destroy()
        self.refresh_datavariable_df()
    # 分组
    def interactive_grouping(self, event):
        # 检查各个数据集变量情况
        try:
            if self.temp.state() == 'normal':
                tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
        except:
            error_num=self.check_all_setting(event)
            if error_num==0:
                node_save_path = self.project_path + '/' + '%s.PLC' % self.node_name
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                self.node_setting = {'node_type': 'PLC',
                                     'node_name': self.node_name,

                                    'par_min_lift': self.par_min_lift ,
                                    'par_min_badnum':self.par_min_badnum ,
                                    'par_min_badrate': self.par_min_badrate ,
                                    'par_num_s_bin': self.par_num_s_bin ,

                                     'node_save_path': node_save_path,
                                     'par_use_freezing_flag': self.par_use_freezing_flag,
                                     'par_import_groupdataname': self.par_import_groupdataname,
                                     'par_num_s_bin': self.par_num_s_bin,
                                     'par_use_specialcode_flag': self.par_use_specialcode_flag,
                                     'par_specialcode_data': self.par_specialcode_data,
                                     'par_sepcialcode_dataname': self.par_sepcialcode_dataname,
                                     'par_char_restric_flag': self.par_char_restric_flag,
                                     'par_char_restric_num': self.par_char_restric_num,
                                     'par_char_restric_pct': self.par_char_restric_pct,
                                     'par_tree_criterion': self.par_tree_criterion,
                                     'par_num_f_group': self.par_num_f_group,
                                     'par_min_num_group': self.par_min_num_group,
                                     'par_min_pct_group': self.par_min_pct_group,
                                     'par_variable_reject_flag': self.par_variable_reject_flag,
                                     'par_variable_reject_iv': self.par_variable_reject_iv,
                                     'IGNvariable_setting':self.IGNvariable_setting ,
                                     'time': nowTime,
                                     'previous_node_name': [self.par_train_dataname, self.par_reject_dataname,
                                                            self.par_oot_dataname],
                                     'previous_node_time': [self.par_train_dataname_time, self.par_reject_dataname_time,
                                                            self.par_oot_dataname_time],
                                     # 'check_change': [{'node_name': self.node_name, 'node_time': nowTime}]
                                     #                 + self.previous_train_check_change + self.previous_reject_check_change +
                                     #                 self.previous_oot_check_change,
                                     'data_variable_setting': self.par_traindatavariable_setting,
                                     'reject_data_variable_setting': self.par_rejectdatavariable_setting,
                                     'oot_data_variable_setting': self.par_ootdatavariable_setting,
                                     'use_node': [self.node_name] + self.previous_train_node_usedlist
                                                 + self.previous_reject_node_usedlist + self.previous_oot_node_usedlist
                                     }
                self.target_train = \
                list(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
                if len(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == 'TimeID']) == 1:
                    self.flag_timeid_train = True
                    self.timeid_train = \
                    self.par_traindatavariable_setting.loc[self.par_traindatavariable_setting['变量角色'] == 'TimeID'][
                        '变量名称'].values[0]

                if self.par_reject_data.empty != True:
                    # 拒绝集变量
                    try:
                        self.target_reject = \
                        list(self.par_rejectdatavariable_setting[self.par_rejectdatavariable_setting['变量角色'] == '目标']['变量名称'])[
                            0]
                    except:
                        pass
                    self.varchar_reject = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                                        (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                                        (self.IGNvariable_setting['变量类型'] == '字符型')&
                                                                        (self.IGNvariable_setting['变量类型_拒绝样本'] == '字符型')][
                                                   '变量名称_拒绝样本'])
                    self.varnum_reject = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                                       (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                                       (self.IGNvariable_setting['变量类型'] == '数值型')&
                                                                       (self.IGNvariable_setting['变量类型_拒绝样本'] == '数值型')][
                                                  '变量名称_拒绝样本'])

                    if len(self.par_rejectdatavariable_setting[self.par_rejectdatavariable_setting['变量角色'] == 'TimeID']) == 1:
                        self.flag_timeid_reject = True
                        self.timeid_reject = \
                        self.par_rejectdatavariable_setting.loc[self.par_rejectdatavariable_setting['变量角色'] == 'TimeID'][
                            '变量名称'].values[0]
                else:
                    self.target_reject=None
                    self.flag_timeid_reject=False
                    self.timeid_reject=None
                if self.par_oot_data.empty != True:
                    # oot变量
                    try:
                        self.target_oot = \
                        list(self.par_ootdatavariable_setting[self.par_ootdatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
                    except:
                        pass
                    self.varchar_oot = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                                     (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                                     (self.IGNvariable_setting['变量类型'] == '字符型')&
                                                                     (self.IGNvariable_setting['变量类型_时间外样本'] == '字符型')]['变量名称_时间外样本'])
                    self.varnum_oot = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                                    (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                                    (self.IGNvariable_setting['变量类型'] == '数值型')&
                                                                    (self.IGNvariable_setting['变量类型_时间外样本'] == '数值型')]['变量名称_时间外样本'])

                    if len(self.par_ootdatavariable_setting[self.par_ootdatavariable_setting['变量角色'] == 'TimeID']) == 1:
                        self.flag_timeid_oot = True
                        self.timeid_oot = \
                        self.par_ootdatavariable_setting.loc[self.par_ootdatavariable_setting['变量角色'] == 'TimeID'][
                            '变量名称'].values[0]
                else:
                    self.target_oot=None
                    self.flag_timeid_oot=False
                    self.timeid_oot=None

                # 训练集变量
                self.varchar = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                             (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                             (self.IGNvariable_setting['变量类型'] == '字符型')]['变量名称'])
                self.varnum = list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '使用') &
                                                            (self.IGNvariable_setting['变量角色'] == '自变量') &
                                                            (self.IGNvariable_setting['变量类型'] == '数值型')]['变量名称'])
                #重新fit group
                error2 = Toplevel(self.master)
                screenwidth = self.master.winfo_screenwidth()
                screenheight = self.master.winfo_screenheight()
                error2.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
                L2 = Label(error2, text="运行中。。。\n 请稍等一会会  \n不要重复提交")
                L2.grid()
                self.master.update()
                if (self.par_use_freezing_flag == '否') | (self.groupingdata.empty == True):
                    if (self.par_use_specialcode_flag == '是') & (self.par_specialcode_data.empty == False):
                        if len(self.par_specialcode_data[self.par_specialcode_data['variable']=="_ALL_"])>0:
                            sp_values=list(self.par_specialcode_data[self.par_specialcode_data['variable']=="_ALL_"]['value'])
                            others_df=self.par_specialcode_data[self.par_specialcode_data['variable']!="_ALL_"]
                            for value in sp_values:
                                temp=pd.DataFrame()
                                temp['variable']=self.varnum
                                temp['value']=value
                                others_df=others_df.append(temp)
                            special_code=others_df
                        else:
                            special_code = self.par_specialcode_data
                    else:
                        special_code = pd.DataFrame()
                    if self.par_char_restric_flag == '不限制':
                        min_num = 0
                        min_pct = 0
                    elif self.par_char_restric_flag == '样本数':
                        min_pct = 0
                        min_num = self.par_char_restric_num
                    else:
                        min_num = len(self.par_train_data)
                        min_pct = self.par_char_restric_pct

                    s_group_map, s_group_data = binning.fit_bin_aprior( data = self.par_train_data,varnum = self.varnum,target = self.target_train,
                                                       s_bin_num = self.par_num_s_bin,special_code=special_code)
                    s_group_map['iv'] = 0
                    s_group_map['woe'] = 0
                    s_group_map['f_Bad_rate'] = 0
                    s_group_map['f_N_obs'] = 0
                    s_group_map['f_N_bad'] = 0
                    s_group_map['f_group'] = 0
                    s_group_map['miss_count'] = 0
                    s_group_report, f_group_report = binning.report(group_info=s_group_map, varchar=self.varchar,varnum=self.varnum)
                    merge_data = s_group_report[['variable_name', 's_group', 'value', 'label']]
                    merge_data.loc[merge_data['value'].isnull() == False, 'label'] = merge_data.loc[merge_data[
                                                                                                        'value'].isnull() == False, 'variable_name'] + '_' + \
                                                                                     merge_data.loc[merge_data[
                                                                                                        'value'].isnull() == False, 'value']
                    merge_data['variable_name'] = 's_group_' + merge_data['variable_name']
                    aprior_data = s_group_data[self.varchar + list(merge_data['variable_name'].unique()) + [self.target_train]]

                    for var in merge_data['variable_name'].unique():
                        aprior_data = pd.merge(merge_data[merge_data['variable_name'] == var][['s_group', 'label']],
                                               aprior_data, left_on='s_group', right_on=var, how='right')
                        aprior_data['ap_%s' % var] = aprior_data['label']
                        aprior_data = aprior_data.drop(columns=[var, 'label', 's_group'])
                    for var in self.varchar:
                        aprior_data['ap_%s' % var] = var + aprior_data[var]
                        aprior_data.loc[aprior_data['ap_%s' % var].isnull(), 'ap_%s' % var] = 'miss'
                        aprior_data = aprior_data.drop(columns=[var])
                    aprior_data[self.target_train] = s_group_data[self.target_train]
                    from itertools import combinations
                    print(list(combinations(aprior_data.columns, 2)))

                    pair_list = list(aprior_data.columns)
                    pair_list.remove(self.target_train)

                    pair_list = list(combinations(pair_list, 1))
                    # +list(combinations(pair_list, 2))+list(combinations(pair_list, 3))
                    print(pair_list)

                    re_df = pd.DataFrame()
                    for i in range(len(pair_list)):
                        pair = pair_list[i]
                        tt = aprior_data[list(pair) + [self.target_train]]
                        re = tt.groupby(list(pair))[self.target_train].agg(['sum', 'mean', 'count']).reset_index()
                        re.head()
                        re['rule'] = re.apply(lambda x: [x[t] for t in list(pair)], axis=1)
                        re['variable'] = re.apply(lambda x: list(pair), axis=1)
                        re = re[['sum', 'mean', 'count', 'rule', 'variable']]
                        re_df = re_df.append(re)

                    re_df['lift'] = re_df['mean'] / np.mean(aprior_data[self.target_train])
                    re_df = re_df.sort_values(by='lift', ascending=False)
                    final_rule = re_df[
                        (re_df['count'] > self.par_min_badnum) & (re_df['mean'] > self.par_min_badrate) & (re_df['lift'] > self.par_min_lift)]
                    final_rule = final_rule.reset_index(drop=True)
                    final_rule = final_rule.reset_index()
                    if len(final_rule)==0:
                        # 重新fit group
                        error2 = Toplevel(self.master)
                        screenwidth = self.master.winfo_screenwidth()
                        screenheight = self.master.winfo_screenheight()
                        error2.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
                        L2 = Label(error2, text="没有符合条件的数据请重新调整参数")
                        L2.grid()
                        self.master.update()
                    for i in range(len(final_rule)):
                        rule_t = final_rule.iloc[i]
                        merge_data = pd.DataFrame(data=np.array([rule_t['rule'] + [1]]), columns=np.array(
                            rule_t['variable'] + ['flag_rule_' + str(rule_t['index'])]))
                        aprior_data = pd.merge(aprior_data, merge_data, how='left')
                        aprior_data.loc[
                            aprior_data['flag_rule_' + str(rule_t['index'])].isnull() == True, 'flag_rule_' + str(
                                rule_t['index'])] = 0

                    final_rule['rule_name'] = final_rule['index'].apply(lambda x: 'flag_rule_' + str(x))
                    self.final_rule=final_rule
                    self.aprior_data=aprior_data



                else:
                    #use existing grouping info
                    list_num_exist=list(set(self.groupingdata[self.groupingdata['variable_type']=='num']['variable_name']))
                    list_char_exist = list(
                        set(self.groupingdata[self.groupingdata['variable_type'] == 'char']['variable_name']))
                    list_num_comm_existing=list(set(self.varnum).intersection(set(list_num_exist)))
                    list_char_comm_existing =list(set(self.varchar).intersection(set(list_char_exist)))
                    list_num_left=list(set(self.varnum)-set(list_num_comm_existing))
                    list_char_left=list(set(self.varchar)-set(list_char_comm_existing))
                    if (list_num_left!=[]) or (list_char_left!=[]):
                        if (self.par_use_specialcode_flag == '是') & (self.par_specialcode_data.empty == False):
                            special_code = self.par_specialcode_data
                        else:
                            special_code = pd.DataFrame()
                        if self.par_char_restric_flag == '不限制':
                            min_num = 0
                            min_pct = 0
                        elif self.par_char_restric_flag == '样本数':
                            min_pct = 0
                            min_num = self.par_char_restric_num
                        else:
                            min_num = len(self.par_train_data)
                            min_pct = self.par_char_restric_pct
                        #训练现有分组信息中没有得变量
                        self.groupingdata_new, pardata = binning.fit_bin(data=self.par_train_data,
                                                                                     varnum=list_num_left,
                                                                                     varchar=list_char_left,
                                                                                     target=self.target_train,
                                                                                     s_bin_num=self.par_num_s_bin,
                                                                                     min_num=min_num,
                                                                                     min_pct=min_pct,
                                                                                     special_code=special_code,
                                                                                     criterion=self.par_tree_criterion,
                                                                                     max_depth=20,
                                                                                     min_samples_leaf=min(
                                                                                         self.par_min_num_group,
                                                                                         int(len(
                                                                                             self.par_train_data) * self.par_min_pct_group)),
                                                                                     max_leaf_nodes=self.par_num_f_group,
                                                                                     )
                        self.groupingdata_new['variable_type']=self.groupingdata_new.apply(lambda x: 'num' if x['variable_name'] in list_num_left else 'char', axis=1)
                        #保留以前得变量
                        self.groupingdata=self.groupingdata.append(self.groupingdata_new)
                    self.grouped_train_data = binning.fit_bin_existing(data=self.par_train_data,
                                                                       varnum=self.varnum,
                                                                       varchar=self.varchar,
                                                                       target=self.target_train,
                                                                       group_info=self.groupingdata,
                                                                       data_only=True)
                #self.bing_restdata()
                try:
                    error2.destroy()
                except:
                    pass
                if self.par_variable_reject_flag!='no':
                    self.not_use=list(self.groupingdata[self.groupingdata['iv']<self.par_variable_reject_iv]['variable_name'].unique())
                else:
                    self.not_use=[]
                #完全划分样本的变量
                #perfect_variavle_check=self.groupingdata.copy()
                #perfect_variavle_check['flag'] = perfect_variavle_check.apply(lambda x: 1 if (x['s_N_bad'] != 0) and (x['s_N_obs'] != x['s_N_bad']) else 0,axis=1)
                #self.not_use=self.not_use+list(set(perfect_variavle_check['variable_name']) - set(perfect_variavle_check[perfect_variavle_check['flag'] == 1]['variable_name']))

                #self.not_use=self.not_use+list(self.IGNvariable_setting[(self.IGNvariable_setting['是否使用'] == '不使用') &
                #                                            (self.IGNvariable_setting['变量角色'] == '自变量') ]['变量名称'])
                self.finsh='Y'
                for child in self.master.winfo_children():
                    child.destroy()
                self.adjustsetting()

    def show_result(self,event):

        tt = Toplevel()
        tt.title('规则集')
        screenwidth = tt.winfo_screenwidth()
        screenheight = tt.winfo_screenheight()
        width = tt.winfo_screenwidth() * 0.6
        height = tt.winfo_screenheight() * 0.8
        tt.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
        f = Frame(tt)

        f.grid(column=0, row=1, sticky=(E, W))
        screen_width = f.winfo_screenwidth() * 0.6
        screen_height = f.winfo_screenheight() * 0.8
        show_df=self.final_rule[['rule','mean','lift','sum','count']]
        show_df.columns=['规则','坏账率','提升率','坏客户数','客户数']
        table = ptm = Table(f, dataframe=show_df, height=screen_height, width=screen_width)
        ptm.show()

        plot_df = self.aprior_data.copy()
        plot_df['rm__order__'] = plot_df.index
        tt = self.aprior_data.groupby(['flag_rule_' + str(x) for x in range(len(self.final_rule))])[self.target_train].agg(
            {'count', 'mean'}).reset_index()
        tt['target'] = tt.apply(
            lambda x: "-".join(set([int(x[t]) * t for t in ['flag_rule_' + str(m) for m in range(len(self.final_rule))]])),
            axis=1)
        tt['label'] = tt.apply(lambda x: "&".join(
            set([int(x['flag_rule_%s' % m]) * str(self.final_rule['rule'][m]) for m in range(len(self.final_rule))])), axis=1)

        tt['start'] = tt.apply(
            lambda x: set([int(x[t]) * t for t in ['flag_rule_' + str(m) for m in range(len(self.final_rule))]]), axis=1)
        links_list = []
        nodes_list = []
        for i in range(len(tt)):
            for s in tt.iloc[i]['start']:
                if s != '':
                    links_list.append({'source': s, 'target': tt.iloc[i]['target'], 'value': tt.iloc[i]['count'],
                                       'badrate': tt.iloc[i]['mean']})
                    nodes_list.append(s)
                    nodes_list.append(tt.iloc[i]['target'])

                    # nodes_list.append({'name':s})
                    # nodes_list.append({'name':tt.iloc[i]['target']})
        nodes_list = list(set(nodes_list))
        nodes_list = [{'name': x} for x in nodes_list]


        pio.renderers.default = 'browser'

        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % rgb

        dd = pd.DataFrame(links_list)
        labels = list(set(list(dd['source']) + list(dd['target'])))
        dd['num'] = dd['value'] * dd['badrate']
        fi = dd.groupby('source')['num', 'value'].sum().reset_index()
        fi['rate'] = fi['num'] / fi['value']

        badrate_df = pd.concat([fi[['source', 'rate']].rename({'source': 'label', 'rate': 'rate'}, axis=1),
                                dd[['target', 'badrate']].rename({'target': 'label', 'badrate': 'rate'}, axis=1)])

        comment_df = pd.concat(
            [self.final_rule[['rule_name', 'rule']].rename({'rule_name': 'label', 'rule': 'comment'}, axis=1),
             tt[['target', 'label']].rename({'target': 'label', 'label': 'comment'}, axis=1)])

        bad_rate_list = [list(badrate_df.loc[badrate_df['label'] == m, 'rate'])[0] for m in labels]

        commend = [list(comment_df.loc[comment_df['label'] == m, 'comment'])[0] for m in labels]

        col_code = [rgb_to_hex((255 - int(x * 255), 255 - int(x * 255), 255 - int(x * 255))) for x in bad_rate_list]
        #col_code = [rgb(256 - int(x * 255), 256 - int(x * 255), 256 - int(x * 255)) for x in bad_rate_list]
        mergedd = pd.DataFrame(labels)
        mergedd['order'] = mergedd.index
        mergedd = mergedd.rename({0: 'var'}, axis=1)

        sankey_df = pd.merge(dd, mergedd, left_on='source', right_on='var', how='left')
        sankey_df = sankey_df.rename({'order': 'source_order'}, axis=1)

        sankey_df = pd.merge(sankey_df, mergedd, left_on='target', right_on='var', how='left')
        sankey_df = sankey_df.rename({'order': 'target_order'}, axis=1)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[commend[m] for m in range(len(commend))],
                color=col_code,
                customdata=[[commend[m], round(bad_rate_list[m], 3)] for m in range(len(commend))],
                hovertemplate='Node: %{label} <br /> Rules:%{customdata[0]} <br /> badrate: %{customdata[1]}<extra></extra>',
            ),
            link=dict(
                source=list(sankey_df['source_order']),  # indices correspond to labels, eg A1, A2, A1, B1, ...
                target=list(sankey_df['target_order']),
                value=list(sankey_df['value']),
                color=[rgb_to_hex((255 - int(x * 255), 255 - int(x * 255), 255 - int(x * 255))) for x in
                       list(sankey_df['badrate'])]
            ))])
        fig.update_layout(title_text="规则图", font_size=10)
        fig.show()


    def save_project(self,event):
        try:
            node_save_path = self.project_path + '/' + '%s.PLC' % self.node_name
            self.f_group_report=pd.DataFrame()
            self.s_group_report=pd.DataFrame()
            self.not_use=[]
            group_result=[self.groupingdata,self.f_group_report,self.s_group_report,self.not_use,self.final_rule]
            data_save = (self.node_setting, group_result,self.grouped_train_data,self.grouped_valid_data,
                         self.grouped_reject_data,self.grouped_oot_data,self.aprior_data)
            error2 = Toplevel(self.master)
            screenwidth = self.master.winfo_screenwidth()
            screenheight = self.master.winfo_screenheight()
            error2.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
            L2 = Label(error2, text="保存中")
            L2.grid()
            self.master.update()
            filename = node_save_path
            fw = open(filename, 'wb')
            pickle.dump(data_save, fw, protocol = 4)
            fw.close()
            self.save = 'Y'
            try:
                error2.destroy()
                self.master.destroy()
            except:
                pass
        except Exception as e:
            tk.messagebox.showwarning('错误', e)
    def valid_report(self, s_group_report, f_group_report):
        colum = self.varnum + self.varchar
        vadild_data = self.grouped_valid_data
        vaild_ana = pd.DataFrame()
        for col in colum:
            tt1 = vadild_data.groupby(['f_group_%s' % col]).agg(
                {self.target_train: ['mean', 'count', 'sum']}).reset_index().rename(
                {'mean': 'badrate_vaild', 'count': 'n_vaild', 'sum': 'bad_n_vaild'}, axis=1)
            tt1['variable_name'] = col

            temp = pd.DataFrame()
            temp['variable_name'] = tt1['variable_name']
            temp['badrate_vaild'] = tt1[self.target_train]['badrate_vaild']
            temp['n_vaild'] = tt1[self.target_train]['n_vaild']
            temp['bad_n_vaild'] = tt1[self.target_train]['bad_n_vaild']
            temp['f_group'] = tt1['f_group_%s' % col]

            dd = pd.merge(f_group_report[f_group_report['variable_name'] == col], temp, how='outer',
                          on=['variable_name', 'f_group'])
            n_bad_vaild = dd['bad_n_vaild'].sum()
            n_vaild = dd['n_vaild'].sum()
            n_bad_train = dd['f_N_bad'].sum()
            n_train = dd['f_N_obs'].sum()
            dd['pct_n_bad_vaild'] = (1 + dd['bad_n_vaild']) / (1 + n_bad_vaild)
            dd['pct_n_vaild'] = (1 + dd['n_vaild']) / (1 + n_vaild)
            dd['pct_f_N_bad'] = (1 + dd['f_N_bad']) / (1 + n_bad_train)
            dd['pct_f_N_obs'] = (1 + dd['f_N_obs']) / (1 + n_train)
            dd['psi_bad_min'] = (dd['pct_n_bad_vaild'] - dd['pct_f_N_bad'])
            dd['psi_bad_log'] = dd.apply(lambda x: math.log(x['pct_n_bad_vaild'] / x['pct_f_N_bad']), axis=1)
            dd['psi_bad'] = (dd['psi_bad_min'] * dd['psi_bad_log']).sum()
            dd['psi_n_min'] = (dd['pct_n_vaild'] - dd['pct_f_N_obs'])
            dd['psi_n_log'] = dd.apply(lambda x: math.log(x['pct_n_vaild'] / x['pct_f_N_obs']), axis=1)
            dd['psi_n'] = (dd['psi_n_min'] * dd['psi_n_log']).sum()
            dd['badrate_vaild'] = dd.apply(lambda x: "%.2f%%" % (x['badrate_vaild'] * 100), axis=1)
            dd['pct_f_N_obs'] = dd.apply(lambda x: "%.2f%%" % (x['pct_f_N_obs'] * 100), axis=1)
            dd['psi_n'] = round(dd['psi_n'], 4)
            dd['psi_bad'] = round(dd['psi_bad'], 4)
            dd = dd[['variable_name', 'f_group', 'pct_f_N_obs', 'badrate_vaild', 'psi_n', 'psi_bad']]
            dd = dd.drop_duplicates()
            vaild_ana = vaild_ana.append(dd)
        s_group_report_vaild = pd.merge(s_group_report, vaild_ana, how='left', on=['variable_name', 'f_group'])
        f_group_df_vaild = pd.merge(f_group_report, vaild_ana, how='left', on=['variable_name', 'f_group'])
        return s_group_report_vaild, f_group_df_vaild
    def bing_restdata(self):
        if self.par_valid_data.empty == False:
            self.grouped_valid_data = binning.fit_bin_existing(data=self.par_valid_data,
                                                               varnum=self.varnum,
                                                               varchar=self.varchar,
                                                               target=self.target_train,
                                                               group_info=self.groupingdata,
                                                               data_only=True)
        if self.par_reject_data.empty == False:
            self.grouped_reject_data = binning.fit_bin_existing(data=self.par_reject_data,
                                                                varnum=self.varnum_reject,
                                                                varchar=self.varchar_reject,
                                                                target=self.target_reject,
                                                                group_info=self.groupingdata,
                                                                data_only=True)
        if self.par_oot_data.empty == False:
            self.grouped_oot_data = binning.fit_bin_existing(data=self.par_oot_data,
                                                             varnum=self.varnum_oot,
                                                             varchar=self.varchar_oot,
                                                             target=self.target_oot,
                                                             group_info=self.groupingdata,
                                                             data_only=True)

        if self.par_valid_data.empty == True:
            self.s_group_report, self.f_group_report = binning.report(group_info=self.groupingdata,
                                                                      varchar=self.varchar,
                                                                      varnum=self.varnum)
        else:
            s_group_report, f_group_report = binning.report(group_info=self.groupingdata, varchar=self.varchar,
                                                            varnum=self.varnum)
            self.s_group_report, self.f_group_report = self.valid_report(s_group_report, f_group_report)