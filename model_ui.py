import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
from pandastable import Table
import pickle as pickle
from .func import binning
import math
from .model import lrmodel

lrmodel = lrmodel()
import datetime

binning = binning()
import statsmodels.api as sm
from sklearn.linear_model.logistic import LogisticRegression
from joblib import Parallel, delayed
import joblib
from .model_result_ui import scorecard_result_ui
import threading
from sklearn.metrics import roc_curve, auc
import os
import numpy as np
import random
from .var_clus import VarClus


class model():
    def __init__(self, mianframe, project_info):
        self.master = mianframe
        # project参数
        self.save = 'N'
        self.node_type = 'SCR'
        self.project_info = project_info
        self.project_path = os.path.split(project_info[project_info['模块类型'] == 'project']['保存地址'][0])[0]
        self.node_name = 'model'
        self.exist_data = list(project_info['模块名字'])
        self.load = 'N'
        self.finsh = 'N'
        self.n_job = joblib.cpu_count() - 2
        # IGN参数
        self.IGN_node_name = None
        self.IGN_par_train_dataname = None
        self.IGN_par_reject_dataname = None
        self.IGN_par_oot_dataname = None
        self.IGN_par_use_freezing_flag = None
        self.IGN_par_import_groupdataname = None
        self.IGN_par_num_s_bin = None
        self.IGN_par_use_specialcode_flag = None
        self.IGN_par_specialcode_data = None
        self.IGN_par_sepcialcode_dataname = None
        self.IGN_par_char_restric_flag = None
        self.IGN_par_char_restric_num = None
        self.IGN_par_char_restric_pct = None
        self.IGN_par_tree_criterion = None
        self.IGN_par_num_f_group = None
        self.IGN_par_min_num_group = None
        self.IGN_par_min_pct_group = None
        self.IGN_par_variable_reject_flag = None
        self.IGN_par_variable_reject_iv = None
        self.IGN_IGNvariable_setting = None
        self.IGN_groupingdata = pd.DataFrame()
        self.IGN_f_group_report = pd.DataFrame()
        self.IGN_s_group_report = pd.DataFrame()
        self.IGN_not_use = []
        self.IGN_grouped_train_data = pd.DataFrame()
        self.IGN_grouped_valid_data = pd.DataFrame()
        self.IGN_grouped_reject_data = pd.DataFrame()
        self.IGN_grouped_oot_data = pd.DataFrame()
        self.IGN_check_list = None
        self.IGN_par_traindatavariable_setting = None
        self.IGN_node_time = None
        self.timeid_reject = None
        self.timeid_oot = None
        self.timeid_train = None
        self.target_reject = None
        self.target_oot = None
        self.target_train = None
        # 模型参数
        self.par_use_freezing_flag = '否'
        self.par_inditor_help = False
        self.par_import_modelname = None
        self.par_intercept_flag = True
        self.par_p_value = 0.05
        self.par_stay_p_value = 0.05
        self.par_criterion = 'aic'
        self.par_direction = 'stepwise'
        self.par_variable_type = 'WOE'
        self.par_odds_ratio = 20
        self.par_odds_score_ratio = 600
        self.par_odds_double_score = 20
        self.par_intercept_scorecard = '是'
        self.par_inditor_pct = 0.01
        self.par_inditor_sample = 0.01
        self.model_start_flag = 'N'
        self.lasso_flag = '否'
        # 分组过程参数
        # 评分卡变量
        self.predict_train_data = pd.DataFrame()
        self.predict_vaild_data = pd.DataFrame()
        self.predict_reject_data = pd.DataFrame()
        self.predict_oot_data = pd.DataFrame()
        self.model_ppp = []
        self.scorecard_df = pd.DataFrame()
        self.lasso_df = pd.DataFrame()
        self.pre_data()

    # 模块参数
    def thread_it(self, func, *args):
        '''将函数放入线程中执行'''
        # 创建线程
        t = threading.Thread(target=func, args=args)
        # 守护线程
        t.setDaemon(True)
        # 启动线程
        t.start()

    def import_node(self, node_data, ac):
        self.node_setting = node_data[0]
        self.node_type = node_data[0]['node_type']
        self.node_name = node_data[0]['node_name']
        self.node_time = node_data[0]['time']
        self.node_save_path = node_data[0]['node_save_path']
        # self.par_use_freezing_flag=node_data[0]['ign_node']
        self.par_use_freezing_flag = node_data[0]['par_use_freezing_flag']
        self.par_inditor_help = node_data[0]['par_inditor_help']
        self.par_import_modelname = node_data[0]['par_import_modelname']
        self.par_intercept_flag = node_data[0]['par_intercept_flag']
        self.par_p_value = node_data[0]['par_p_value']
        self.par_stay_p_value = node_data[0]['par_stay_p_value']
        self.par_criterion = node_data[0]['par_criterion']
        self.par_direction = node_data[0]['par_direction']
        self.par_variable_type = node_data[0]['par_variable_type']
        self.par_odds_ratio = node_data[0]['par_odds_ratio']
        self.par_odds_score_ratio = node_data[0]['par_odds_score_ratio']
        self.par_odds_double_score = node_data[0]['par_odds_double_score']
        self.par_intercept_scorecard = node_data[0]['par_intercept_scorecard']
        # 分组过程参数
        # 评分卡变量
        self.predict_train_data = node_data[0]['predict_train_data']
        self.predict_vaild_data = node_data[0]['predict_vaild_data']
        self.predict_reject_data = node_data[0]['predict_reject_data']
        self.predict_oot_data = node_data[0]['predict_oot_data']
        self.model_ppp = node_data[0]['model']
        self.f_scorecard = node_data[0]['scorecard_df']
        self.lasso_df = node_data[0]['lasso_df']
        self.lasso_flag = node_data[0]['lasso_flag']
        self.var_clus = node_data[0]['var_clus']
        self.target_train = node_data[0]['report_para']['train_target']
        self.target_oot = node_data[0]['report_para']['oot_target']
        self.target_reject = node_data[0]['report_para']['reject_target']
        self.timeid_train = node_data[0]['report_para']['timeid_train']
        self.timeid_oot = node_data[0]['report_para']['timeid_oot']
        self.timeid_reject = node_data[0]['report_para']['timeid_reject']
        self.IGN_f_group_report = node_data[0]['report_para']['f_group_report']
        self.vari_list = node_data[0]['report_para']['vari_list']
        self.IGN_node_name = node_data[0]['previous_node_name'][0]
        self.IGN_node_time = node_data[0]['previous_node_time'][0]
        self.IGN_groupingdata = node_data[0]['IGN_grouping_data']
        self.load = 'Y'

        if ac == 'setting':
            path_list = self.project_info[self.project_info['创建时间'] == self.IGN_node_time]['保存地址']
            error_list = []

            if len(path_list) == 0:
                error_list = error_list + [{'name': self.IGN_node_name, 'time': self.IGN_node_time}]

            def continu(event):
                for child in self.master.winfo_children():
                    child.destroy()
                # 以前数据集更新了就重新更新结果
                self.load = 'N'
                self.Start_UI()
                self.adjustsetting()

            def back(event):
                self.IGN_grouped_train_data = self.predict_train_data
                self.IGN_grouped_valid_data = self.predict_vaild_data
                self.IGN_grouped_reject_data = self.predict_reject_data
                self.IGN_grouped_oot_data = self.predict_oot_data
                for child in self.master.winfo_children():
                    child.destroy()
                self.Start_UI()
                self.adjustsetting()

            if len(error_list) > 0:
                self.master.title('提示')
                L00 = Label(self.master, width=80, text="该模块引用的%s 模块 没有在项目中找到，\n可能该模块已经更新，删除，"
                                                        "或未导入\n您可以重新选择分组数据或查看旧模型信息" % (error_list))
                L00.grid(column=0, row=0, columnspan=3, sticky=(W))
                button_contin = ttk.Button(self.master, text='重新选择分组')
                button_contin.grid(column=0, row=1, sticky=(W), padx=10, pady=10)
                button_contin.bind("<Button-1>", continu)
                button_back = ttk.Button(self.master, text='查看旧模型')
                button_back.grid(column=2, row=1, sticky=(W), padx=10, pady=10)
                button_back.bind("<Button-1>", back)
            else:
                try:
                    path = path_list[0]
                    fr = open(path, 'rb')
                    node_data = pickle.load(fr)
                    fr.close()
                    self.IGN_node_name = node_data[0]['node_name']
                    self.IGN_node_time = node_data[0]['time']
                    self.IGN_previous_usedlist = node_data[0]['use_node']
                    self.IGN_par_train_dataname = node_data[0]['previous_node_name'][0]
                    self.IGN_par_reject_dataname = node_data[0]['previous_node_name'][1]
                    self.IGN_par_oot_dataname = node_data[0]['previous_node_name'][2]
                    self.IGN_par_use_freezing_flag = node_data[0]['par_use_freezing_flag']
                    self.IGN_par_import_groupdataname = node_data[0]['par_import_groupdataname']
                    self.IGN_par_num_s_bin = node_data[0]['par_num_s_bin']
                    self.IGN_par_use_specialcode_flag = node_data[0]['par_use_specialcode_flag']
                    self.IGN_par_specialcode_data = node_data[0]['par_specialcode_data']
                    self.IGN_par_sepcialcode_dataname = node_data[0]['par_sepcialcode_dataname']
                    self.IGN_par_char_restric_flag = node_data[0]['par_char_restric_flag']
                    self.IGN_par_char_restric_num = node_data[0]['par_char_restric_num']
                    self.IGN_par_char_restric_pct = node_data[0]['par_char_restric_pct']
                    self.IGN_par_tree_criterion = node_data[0]['par_tree_criterion']
                    self.IGN_par_num_f_group = node_data[0]['par_num_f_group']
                    self.IGN_par_min_num_group = node_data[0]['par_min_num_group']
                    self.IGN_par_min_pct_group = node_data[0]['par_min_pct_group']
                    self.IGN_par_traindatavariable_setting = node_data[0]['data_variable_setting']
                    self.IGN_par_variable_reject_flag = node_data[0]['par_variable_reject_flag']
                    self.IGN_par_variable_reject_iv = node_data[0]['par_variable_reject_iv']
                    self.IGN_IGNvariable_setting = node_data[0]['IGNvariable_setting']
                    self.IGN_groupingdata = node_data[1][0]
                    self.IGN_f_group_report = node_data[1][1]
                    self.IGN_s_group_report = node_data[1][2]
                    self.IGN_not_use = node_data[1][3]
                    self.IGN_grouped_train_data = node_data[2]
                    self.IGN_grouped_valid_data = node_data[3]
                    self.IGN_grouped_reject_data = node_data[4]
                    self.IGN_grouped_oot_data = node_data[5]
                    # self.IGN_check_list = node_data[0]['check_change']
                    self.IGN_IGNvariable_setting['是否使用'] = self.IGN_IGNvariable_setting.apply(
                        lambda x: '不使用' if x['变量名称'] in self.IGN_not_use else x['是否使用'], axis=1)
                except  Exception as e:
                    self.IGN_grouped_train_data = self.predict_train_data
                    self.IGN_grouped_valid_data = self.predict_vaild_data
                    self.IGN_grouped_reject_data = self.predict_reject_data
                    self.IGN_grouped_oot_data = self.predict_oot_data
                    tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (self.IGN_node_name, e))
            if ac == 'setting' and len(error_list) == 0:
                self.Start_UI()
                self.adjustsetting()
        # 'data_variable_setting': self.par_traindatavariable_setting,
        # 'reject_data_variable_setting': self.par_rejectdatavariable_setting,
        # 'oot_data_variable_setting': self.par_ootdatavariable_setting,
        # node_data[0]['use_node': [self.node_name] + self.IGN_previous_usedlist

    def load_data(self, event, datatype):
        try:
            if datatype == 'train':
                if len(str(self.comboxlist_train_data.get())) < 1:
                    tk.messagebox.showwarning('错误', '请先创建交互式分组模块')
                else:
                    path = self.project_info[self.project_info['模块名字'] == self.comboxlist_train_data.get()]['保存地址'][0]
                    fr = open(path, 'rb')
                    node_data = pickle.load(fr)
                    fr.close()
                    self.IGN_node_name = node_data[0]['node_name']
                    self.IGN_node_time = node_data[0]['time']
                    self.IGN_previous_usedlist = node_data[0]['use_node']
                    self.IGN_par_train_dataname = node_data[0]['previous_node_name'][0]
                    self.IGN_par_reject_dataname = node_data[0]['previous_node_name'][1]
                    self.IGN_par_oot_dataname = node_data[0]['previous_node_name'][2]
                    self.IGN_par_use_freezing_flag = node_data[0]['par_use_freezing_flag']
                    self.IGN_par_import_groupdataname = node_data[0]['par_import_groupdataname']
                    self.IGN_par_num_s_bin = node_data[0]['par_num_s_bin']
                    self.IGN_par_use_specialcode_flag = node_data[0]['par_use_specialcode_flag']
                    self.IGN_par_specialcode_data = node_data[0]['par_specialcode_data']
                    self.IGN_par_sepcialcode_dataname = node_data[0]['par_sepcialcode_dataname']
                    self.IGN_par_char_restric_flag = node_data[0]['par_char_restric_flag']
                    self.IGN_par_char_restric_num = node_data[0]['par_char_restric_num']
                    self.IGN_par_char_restric_pct = node_data[0]['par_char_restric_pct']
                    self.IGN_par_tree_criterion = node_data[0]['par_tree_criterion']
                    self.IGN_par_num_f_group = node_data[0]['par_num_f_group']
                    self.IGN_par_min_num_group = node_data[0]['par_min_num_group']
                    self.IGN_par_min_pct_group = node_data[0]['par_min_pct_group']
                    self.IGN_par_traindatavariable_setting = node_data[0]['data_variable_setting']
                    self.IGN_par_variable_reject_flag = node_data[0]['par_variable_reject_flag']
                    self.IGN_par_variable_reject_iv = node_data[0]['par_variable_reject_iv']
                    self.IGN_IGNvariable_setting = node_data[0]['IGNvariable_setting']
                    self.IGN_groupingdata = node_data[1][0]
                    self.IGN_f_group_report = node_data[1][1]
                    self.IGN_s_group_report = node_data[1][2]
                    self.IGN_not_use = node_data[1][3]
                    self.IGN_grouped_train_data = node_data[2]
                    self.IGN_grouped_valid_data = node_data[3]
                    self.IGN_grouped_reject_data = node_data[4]
                    self.IGN_grouped_oot_data = node_data[5]
                    # self.IGN_check_list = node_data[0]['check_change']
                    self.IGN_IGNvariable_setting['是否使用'] = self.IGN_IGNvariable_setting.apply(
                        lambda x: '不使用' if x['变量名称'] in self.IGN_not_use else x['是否使用'], axis=1)
        except  Exception as e:
            tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (datatype, e))

    def pre_data(self):

        try:
            dd = list(self.project_info[(self.project_info['模块类型'] == 'IGN') &
                                        (self.project_info['状态'] == 'Good')]['模块名字'])
            self.IGN_list = dd
        except Exception as e:
            tk.messagebox.showwarning('错误', e)

    def Start_UI(self):
        self.start_window_base = self.master
        width = self.master.winfo_screenwidth() * 0.2
        height = self.master.winfo_screenheight() * 0.7
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()
        self.start_window_base.geometry(
            '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
        self.start_window_base.title('评分卡模型')

    def adjustsetting(self):
        # 导入数据
        self.node_intro = LabelFrame(self.start_window_base, text='模块名称:')
        L8 = Label(self.node_intro, width=25, text="模块名称:", justify="left")
        L8.grid(column=0, row=0, sticky=(W))
        if (self.load == 'N') & (self.finsh == 'N'):
            node_name = tk.StringVar(value=self.node_name)
            self.entry_node_name = Entry(self.node_intro, textvariable=node_name, bd=1, width=18)
            self.entry_node_name.grid(column=1, row=0, sticky=(W))
        else:
            L88 = Label(self.node_intro, width=25, text="%s" % self.node_name, justify="left")
            L88.grid(column=1, row=0, sticky=(W))
        self.node_intro.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        self.start_window_data = LabelFrame(self.start_window_base, text='导入分组数据:')
        L1 = Label(self.start_window_data, width=25, text="分组数据:", justify="left")
        L1.grid(column=0, row=0, sticky=(W))
        self.comboxlist_train_data = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_train_data["value"] = self.IGN_list
        if self.IGN_grouped_train_data.empty != True:
            for i in range(len(self.IGN_list)):
                if self.IGN_list[i] == self.IGN_node_name:
                    self.comboxlist_train_data.current(i)
        self.comboxlist_train_data.bind("<<ComboboxSelected>>", lambda event: self.load_data(event, datatype='train'))
        self.comboxlist_train_data.grid(column=1, row=0, sticky=(W))

        L3 = Label(self.start_window_data, width=25, text="自变量:", justify="left")
        L3.grid(column=0, row=2, sticky=(W))
        self.comboxlist_variable_type = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_variable_type["value"] = ['WOE', 'GRP', 'GRP_ind']
        if self.par_variable_type == 'WOE':
            self.comboxlist_variable_type.current(0)
        else:
            self.comboxlist_variable_type.current(1)

        self.comboxlist_variable_type.grid(column=1, row=2, sticky=(W))

        L4 = Label(self.start_window_data, width=25, text="变量设置:", justify="left")
        L4.grid(column=0, row=3, sticky=(W))
        self.button_data_variablesetting = ttk.Button(self.start_window_data, text='设置:')
        self.button_data_variablesetting.grid(column=1, row=3, sticky=(W))
        self.button_data_variablesetting.bind("<Button-1>", self.show_variabledetail)

        L8 = Label(self.start_window_data, width=25, text="冻结入模变量:", justify="left")
        L8.grid(column=0, row=5, sticky=(W))
        self.comboxlist_freezing_code = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_freezing_code["value"] = ['是', '否']
        if self.par_use_freezing_flag == '否':
            self.comboxlist_freezing_code.current(1)
        else:
            self.comboxlist_freezing_code.current(0)
        self.comboxlist_freezing_code.grid(column=1, row=5, sticky=(W))

        self.start_window_data.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        # 模型参数

        # self.start_window_model_setting = LabelFrame(self.start_window_base, text='模型参数:')

        # L5 = Label(self.start_window_model_setting, width=20, text="导入模型数据:")
        # L5.grid(column=0, row=6, sticky=(W))
        #
        # L55 = Label(self.start_window_model_setting, width=20, text=self.par_import_modelname)
        # L55.grid(column=1, row=6, sticky=(W))
        # self.button_data_grouping_data_import = ttk.Button(self.start_window_model_setting, text='导入:')
        # self.button_data_grouping_data_import.grid(column=1, row=7, sticky=(W))
        # # self.button_data_grouping_data_import.bind("<Button-1>", self.loading_grouping_data)
        # self.start_window_model_setting.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        # 模型设置

        self.start_window_LR_setting = LabelFrame(self.start_window_base, text='模型参数设置:')

        L6 = Label(self.start_window_LR_setting, width=25, text="模型方法:", justify="left")
        L6.grid(column=0, row=5, sticky=(W))
        L7 = Label(self.start_window_LR_setting, width=25, text="逻辑回归", bd=1, justify="left")
        L7.grid(column=1, row=5, sticky=(W))

        L8 = Label(self.start_window_LR_setting, width=25, text="模型评价标准:", justify="left")
        L8.grid(column=0, row=6, sticky=(W))
        self.comboxlist_model_creterion = ttk.Combobox(self.start_window_LR_setting, width=15)
        self.comboxlist_model_creterion["value"] = ['aic', 'bic', 'llr']
        if self.par_criterion == 'aic':
            self.comboxlist_model_creterion.current(0)
        elif self.par_criterion == 'bic':
            self.comboxlist_model_creterion.current(1)
        else:
            self.comboxlist_model_creterion.current(2)
        self.comboxlist_model_creterion.grid(column=1, row=6, sticky=(W))
        # self.entry_s_bin_num.bind('<Return>', lambda event: self.int_num_check(event, 'entry_s_bin_num', 'int'))

        L9 = Label(self.start_window_LR_setting, width=25, text="模型方向:", justify="left")
        L9.grid(column=0, row=7, sticky=(W))
        self.comboxlist_model_direction = ttk.Combobox(self.start_window_LR_setting, width=15)
        self.comboxlist_model_direction["value"] = ['NO', 'forward', 'stepwise']
        if self.par_direction == 'NO':
            self.comboxlist_model_direction.current(0)
        elif self.par_direction == 'forward':
            self.comboxlist_model_direction.current(1)
        else:
            self.comboxlist_model_direction.current(2)
        self.comboxlist_model_direction.grid(column=1, row=7, sticky=(W))

        L10 = Label(self.start_window_LR_setting, width=25, text="变量进入模型P值:", justify="left")
        L10.grid(column=0, row=8, sticky=(W))
        pvalue = tk.StringVar(value=self.par_p_value)
        self.entry_pvalue = Entry(self.start_window_LR_setting, textvariable=pvalue, bd=1, width=18)
        self.entry_pvalue.grid(column=1, row=8, sticky=(W))

        L10_1 = Label(self.start_window_LR_setting, width=25, text="变量保留模型P值:", justify="left")
        L10_1.grid(column=0, row=9, sticky=(W))
        pvalue = tk.StringVar(value=self.par_stay_p_value)
        self.entry_s_pvalue = Entry(self.start_window_LR_setting, textvariable=pvalue, bd=1, width=18)
        self.entry_s_pvalue.grid(column=1, row=9, sticky=(W))

        L11 = Label(self.start_window_LR_setting, width=25, text="模型训练使用截距:", justify="left")
        L11.grid(column=0, row=10, sticky=(W))
        self.comboxlist_model_intercept = ttk.Combobox(self.start_window_LR_setting, width=15)
        self.comboxlist_model_intercept["value"] = ['是', '否']
        if self.par_intercept_flag == True:
            self.comboxlist_model_intercept.current(0)
        else:
            self.comboxlist_model_intercept.current(1)
        self.comboxlist_model_intercept.grid(column=1, row=10, sticky=(W))

        L12 = Label(self.start_window_LR_setting, width=25, text="模型训练是否使用辅助变量:", justify="left")
        L12.grid(column=0, row=11, sticky=(W))
        self.comboxlist_inditor_help = ttk.Combobox(self.start_window_LR_setting, width=15)
        self.comboxlist_inditor_help["value"] = ['是', '否']
        if self.par_inditor_help == True:
            self.comboxlist_inditor_help.current(0)
        else:
            self.comboxlist_inditor_help.current(1)
        self.comboxlist_inditor_help.grid(column=1, row=11, sticky=(W))
        self.par_inditor_pct = 0.01
        self.par_inditor_sample = 0.01
        L10_12 = Label(self.start_window_LR_setting, width=20, text="辅助变量最小坏账率偏移:", justify="left")
        L10_12.grid(column=0, row=12, sticky=(W))
        par_inditor_pct = tk.StringVar(value=self.par_inditor_pct)
        self.entry_par_inditor_pct = Entry(self.start_window_LR_setting, textvariable=par_inditor_pct, bd=1, width=18)
        self.entry_par_inditor_pct.grid(column=1, row=12, sticky=(W))

        L10_13 = Label(self.start_window_LR_setting, width=20, text="辅助变量最小样本占比:", justify="left")
        L10_13.grid(column=0, row=13, sticky=(W))
        par_inditor_sample = tk.StringVar(value=self.par_inditor_sample)
        self.entry_par_inditor_sample = Entry(self.start_window_LR_setting, textvariable=par_inditor_sample, bd=1,
                                              width=18)
        self.entry_par_inditor_sample.grid(column=1, row=13, sticky=(W))

        L13 = Label(self.start_window_LR_setting, width=20, text="模型训练并行数:", justify="left")
        L13.grid(column=0, row=14, sticky=(W))
        self.comboxlist_n_job = ttk.Combobox(self.start_window_LR_setting, width=15)
        self.comboxlist_n_job["value"] = [x for x in range(1, max(self.n_job + 1, joblib.cpu_count() - 1))]
        self.comboxlist_n_job.current(self.n_job - 1)
        self.comboxlist_n_job.grid(column=1, row=14, sticky=(W))

        L14 = Label(self.start_window_LR_setting, width=20, text="LASSO变量选择:", justify="left")
        L14.grid(column=0, row=15, sticky=(W))
        self.comboxlist_lasso_flag = ttk.Combobox(self.start_window_LR_setting, width=15)
        self.comboxlist_lasso_flag["value"] = ['是', '否']
        if self.lasso_flag == '否':
            self.comboxlist_lasso_flag.current(1)
        else:
            self.comboxlist_lasso_flag.current(0)
        self.comboxlist_lasso_flag.grid(column=1, row=15, sticky=(W))

        self.start_window_LR_setting.grid(columnspan=3, sticky=(W), padx=10, pady=10)
        # 评分卡设置

        self.start_window_scorecard_setting = LabelFrame(self.start_window_base, text='评分卡设置:')
        L15 = Label(self.start_window_scorecard_setting, width=20, text="是否显示截距", justify="left")
        L15.grid(column=0, row=4, sticky=(W))
        self.comboxlist_intercept_scorecard = ttk.Combobox(self.start_window_scorecard_setting, width=15)
        self.comboxlist_intercept_scorecard["value"] = ['是', '否']
        if self.par_intercept_scorecard == '是':
            self.comboxlist_intercept_scorecard.current(0)
        else:
            self.comboxlist_intercept_scorecard.current(1)
        self.comboxlist_intercept_scorecard.grid(column=1, row=4, sticky=(W))

        odds_ratio = tk.StringVar(value=self.par_odds_ratio)
        L13 = Label(self.start_window_scorecard_setting, width=20, text="优比:", justify="left")
        L13.grid(column=0, row=5, sticky=(W))
        self.entry_odds_ratio = Entry(self.start_window_scorecard_setting, textvariable=odds_ratio, bd=1, width=18)
        self.entry_odds_ratio.grid(column=1, row=5, sticky=(W))
        # self.entry_odds_ratio.bind('<Return>', lambda event: self.int_num_check(event, 'entry_min_num_char', 'gg'))

        odds_score_ratio = tk.StringVar(value=self.par_odds_score_ratio)
        L14 = Label(self.start_window_scorecard_setting, width=20, text="评分卡点数:", justify="left")
        L14.grid(column=0, row=6, sticky=(W))
        self.entry_odds_score = Entry(self.start_window_scorecard_setting, bd=1, textvariable=odds_score_ratio,
                                      width=18)
        self.entry_odds_score.grid(column=1, row=6, sticky=(W))
        # self.entry_odds_score.bind('<Return>', lambda event: self.int_num_check(event, 'entry_min_pct_char', 'gg'))

        self.start_window_scorecard_setting.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        L10 = Label(self.start_window_scorecard_setting, width=20, text="翻倍点数:", justify="left")
        L10.grid(column=0, row=10, sticky=(W))
        f_num_bin = tk.StringVar(value=self.par_odds_double_score)
        self.entry_f_bin_num = Entry(self.start_window_scorecard_setting, textvariable=f_num_bin, width=18, bd=1)
        self.entry_f_bin_num.grid(column=1, row=10, sticky=(W))
        # self.entry_f_bin_num.bind('<Return>', lambda event: self.int_num_check(event, 'entry_f_bin_num', 'gg'))

        self.button_setting_save = ttk.Button(self.start_window_base, text='（保存）退出')
        self.button_setting_save.grid(column=0, row=7, sticky=(W), padx=10, pady=10)
        self.button_setting_save.bind("<Button-1>", self.save_project)
        if (self.load == 'Y') | (self.finsh == 'Y'):
            self.check_result = ttk.Button(self.start_window_base, text='查看结果')
            self.check_result.grid(column=1, row=7, sticky=(W), padx=10, pady=10)
            self.check_result.bind("<Button-1>", self.scorecard_result_show_ui)
        if (self.load == 'N') & (self.finsh == 'N'):
            self.button_setting_run = ttk.Button(self.start_window_base, text='应用'
                                                 # ,command=lambda event :self.thread_it(self.LR() , event)
                                                 )
            self.button_setting_run.grid(column=2, row=7, sticky=(W))
            # self.button_setting_run.bind("<Button-1>", lambda event :self.thread_it(self.LR(event)  ))
            self.button_setting_run.bind("<Button-1>", self.LR)
        else:
            self.button_refresh_run = ttk.Button(self.start_window_base, text='刷新结果')
            self.button_refresh_run.grid(column=2, row=7, sticky=(W))
            self.button_refresh_run.bind("<Button-1>", self.LR)
            self.button_modify_manually = ttk.Button(self.start_window_base, text='手动调整变量')
            self.button_modify_manually.grid(column=0, row=8, sticky=(W), padx=10, pady=10)
            self.button_modify_manually.bind("<Button-1>", self.modify_model)

            self.button_output = ttk.Button(self.start_window_base, text='导出数据集')
            self.button_output.grid(column=1, row=8, sticky=(W), padx=10, pady=10)
            self.button_output.bind("<Button-1>", self.out_dataset)

    def out_dataset(self, event):
        try:
            word = '导出数据集：\n 训练集：%s/%s_train.csv \n' % (self.project_path, self.node_name)
            self.predict_train_data.to_csv(self.project_path + '/' + '%s_train.csv' % self.node_name, index=False,
                                           encoding='utf-8')
            if self.predict_vaild_data.empty == False:
                word = word + '验证集：%s/%s_valid.csv \n' % (self.project_path, self.node_name)
                self.predict_vaild_data.to_csv(self.project_path + '/' + '%s_valid.csv' % self.node_name, index=False,
                                               encoding='utf-8')
            if self.predict_reject_data.empty == False:
                word = word + '验证集：%s/%s_reject.csv \n' % (self.project_path, self.node_name)
                self.predict_reject_data.to_csv(self.project_path + '/' + '%s_reject.csv' % self.node_name, index=False,
                                                encoding='utf-8')
            if self.predict_oot_data.empty == False:
                word = word + '验证集：%s/%s_oot.csv \n' % (self.project_path, self.node_name)
                self.predict_oot_data.to_csv(self.project_path + '/' + '%s_oot.csv' % self.node_name, index=False,
                                             encoding='utf-8')
            tk.messagebox.showwarning('成功', word)
        except  Exception as e:
            tk.messagebox.showwarning('错误', e)

    def modify_model(self, event):
        try:
            if self.modify_model_ui.state() == 'normal':
                tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
        except:
            self.df_modify_model, self.record_list_modify, self.model_modify, self.model_variable_df_modify = self.add_delet_var(
                record_list=self.model_ppp[0]
                , input_model=self.model_ppp[1], model_variable_df=self.model_ppp[2], modify_var='f', flag='first',
                var_list=self.vari_list, n_job=self.n_job,
                predict_train_data=self.predict_train_data,
                target_train=self.target_train, predict_vaild_data=self.predict_vaild_data,
                par_intercept_flag=self.par_intercept_flag, par_variable_type=self.par_variable_type
                )
            self.modify_model_ui = Toplevel(self.start_window_base)
            f = LabelFrame(self.modify_model_ui, text='调整模型')
            screen_width = f.winfo_screenwidth() * 0.5
            screen_height = f.winfo_screenheight() * 0.8
            self.table_sum = self.pt = Table(f, dataframe=self.df_modify_model, height=screen_height,
                                             width=screen_width)
            self.pt.show()
            self.table_sum.bind("<Button-3>", self.handle_left_click)
            # self.table.bind("<Button-1>",self.handle_left_click)
            self.table_sum.bind("<Button-2>", self.handle_left_click)
            # self.table_sum.bind("<Double-Button-3>", self.handle_left_click)
            # self.table_sum.bind("<Double-Button-1>", self.handle_left_click)
            # self.table_sum.bind("<Double-Button-2>", self.handle_left_click)
            self.table_sum.bind("<Triple-Button-3>", self.handle_left_click)
            self.table_sum.bind("<Triple-Button-1>", self.handle_left_click)
            self.table_sum.bind("<Triple-Button-2>", self.handle_left_click)
            f.pack()

    def handle_left_click(self, event):
        rowclicked = self.pt.get_row_clicked(event)
        self.modify_var = self.df_modify_model.iloc[rowclicked]['变量名称']
        if self.modify_var in list(self.df_modify_model[self.df_modify_model['是否在模型中'] == 'Y']['变量名称']):
            flag = tk.messagebox.askyesno('提示', '是否要把%s从模型中删除:' % self.modify_var)
            self.modify_var_flag = 'del'

        else:
            flag = tk.messagebox.askyesno('提示', '是否要把%s添加到现有模型中:' % self.modify_var)
            self.modify_var_flag = 'add'
        if flag == True:
            self.modify_model_calcu()

    def modify_model_calcu(self):
        self.df_modify_model, self.record_list_modify, self.model_modify, self.model_variable_df_modify = self.add_delet_var(
            record_list=self.record_list_modify, input_model=self.model_modify,
            model_variable_df=self.model_variable_df_modify, modify_var=self.modify_var, flag=self.modify_var_flag,
            var_list=self.vari_list, n_job=self.n_job, predict_train_data=self.predict_train_data,
            target_train=self.target_train, predict_vaild_data=self.predict_vaild_data,
            par_intercept_flag=self.par_intercept_flag, par_variable_type=self.par_variable_type)
        self.record_list_modify.append('手动%s %s' % (self.modify_var_flag, self.modify_var))
        self.record_list_modify.append(self.model_modify.summary2())
        self.modify_model_ui.destroy()
        self.modify_model_ui = Toplevel(self.master)

        def close(event):
            self.modify_model_ui.destroy()

        def model_save(event):
            self.modify_model_ui.destroy()
            error2_f = Toplevel(self.master)
            screenwidth = self.master.winfo_screenwidth()
            screenheight = self.master.winfo_screenheight()
            error2_f.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
            L2 = Label(error2_f, text="保存中。。。")
            L2.grid()
            self.master.update()
            self.model_ppp = [self.record_list_modify, self.model_modify, self.model_variable_df_modify, ]
            self.f_scorecard = self.scorecard_data_pre(self.model_ppp)
            self.node_setting['model'] = self.model_ppp
            try:
                error2_f.destroy()
            except:
                pass

        d = LabelFrame(self.modify_model_ui)
        self.check_result = ttk.Button(d, text='保存(关闭)')
        self.check_result.pack(side=LEFT)
        self.check_result.bind("<Button-1>", model_save)
        self.check_result = ttk.Button(d, text='关闭')
        self.check_result.pack(side=LEFT)
        self.check_result.bind("<Button-1>", close)
        d.pack()

        f = LabelFrame(self.modify_model_ui, text='调整模型')
        screen_width = f.winfo_screenwidth() * 0.5
        screen_height = f.winfo_screenheight() * 0.8
        self.table_sum = self.pt = Table(f, dataframe=self.df_modify_model, height=screen_height, width=screen_width)
        self.pt.show()
        self.table_sum.bind("<Button-3>", self.handle_left_click)
        # self.table.bind("<Button-1>",self.handle_left_click)
        self.table_sum.bind("<Button-2>", self.handle_left_click)
        self.table_sum.bind("<Double-Button-3>", self.handle_left_click)
        self.table_sum.bind("<Double-Button-1>", self.handle_left_click)
        self.table_sum.bind("<Double-Button-2>", self.handle_left_click)
        self.table_sum.bind("<Triple-Button-3>", self.handle_left_click)
        self.table_sum.bind("<Triple-Button-1>", self.handle_left_click)
        self.table_sum.bind("<Triple-Button-2>", self.handle_left_click)
        f.pack()

    def save_project(self, event):
        try:
            node_save_path = self.project_path + '/' + '%s.model' % self.node_name
            error2 = Toplevel(self.master)
            screenwidth = self.master.winfo_screenwidth()
            screenheight = self.master.winfo_screenheight()
            error2.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
            L2 = Label(error2, text="保存中")
            L2.grid()
            self.master.update()
            filename = node_save_path
            fw = open(filename, 'wb')
            pickle.dump([self.node_setting, 'A'], fw, protocol=4)
            fw.close()
            self.save = 'Y'
            try:
                error2.destroy()
            except:
                pass
            try:
                self.master.destroy()
            except:
                pass
        except Exception as e:
            tk.messagebox.showwarning('错误', e)

    def get_par(self):
        self.par_use_freezing_flag = self.comboxlist_freezing_code.get()
        self.par_import_modelname = None
        self.par_intercept_flag = self.comboxlist_model_intercept.get() == '是'
        self.par_p_value = float(self.entry_pvalue.get())
        self.par_stay_p_value = float(self.entry_s_pvalue.get())
        self.par_criterion = self.comboxlist_model_creterion.get()
        self.par_direction = self.comboxlist_model_direction.get()
        self.par_variable_type = self.comboxlist_variable_type.get()
        self.par_odds_ratio = float(self.entry_odds_ratio.get())
        self.par_odds_score_ratio = float(self.entry_odds_score.get())
        self.par_odds_double_score = float(self.entry_f_bin_num.get())
        self.par_inditor_help = self.comboxlist_inditor_help.get() == '是'
        self.par_intercept_scorecard = self.comboxlist_intercept_scorecard.get()
        self.par_inditor_pct = float(self.entry_par_inditor_pct.get())
        self.par_inditor_sample = float(self.entry_par_inditor_sample.get())
        self.lasso_flag = self.comboxlist_lasso_flag.get()
        self.n_job = int(self.comboxlist_n_job.get())
        if (self.finsh == 'N') & (self.load == 'N'):
            self.node_name = self.entry_node_name.get()

    def check_all_setting(self):
        self.get_par()
        mm = 0
        if (self.node_name in self.exist_data) & (self.load == 'N'):
            mm = mm + 1
            tk.messagebox.showwarning('错误', "该名称已经被占用，请更改")
        elif len(self.comboxlist_train_data.get()) < 1:
            mm = mm + 1
            tk.messagebox.showwarning('错误', "请选择分组数据")
        elif len(self.node_name) < 1:
            mm = mm + 1
            tk.messagebox.showwarning('错误', "请输入模块名称")
        else:
            total = ['par_odds_ratio', 'par_odds_score_ratio', 'par_odds_double_score', 'par_p_value',
                     'par_stay_p_value', 'par_inditor_pct', 'par_inditor_sample']
            for p in total:

                entry_p = p
                pp = self.int_num_check(entry_p)
                mm = mm + pp

        return mm

    def int_num_check(self,  entry_p):
        a = 0
        if entry_p == 'par_odds_ratio':
            inputnum = self.entry_odds_ratio.get()
            tip = '优比:'
        elif entry_p == 'par_odds_score_ratio':
            inputnum = self.entry_odds_score.get()
            tip = '评分卡点数:'
        elif entry_p == 'par_odds_double_score':
            inputnum = self.entry_f_bin_num.get()
            tip = '翻倍点数:'
        elif entry_p == 'par_p_value':
            inputnum = self.entry_pvalue.get()
            tip = '变量进入模型P值:'
        elif entry_p == 'par_stay_p_value':
            inputnum = self.entry_s_pvalue.get()
            tip = '变量保留模型P值:'
        elif entry_p == 'par_inditor_pct':
            inputnum = self.entry_par_inditor_pct.get()
            tip = '辅助变量坏账率偏移值:'
        elif entry_p == 'par_inditor_sample':
            inputnum = self.entry_par_inditor_sample.get()
            tip = '辅助变量样本占比:'
        else:
            pass


        try:
            num = float(inputnum)
        except Exception as e:
            a = a + 1
            tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
        return a

    def modify_variable_role(self, event):
        try:
            self.comboxlist_modify_f_group.destroy()
        except:
            pass
        self.rowclicked = self.ptm.get_row_clicked(event)
        self.colclicked = self.ptm.get_col_clicked(event)

        if list(self.IGN_IGNvariable_setting.columns)[self.colclicked] == '是否使用':
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
        self.table = self.ptm = Table(f, dataframe=self.IGN_IGNvariable_setting, colspan=7,
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
        variable = self.IGN_IGNvariable_setting.iloc[self.rowclicked]['变量名称']
        if variable in self.IGN_not_use:
            self.comboxlist_modify_f_group.destroy()
            tk.messagebox.showwarning('错误', "%s 已经再前一个模块被禁用" % variable)
        else:
            value = self.comboxlist_modify_f_group.get()
            self.IGN_IGNvariable_setting.iloc[self.rowclicked, self.colclicked] = value
            self.comboxlist_modify_f_group.destroy()
            self.refresh_datavariable_df()

    def LR(self, event):
        # 检查各个数据集变量情况
        # try:
            error_num = self.check_all_setting()
            if (error_num == 0) & (self.model_start_flag == 'N'):

                self.model_start_flag = 'Y'
                self.target_train = \
                    list(self.IGN_par_traindatavariable_setting[self.IGN_par_traindatavariable_setting['变量角色'] == '目标'][
                             '变量名称'])[0]
                if len(self.IGN_par_traindatavariable_setting[
                           self.IGN_par_traindatavariable_setting['变量角色'] == 'TimeID']) == 1:
                    self.flag_timeid_train = True
                    self.timeid_train = self.IGN_par_traindatavariable_setting.loc[
                        self.IGN_par_traindatavariable_setting['变量角色'] == 'TimeID'][
                        '变量名称'].values[0]
                else:
                    self.flag_timeid_train = False
                    self.timeid_train = None
                if self.IGN_grouped_reject_data.empty != True:
                    # 拒绝集变量
                    try:
                        self.target_reject = list(
                            self.IGN_IGNvariable_setting[self.IGN_IGNvariable_setting['变量角色_拒绝样本'] == '目标'][
                                '变量名称_拒绝样本'])[0]
                    except:
                        self.target_reject = None
                    self.varchar_reject = list(
                        self.IGN_IGNvariable_setting[(self.IGN_IGNvariable_setting['是否使用'] == '使用') &
                                                     (self.IGN_IGNvariable_setting[
                                                          '变量角色'] == '自变量') &
                                                     (self.IGN_IGNvariable_setting[
                                                          '变量类型'] == '字符型')][
                            '变量名称_拒绝样本'])
                    self.varnum_reject = list(
                        self.IGN_IGNvariable_setting[(self.IGN_IGNvariable_setting['是否使用'] == '使用') &
                                                     (self.IGN_IGNvariable_setting['变量角色'] == '自变量') &
                                                     (self.IGN_IGNvariable_setting['变量类型'] == '数值型')][
                            '变量名称_拒绝样本'])
                    if len(self.IGN_IGNvariable_setting[
                               self.IGN_IGNvariable_setting['变量角色_拒绝样本'] == 'TimeID']) == 1:
                        self.flag_timeid_reject = True
                        self.timeid_reject = self.IGN_IGNvariable_setting.loc[
                            self.IGN_IGNvariable_setting['变量角色_拒绝样本'] == 'TimeID']['变量名称_拒绝样本'].values[0]
                else:
                    self.target_reject = None
                    self.flag_timeid_reject = False
                    self.timeid_reject = None
                if self.IGN_grouped_oot_data.empty != True:
                    # oot变量
                    try:
                        self.target_oot = \
                            list(self.IGN_IGNvariable_setting[self.IGN_IGNvariable_setting['变量角色_时间外样本'] == '目标'][
                                     '变量名称_时间外样本'])[0]
                    except:
                        self.target_oot = None
                    self.varchar_oot = list(
                        self.IGN_IGNvariable_setting[(self.IGN_IGNvariable_setting['是否使用'] == '使用') &
                                                     (self.IGN_IGNvariable_setting['变量角色'] == '自变量') &
                                                     (self.IGN_IGNvariable_setting['变量类型'] == '字符型')][
                            '变量名称_时间外样本'])
                    self.varnum_oot = list(self.IGN_IGNvariable_setting[(self.IGN_IGNvariable_setting['是否使用'] == '使用') &
                                                                        (self.IGN_IGNvariable_setting[
                                                                             '变量角色'] == '自变量') &
                                                                        (self.IGN_IGNvariable_setting[
                                                                             '变量类型'] == '数值型')][
                                               '变量名称_时间外样本'])

                    if len(self.IGN_IGNvariable_setting[self.IGN_IGNvariable_setting['变量角色_时间外样本'] == 'TimeID']) == 1:
                        self.flag_timeid_oot = True
                        self.timeid_oot = \
                            self.IGN_IGNvariable_setting.loc[self.IGN_IGNvariable_setting['变量角色_时间外样本'] == 'TimeID'][
                                '变量名称_时间外样本'].values[0]
                else:
                    self.target_oot = None
                    self.flag_timeid_oot = False
                    self.timeid_oot = None

                # 训练集变量
                self.varchar = list(self.IGN_IGNvariable_setting[(self.IGN_IGNvariable_setting['是否使用'] == '使用') &
                                                                 (self.IGN_IGNvariable_setting['变量角色'] == '自变量') &
                                                                 (self.IGN_IGNvariable_setting['变量类型'] == '字符型')][
                                        '变量名称'])
                self.varnum = list(self.IGN_IGNvariable_setting[(self.IGN_IGNvariable_setting['是否使用'] == '使用') &
                                                                (self.IGN_IGNvariable_setting['变量角色'] == '自变量') &
                                                                (self.IGN_IGNvariable_setting['变量类型'] == '数值型')][
                                       '变量名称'])
                self.vari_list = self.varchar + self.varnum
                try:
                    if self.temp.state() == 'normal':
                        tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
                except:
                    error_num = self.check_all_setting()
                    if error_num == 0:

                        if (self.par_variable_type == 'WOE') and (
                                (self.par_use_freezing_flag == '否') | (self.model_ppp == [])):
                            self.model_ppp = lrmodel.woe_logistic_regression(mianframe=self.start_window_base,
                                                                             inditor_pct=self.par_inditor_pct,
                                                                             inditor_sample=self.par_inditor_sample,
                                                                             var=self.varnum + self.varchar,
                                                                             p_value_entry=self.par_p_value,
                                                                             p_value_stay=self.par_stay_p_value,
                                                                             add_inditor=self.par_inditor_help,
                                                                             intercept=self.par_intercept_flag,
                                                                             criterion=self.par_criterion,
                                                                             df=self.IGN_grouped_train_data,
                                                                             response=self.target_train,
                                                                             direction=self.par_direction,
                                                                             show_step=True, apply_restrict=True,
                                                                             n_job=self.n_job)
                        elif (self.par_variable_type == 'GRP') and (
                                (self.par_use_freezing_flag == '否') | (self.model_ppp == [])):
                            self.model_ppp = lrmodel.grp_logistic_regression(mianframe=self.start_window_base,
                                                                             var=self.vari_list,
                                                                             p_value_entry=self.par_p_value,
                                                                             p_value_stay=self.par_stay_p_value,
                                                                             intercept=self.par_intercept_flag,
                                                                             criterion=self.par_criterion,
                                                                             df=self.IGN_grouped_train_data,
                                                                             response=self.target_train,
                                                                             direction=self.par_direction,
                                                                             show_step=True, apply_restrict=True,
                                                                             n_job=self.n_job)
                        elif (self.par_variable_type == 'GRP_ind') and (
                                (self.par_use_freezing_flag == '否') | (self.model_ppp == [])):
                            self.model_ppp = lrmodel.grp_ind_logistic_regression(mianframe=self.start_window_base,
                                                                                 var=self.vari_list,
                                                                                 p_value_entry=self.par_p_value,
                                                                                 p_value_stay=self.par_stay_p_value,
                                                                                 intercept=self.par_intercept_flag,
                                                                                 criterion=self.par_criterion,
                                                                                 df=self.IGN_grouped_train_data,
                                                                                 response=self.target_train,
                                                                                 direction=self.par_direction,
                                                                                 show_step=True, apply_restrict=True,
                                                                                 n_job=self.n_job)
                        else:
                            if self.par_variable_type == 'WOE':

                                woe_model_re = self.model_ppp[1]
                                cof = pd.DataFrame(woe_model_re.params).reset_index().rename(
                                    {'index': 'woe_variable_name', 0: 'coff'}, axis=1)
                                variable_df = self.model_ppp[2]
                                woe_score = pd.merge(variable_df, cof, how='outer', left_on='model_var',
                                                     right_on='woe_variable_name')
                                woe_score['ori_var'][woe_score['woe_variable_name'] == 'const'] = 'const'
                                ori_var = list(
                                    woe_score[(woe_score['var_type'] == 'ori') & (woe_score['coff'].isnull() == False)][
                                        'ori_var'])
                                try:
                                    ori_var.remove('const')
                                except:
                                    pass
                                ori_var = ['woe_' + x for x in ori_var]
                                woe_var=ori_var
                                group_variable =[]
                                if len(woe_score[(woe_score['var_type'] == 'add') & (
                                        woe_score['coff'].isnull() == False)]) > 0:
                                    add_variabile_df = woe_score[
                                        (woe_score['var_type'] == 'add') & (woe_score['coff'].isnull() == False)]
                                    add_variabile_df['group_a'] = add_variabile_df['group_a'].astype('int')
                                    add_variabile_df['group_b'] = add_variabile_df['group_b'].astype('int')
                                    add_variabile_df['f_group_variable']=add_variabile_df.apply(lambda x: 'ind_f_group_%s_%s_f_group_%s_%s' % (x['vara'], int(x['group_a']), x['varb'], int(x['group_b'])),axis=1)
                                    ori_var = ori_var + ['f_group_' + x for x in add_variabile_df['vara']] + [
                                        'f_group_' + x for x in add_variabile_df['varb']]
                                    group_variable=list(add_variabile_df['f_group_variable'])
                                if list(set(ori_var) - set(self.IGN_grouped_train_data.columns)) == []:
                                    df = self.IGN_grouped_train_data.copy()
                                    if len(woe_score[(woe_score['var_type'] == 'add') & (
                                            woe_score['coff'].isnull() == False)]) > 0:
                                        def add_indictor(vara, varb, groupa, df, groupb):
                                            df['ind_f_group_%s_%s_f_group_%s_%s' % (
                                                vara, int(groupa), varb, int(groupb))] = df.apply(
                                                lambda x: 1 if (x['f_group_%s' % vara] == groupa) & (
                                                        x['f_group_%s' % varb] == groupb) else 0, axis=1)
                                            df['ind_f_group_%s_%s_f_group_%s_%s' % (
                                            vara, int(groupa), varb, int(groupb))] = df[
                                                'ind_f_group_%s_%s_f_group_%s_%s' % (
                                                    vara, int(groupa), varb, int(groupb))].astype('int8')

                                        add_variabile_df.apply(
                                            lambda x: add_indictor(df=df, vara=x['vara'], varb=x['varb'],
                                                                   groupa=x['group_a'], groupb=x['group_b']), axis=1)

                                    model_ppp_select = lrmodel.woe_logistic_regression(mianframe=self.start_window_base,
                                                                                       inditor_pct=self.par_inditor_pct,
                                                                                       inditor_sample=self.par_inditor_sample,
                                                                                       var=list(set(woe_var+group_variable)),
                                                                                       p_value_entry=self.par_p_value,
                                                                                       p_value_stay=self.par_stay_p_value,
                                                                                       add_inditor=self.par_inditor_help,
                                                                                       intercept=self.par_intercept_flag,
                                                                                       criterion=self.par_criterion,
                                                                                       df=df,
                                                                                       response=self.target_train,
                                                                                       direction='NO',
                                                                                       show_step=True,
                                                                                       apply_restrict=False,
                                                                                       flag_IGN=False,
                                                                                       n_job=self.n_job)
                                    modelvar_match_df = pd.DataFrame()
                                    woe_varlist = ['woe_' + x for x in self.varnum + self.varchar]
                                    modelvar_match_df['ori_var'] = self.varnum + self.varchar
                                    modelvar_match_df['model_var'] = woe_varlist
                                    modelvar_match_df['var_type'] = 'ori'
                                    try:
                                        add_df=self.model_ppp[2].copy()
                                        modelvar_match_df = modelvar_match_df.append(add_df[add_df['var_type'] == 'add'])

                                    except Exception as e:
                                        pass
                                    self.model_ppp = [model_ppp_select[0], model_ppp_select[1], modelvar_match_df]
                                else:
                                    tk.messagebox.showwarning('错误', "训练集中没有如下变量%" % (
                                        list(set(ori_var) - set(self.IGN_grouped_train_data.columns))))

                            else:
                                grp_ppp = self.model_ppp
                                grp_model = grp_ppp[1]
                                cof = pd.DataFrame(grp_model.params).reset_index().rename(
                                    {'index': 'grp_variable_name', 0: 'coff'}, axis=1)
                                variable_df = grp_ppp[2]
                                grp_score = pd.merge(variable_df, cof, how='outer', left_on='model_var',
                                                     right_on='grp_variable_name')
                                grp_score['ori_var'][grp_score['grp_variable_name'] == 'const'] = 'const'
                                ori_var = list(
                                    grp_score[(grp_score['var_type'] == 'ori') & (grp_score['coff'].isnull() == False)][
                                        'ori_var'])
                                if list(set(ori_var) - set(self.IGN_grouped_train_data.columns)) == []:
                                    if self.par_variable_type == 'GRP':
                                        self.model_ppp = lrmodel.grp_logistic_regression(
                                            mianframe=self.start_window_base,
                                            var=list(set(ori_var)),
                                            p_value_entry=self.par_p_value,
                                            p_value_stay=self.par_stay_p_value,
                                            intercept=self.par_intercept_flag,
                                            criterion=self.par_criterion,
                                            df=self.IGN_grouped_train_data,
                                            response=self.target_train,
                                            direction=self.par_direction,
                                            show_step=True, apply_restrict=False,
                                            n_job=1)
                                    else:
                                        self.model_ppp = lrmodel.grp_ind_logistic_regression(
                                            mianframe=self.start_window_base,
                                            var=list(set(ori_var)),
                                            p_value_entry=self.par_p_value,
                                            p_value_stay=self.par_stay_p_value,
                                            intercept=self.par_intercept_flag,
                                            criterion=self.par_criterion,
                                            df=self.IGN_grouped_train_data,
                                            response=self.target_train,
                                            direction=self.par_direction,
                                            show_step=True,
                                            apply_restrict=False,
                                            n_job=1)
                                else:
                                    tk.messagebox.showwarning('错误', "训练集中没有如下变量%" % (
                                        list(set(ori_var) - set(self.IGN_grouped_train_data.columns))))
                self.f_scorecard = self.scorecard_data_pre(self.model_ppp)
                if self.lasso_flag == '是':
                    self.lasso_df = self.func_lasso_df(variable_list=self.vari_list, train_target=self.target_train,
                                                       predict_train_data=self.predict_train_data,
                                                       predict_vaild_data=self.predict_vaild_data,
                                                       n_job=self.n_job)
                else:
                    self.lasso_df = pd.DataFrame()
                self.var_clus = self.func_var_clus(variable_list=self.vari_list,
                                                   predict_train_data=self.predict_train_data,
                                                   scorecarddf=self.f_scorecard)
                node_save_path = self.project_path + '/' + '%s.model' % self.node_name
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                self.node_setting = {'node_type': 'SCR',
                                     'node_name': self.node_name,
                                     'node_save_path': node_save_path,
                                     # 'ign_node': self.par_use_freezing_flag,
                                     'par_use_freezing_flag': self.par_use_freezing_flag,
                                     'par_inditor_help': self.par_inditor_help,
                                     'par_import_modelname': self.par_import_modelname,
                                     'par_intercept_flag': self.par_intercept_flag,
                                     'par_p_value': self.par_p_value,
                                     'par_stay_p_value': self.par_stay_p_value,
                                     'par_criterion': self.par_criterion,
                                     'par_direction': self.par_direction,
                                     'par_variable_type': self.par_variable_type,
                                     'par_odds_ratio': self.par_odds_ratio,
                                     'par_odds_score_ratio': self.par_odds_score_ratio,
                                     'par_odds_double_score': self.par_odds_double_score,
                                     'par_intercept_scorecard': self.par_intercept_scorecard,
                                     # 分组过程参数
                                     # 评分卡变量
                                     'predict_train_data': self.predict_train_data,
                                     'predict_vaild_data': self.predict_vaild_data,
                                     'predict_reject_data': self.predict_reject_data,
                                     'predict_oot_data': self.predict_oot_data,
                                     'model': self.model_ppp,
                                     'scorecard_df': self.f_scorecard,
                                     'lasso_df': self.lasso_df,
                                     'lasso_flag': self.lasso_flag,
                                     'time': nowTime,
                                     'previous_node_name': [self.IGN_node_name],
                                     'var_clus': self.var_clus,
                                     'previous_node_time': [self.IGN_node_time],
                                     'IGN_grouping_data': self.IGN_groupingdata,
                                     'report_para': {'train_target': self.target_train,
                                                     'oot_target': self.target_oot,
                                                     'reject_target': self.target_reject,
                                                     'timeid_train': self.timeid_train,
                                                     'timeid_oot': self.timeid_oot,
                                                     'timeid_reject': self.timeid_reject,
                                                     'f_group_report': self.IGN_f_group_report,
                                                     'vari_list': self.vari_list},
                                     # 'data_variable_setting': self.par_traindatavariable_setting,
                                     # 'reject_data_variable_setting': self.par_rejectdatavariable_setting,
                                     # 'oot_data_variable_setting': self.par_ootdatavariable_setting,
                                     'use_node': [self.node_name] + self.IGN_previous_usedlist
                                     }
                self.finsh = 'Y'
                for child in self.master.winfo_children():
                    child.destroy()
                self.adjustsetting()
                self.model_start_flag = 'N'
        # except Exception as e:
        #     tk.messagebox.showwarning('错误', e)
        #     self.model_start_flag = 'N'

    def scorecard_result_show_ui(self, event):
        try:
            if self.result_page.state() == 'normal':
                tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
        except:
            self.result_page = Toplevel(self.master)
            scorecard_result_ui(mainframe=self.result_page, project_path=self.project_path, node_name=self.node_name,
                                predict_train_data=self.predict_train_data, predict_vaild_data=self.predict_vaild_data,
                                predict_oot_data=self.predict_oot_data, predict_reject_data=self.predict_reject_data,
                                train_target=self.target_train, oot_target=self.target_oot,
                                reject_target=self.target_reject,
                                train_time_id=self.timeid_train, oot_time_id=self.timeid_oot,
                                reject_time_id=self.timeid_reject,
                                record_list=self.model_ppp[0], model=self.model_ppp[1], scorecarddf=self.f_scorecard,
                                f_group_report=self.IGN_f_group_report
                                , variable_list=self.vari_list, lasso_df=self.lasso_df,
                                model_var_type=self.par_variable_type, var_clus=self.var_clus)

    def scorecard_data_pre(self, model_re):
        def score_predict(scorecard, df):
            if len(scorecard[scorecard['variable_name']=='const'])==1:
                df['SCORE']=list(scorecard[scorecard['variable_name'] =='const']['scorecard'])[0]
            else:
                df['SCORE'] = 0
            for var in list(scorecard['variable_name'].unique()):
                if var != 'const':
                    df['SCR_%s' % var] = 0
                    for group in scorecard[scorecard['variable_name'] == var]['f_group']:
                        df['SCR_%s' % var][df['f_group_%s' % var] == group] = \
                        list(scorecard[(scorecard['variable_name'] == var) & (scorecard['f_group'] == group)]['scorecard'])[0]
                    df['SCORE'] = df['SCORE'] + df['SCR_%s' % var]
            return df
        if self.par_variable_type == 'WOE':
            # woe评分卡
            def woe_predict(model, intercept, df, woe_score):
                if len(woe_score[(woe_score['var_type'] == 'add') & (woe_score['coff'].isnull() == False)]) > 0:
                    add_variabile_df = woe_score[
                        (woe_score['var_type'] == 'add') & (woe_score['coff'].isnull() == False)]
                    add_variabile_df['group_a'] = add_variabile_df['group_a'].astype('int')
                    add_variabile_df['group_b'] = add_variabile_df['group_b'].astype('int')

                    def add_indictor(vara, varb, groupa, df, groupb):
                        df['ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb))] = df.apply(
                            lambda x: 1 if (x['f_group_%s' % vara] == groupa) & (
                                        x['f_group_%s' % varb] == groupb) else 0, axis=1)
                        df['f_group_ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb))] = df[
                            'ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb))].astype('int8')

                    add_variabile_df.apply(
                        lambda x: add_indictor(df=df, vara=x['vara'], varb=x['varb'], groupa=x['group_a'],
                                               groupb=x['group_b']), axis=1)

                input_list = list(
                    pd.DataFrame(model.params).reset_index().rename({'index': 'woe_variable_name', 0: 'coff'}, axis=1)[
                        'woe_variable_name'])

                try:
                    input_list.remove('const')
                except:
                    pass
                if intercept == True:
                    df['SCORECARD_LR_p_1'] = model.predict(sm.add_constant(df[input_list]))
                else:
                    df['SCORECARD_LR_p_1'] = model.predict(df[input_list])
                return df

            woe_model_re = model_re[1]
            cof = pd.DataFrame(woe_model_re.params).reset_index().rename({'index': 'woe_variable_name', 0: 'coff'},
                                                                         axis=1)
            group_report = self.IGN_f_group_report
            total = group_report.groupby(['variable_name'])['f_N_obs'].sum().reset_index().rename(
                {'f_N_obs': 'total_count'}, axis=1)
            total_bad = group_report.groupby(['variable_name'])['f_N_bad'].sum().reset_index().rename(
                {'f_N_bad': 'total_count_bad'}, axis=1)
            total_num = total['total_count'].max()
            total_bad = total_bad['total_count_bad'].max()
            group_report = pd.merge(group_report, total, how='left', on='variable_name')
            group_report['pct_f_N_obs'] = group_report['f_N_obs'] / group_report['total_count']
            # cof=dd
            variable_df = model_re[2]
            woe_score = pd.merge(variable_df, cof, how='outer', left_on='model_var', right_on='woe_variable_name')
            woe_score['ori_var'][woe_score['woe_variable_name'] == 'const'] = 'const'
            use = woe_score.groupby('ori_var')['coff'].max().reset_index()
            use = list(use[use['coff'].isnull() == False]['ori_var'])
            woe_model_df = woe_score[woe_score['ori_var'].isin(use)].fillna(0)
            woe_model_df = woe_model_df.rename({'group': 'f_group'}, axis=1)
            woe_model_df['variable_name'] = woe_model_df['ori_var']
            # 有辅助变量加入
            scorecard = pd.merge(woe_model_df, group_report, how='left', on=['variable_name'])[
                ['variable_name', 'f_group', 'var_type', 'label', 'f_N_obs', 'f_Bad_rate', 'woe', 'pct_f_N_obs',
                 'coff']]
            if len(woe_score[(woe_score['var_type'] == 'add') & (woe_score['coff'].isnull() == False)]) > 0:
                add_variabile_df = woe_score[(woe_score['var_type'] == 'add') & (woe_score['coff'].isnull() == False)]
                add_variabile_df_0 = add_variabile_df.copy()
                add_variabile_df_0['label'] = add_variabile_df_0.apply(
                    lambda x: 'f_group_%s != %s   or f_group_%s != %s ' % (
                    x['vara'], int(x['group_a']), x['varb'], int(x['group_b'])), axis=1)
                add_variabile_df_0['f_group'] = 0
                add_variabile_df_0['woe'] = 0
                add_variabile_df_0['count'] = total_num - add_variabile_df_0['count']
                add_variabile_df_0['badrate'] = (total_bad - add_variabile_df_0['count'] * add_variabile_df_0[
                    'badrate']) / add_variabile_df_0['count']
                add_variabile_df_0['pct_f_N_obs'] = add_variabile_df_0['count'] / total_num
                add_variabile_df['pct_f_N_obs'] = add_variabile_df['count'] / total_num
                add_variabile_df['label'] = add_variabile_df.apply(lambda x: 'f_group_%s = %s  and f_group_%s = %s ' % (
                x['vara'], int(x['group_a']), x['varb'], int(x['group_b'])), axis=1)
                add_variabile_df['f_group'] = 1
                add_variabile_df['woe'] = 1
                add_variabile_df = add_variabile_df.rename(
                    {'model_var': 'variable_name', 'badrate': 'f_Bad_rate', 'count': 'f_N_obs'}, axis=1)
                add_variabile_df_0 = add_variabile_df_0.rename(
                    {'model_var': 'variable_name', 'badrate': 'f_Bad_rate', 'count': 'f_N_obs'}, axis=1)
                add_scorecard = add_variabile_df.append(add_variabile_df_0)[
                    ['variable_name', 'f_group', 'label', 'f_N_obs', 'var_type', 'f_Bad_rate', 'woe', 'pct_f_N_obs',
                     'coff']]
                add_scorecard['f_Bad_rate'] = add_scorecard.apply(lambda x: "%.2f%%" % (x['f_Bad_rate'] * 100), axis=1)
                scorecard = scorecard.append(add_scorecard)
            B = self.par_odds_double_score / math.log(2)
            A = self.par_odds_score_ratio - B * math.log(self.par_odds_ratio)
            if self.par_intercept_flag == False:
                add_df=pd.DataFrame([{'variable_name':'const','coff':0}])
                scorecard=scorecard.append(add_df)
            scorecard['SCORE'] = scorecard.apply(
                lambda x: A - B * x['coff'] if x['variable_name'] == 'const' else -B * x['coff'] * x['woe'], axis=1)
            score_adjust = scorecard.groupby('variable_name')['SCORE'].min().reset_index().rename(
                {'SCORE': 'score_min'}, axis=1)
            adjust_num = score_adjust[score_adjust['score_min'] < 0]['score_min'].sum()
            score_adjust['score_min'][score_adjust['variable_name'] == 'const'] = -adjust_num
            f_scorecard = pd.merge(scorecard, score_adjust, how='left', on='variable_name')
            f_scorecard['scorecard'] = f_scorecard['SCORE'] - f_scorecard['score_min']
            f_scorecard['scorecard'] = f_scorecard['scorecard'].apply(lambda x: int(x))
            f_scorecard['coff'] = round(f_scorecard['coff'], 4)
            f_scorecard = f_scorecard.fillna(0)
            f_scorecard['f_N_obs'] = f_scorecard['f_N_obs'].astype('int')
            f_scorecard['pct_f_N_obs'] = f_scorecard.apply(lambda x: "%.2f%%" % (x['pct_f_N_obs'] * 100), axis=1)
            f_scorecard = f_scorecard.sort_values(by=['variable_name', 'f_group'])
            f_scorecard = f_scorecard[f_scorecard['variable_name'] == 'const'].append(f_scorecard[f_scorecard['variable_name'] != 'const'])
            # 给数据集打分
            self.predict_train_data = woe_predict(model=woe_model_re, intercept=self.par_intercept_flag,
                                                  df=self.IGN_grouped_train_data, woe_score=woe_score)
            self.predict_train_data =score_predict(f_scorecard, self.predict_train_data )
            if self.IGN_grouped_valid_data.empty == False:
                self.predict_vaild_data = woe_predict(model=woe_model_re, intercept=self.par_intercept_flag,
                                                      df=self.IGN_grouped_valid_data, woe_score=woe_score)
                self.predict_vaild_data = score_predict(f_scorecard, self.predict_vaild_data)
            else:
                self.predict_vaild_data = pd.DataFrame()
            if self.IGN_grouped_reject_data.empty == False:
                woe_vari_list = list(
                    woe_score[(woe_score['coff'].isnull() == False) & (woe_score['var_type'] == 'ori')]['model_var'])
                try:
                    add_variable_a = list(
                        woe_score[(woe_score['coff'].isnull() == False) & (woe_score['var_type'] == 'add')]['vara'])
                    add_variable_b = list(
                        woe_score[(woe_score['coff'].isnull() == False) & (woe_score['var_type'] == 'add')]['varb'])
                    add_variable_list = list(set(['f_group_' + x for x in add_variable_b + add_variable_a]))
                except:
                    add_variable_list = []
                datacol = list(self.IGN_grouped_reject_data.columns)
                not_exist = []
                for va in add_variable_list + woe_vari_list:
                    if (va in datacol) == False:
                        not_exist.append(va)
                if len(not_exist) > 0:
                    tip = Toplevel(self.master)
                    tip.title('警告：')
                    lb = Label(tip, text='下面变量没有在拒绝样本中\n找到%s' % not_exist)
                    lb.pack()
                else:
                    self.predict_reject_data = woe_predict(model=woe_model_re, intercept=self.par_intercept_flag,
                                                           df=self.IGN_grouped_reject_data, woe_score=woe_score)
                    self.predict_reject_data = score_predict(f_scorecard, self.predict_reject_data)
            else:
                self.predict_reject_data = pd.DataFrame()
            if self.IGN_grouped_oot_data.empty == False:
                woe_vari_list = list(
                    woe_score[(woe_score['coff'].isnull() == False) & (woe_score['var_type'] == 'ori')]['model_var'])
                try:
                    add_variable_a = list(
                        woe_score[(woe_score['coff'].isnull() == False) & (woe_score['var_type'] == 'add')]['vara'])
                    add_variable_b = list(
                        woe_score[(woe_score['coff'].isnull() == False) & (woe_score['var_type'] == 'add')]['varb'])
                    add_variable_list = list(set(['f_group_' + x for x in add_variable_b + add_variable_a]))
                except:
                    add_variable_list = []
                datacol = list(self.IGN_grouped_oot_data.columns)
                not_exist = []
                for va in add_variable_list + woe_vari_list:
                    if (va in datacol) == False:
                        not_exist.append(va)
                if len(not_exist) > 0:
                    tip1 = Toplevel(self.master)
                    tip1.title('警告：')
                    lb = Label(tip1, text='下面变量没有在OOT样本中\n找到%s' % not_exist)
                    lb.pack()
                else:
                    self.predict_oot_data = woe_predict(model=woe_model_re, intercept=self.par_intercept_flag,
                                                        df=self.IGN_grouped_oot_data, woe_score=woe_score)
                    self.predict_oot_data = score_predict(f_scorecard, self.predict_oot_data)
            else:
                self.predict_oot_data = pd.DataFrame()
        else:
            # group 评分卡
            grp_ppp = model_re
            grp_model = grp_ppp[1]
            cof = pd.DataFrame(grp_model.params).reset_index().rename({'index': 'grp_variable_name', 0: 'coff'}, axis=1)
            group_report = self.IGN_f_group_report
            variable_df = grp_ppp[2]
            total = group_report.groupby(['variable_name'])['f_N_obs'].sum().reset_index().rename(
                {'f_N_obs': 'total_count'}, axis=1)
            group_report = pd.merge(group_report, total, how='left', on='variable_name')
            group_report['pct_f_N_obs'] = group_report['f_N_obs'] / group_report['total_count']
            grp_score = pd.merge(variable_df, cof, how='outer', left_on='model_var', right_on='grp_variable_name')
            grp_score['variable'][grp_score['grp_variable_name'] == 'const'] = 'const'
            use = grp_score.groupby('variable')['coff'].max().reset_index()
            use = list(use[use['coff'].isnull() == False]['variable'])
            grp_model_df = grp_score[grp_score['variable'].isin(use)].fillna(0)
            grp_model_df = grp_model_df.rename({'group': 'f_group'}, axis=1)
            grp_model_df['variable_name'] = grp_model_df['variable'].apply(lambda x: 'const' if x == 'const' else x[8:])
            scorecard = pd.merge(grp_model_df, group_report, how='left', on=['variable_name', 'f_group'])[
                ['variable_name', 'f_group', 'var_type', 'f_N_obs', 'label', 'f_Bad_rate', 'pct_f_N_obs', 'coff',
                 'woe']]
            if self.par_intercept_flag == False:
                add_df=pd.DataFrame([{'variable_name':'const','coff':0}])
                scorecard=scorecard.append(add_df)
            B = self.par_odds_double_score / math.log(2)
            A = self.par_odds_score_ratio + B * math.log(self.par_odds_ratio)
            scorecard['SCORE'] = scorecard.apply(
                lambda x: A - B * x['coff'] if x['variable_name'] == 'const' else -B * x['coff'], axis=1)
            score_adjust = scorecard.groupby('variable_name')['SCORE'].min().reset_index().rename(
                {'SCORE': 'score_min'}, axis=1)
            adjust_num = score_adjust[score_adjust['score_min'] < 0]['score_min'].sum()
            score_adjust['score_min'][score_adjust['variable_name'] == 'const'] = -adjust_num
            f_scorecard = pd.merge(scorecard, score_adjust, how='left', on='variable_name')
            f_scorecard['scorecard'] = f_scorecard['SCORE'] - f_scorecard['score_min']
            f_scorecard['scorecard'] = f_scorecard['scorecard'].apply(lambda x: int(x))
            f_scorecard['coff'] = round(f_scorecard['coff'], 4)
            f_scorecard = f_scorecard.fillna(0)
            f_scorecard['f_N_obs'] = f_scorecard['f_N_obs'].astype('int')
            f_scorecard['pct_f_N_obs'] = f_scorecard.apply(lambda x: "%.2f%%" % (x['pct_f_N_obs'] * 100), axis=1)
            f_scorecard = f_scorecard.sort_values(by=['variable_name', 'f_group'])
            f_scorecard = f_scorecard[f_scorecard['variable_name'] == 'const'].append(
                f_scorecard[f_scorecard['variable_name'] != 'const'])

            # 给数据集打分
            def grp_predict(model, intercept, df):
                input_list = list(
                    pd.DataFrame(model.params).reset_index().rename({'index': 'grp_variable_name', 0: 'coff'}, axis=1)[
                        'grp_variable_name'])
                try:
                    input_list.remove('const')
                except:
                    pass
                if intercept == True:
                    df['SCORECARD_LR_p_1'] = model.predict(sm.add_constant(df[input_list]))
                else:
                    df['SCORECARD_LR_p_1'] = model.predict(df[input_list])
                return df

            def group_data_pre(df, f_scorecard):
                for varable in list(set(f_scorecard[f_scorecard['variable_name'] != 'const']['variable_name'])):
                    grouplist = list(set(f_scorecard[f_scorecard['variable_name'] == varable]['f_group']))
                    for value in grouplist:
                        df['f_group_%s_%s' % (varable, int(value))] = df['f_group_%s' % varable].apply(
                            lambda x: 1 if x == int(value) else 0)
                        df['f_group_%s_%s' % (varable, int(value))] = df[
                            'f_group_%s_%s' % (varable, int(value))].astype('int8')
                return df

            self.predict_train_data = grp_predict(model=grp_model, intercept=self.par_intercept_flag,
                                                  df=group_data_pre(self.IGN_grouped_train_data, f_scorecard))
            self.predict_train_data = score_predict(f_scorecard, self.predict_train_data)
            if self.IGN_grouped_valid_data.empty == False:
                self.predict_vaild_data = grp_predict(model=grp_model, intercept=self.par_intercept_flag,
                                                      df=group_data_pre(self.IGN_grouped_valid_data, f_scorecard))
            if self.IGN_grouped_reject_data.empty == False:
                grp_vari_list = list(set(
                    grp_score[(grp_score['coff'].isnull() == False) & (grp_score['var_type'] == 'ori')]['variable']))
                datacol = list(self.IGN_grouped_reject_data.columns)
                not_exist = []
                for va in grp_vari_list:
                    if (va in datacol) == False:
                        not_exist.append(va)
                if len(not_exist) > 0:
                    tip2 = Toplevel(self.master)
                    tip2.title('警告：')
                    self.text = StringVar()
                    self.label_list = '下面变量没有在拒绝样本中找到%s' % not_exist
                    lb = Label(tip2, textvariable=self.text)
                    lb.pack()
                else:
                    self.predict_reject_data = grp_predict(model=grp_model, intercept=self.par_intercept_flag,
                                                           df=group_data_pre(self.IGN_grouped_reject_data, f_scorecard))
                    self.predict_reject_data = score_predict(f_scorecard, self.predict_reject_data)
            if self.IGN_grouped_oot_data.empty == False:
                grp_vari_list = list(
                    set(grp_score[(grp_score['coff'].isnull() == False) & (grp_score['var_type'] == 'ori')][
                            'variable']))
                datacol = list(self.IGN_grouped_oot_data.columns)
                not_exist = []
                for va in grp_vari_list:
                    if (va in datacol) == False:
                        not_exist.append(va)
                if len(not_exist) > 0:
                    tip3 = Toplevel(self.master)
                    tip3.title('警告：')
                    self.text = StringVar()
                    self.label_list = '下面变量没有在OOT样本中找到%s' % not_exist
                    lb = Label(tip3, textvariable=self.text)
                    lb.pack()
                else:
                    self.predict_oot_data = grp_predict(model=grp_model, intercept=self.par_intercept_flag,
                                                        df=group_data_pre(self.IGN_grouped_oot_data, f_scorecard))
                    self.predict_oot_data = score_predict(f_scorecard, self.predict_oot_data)
        return f_scorecard

    def func_lasso_df(self, variable_list, train_target, predict_train_data, predict_vaild_data, n_job):
        woe_list = ['woe_' + x for x in variable_list]
        x_train = predict_train_data[woe_list]
        y_train = predict_train_data[train_target]
        tlist = [num / 100 for num in range(1, 100, 1)] + [num / 5000 for num in range(1, 50, 1)] + [num for num in
                                                                                                     range(1, 20, 1)]
        random.shuffle(tlist)
        lent = math.ceil(len(tlist) / n_job)

        def func(num):
            # summ = pd.DataFrame()
            summ = []
            for l in range((num - 1) * lent, min(len(tlist), num * lent)):
                p_c = tlist[l]
                model = LogisticRegression(penalty='l1', C=p_c, solver='liblinear')
                h = model.fit(x_train, y_train)
                temp = predict_train_data
                temp['lasso_p_pro'] = pd.DataFrame(h.predict_proba(x_train))[1]
                temp['llr'] = np.log(temp['lasso_p_pro']) * temp[train_target] + np.log(1 - temp['lasso_p_pro']) * (
                            1 - temp[train_target])
                llr = temp['llr'].sum()

                var_num = []
                for cof in h.coef_[0]:
                    if cof != 0:
                        var_num.append(cof)
                k = len(var_num)
                num = list(h.coef_[0])
                var = woe_list
                # add_ = pd.DataFrame(dict(zip(var, num)), index=[1])
                add_ = dict(zip(var, num))
                add_['C'] = p_c
                add_['k'] = k
                add_['llr'] = llr

                try:
                    if predict_vaild_data.empty == False:
                        temp_v = predict_vaild_data
                        x_train_v = predict_vaild_data[woe_list]
                        temp_v['lasso_p_pro'] = pd.DataFrame(h.predict_proba(x_train_v))[1]
                        temp_v['llr'] = np.log(temp_v['lasso_p_pro']) * temp_v[train_target] + np.log(
                            1 - temp_v['lasso_p_pro']) * (1 - temp_v[train_target])
                        llr_v = temp_v['llr'].sum()
                        add_['llr_v'] = llr_v
                except:
                    pass

                # summ = summ.append(add_)
                summ.append(add_)

            summ_d = pd.DataFrame(summ)
            return summ_d

        scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(
            delayed(func)(num) for num in range(1, 1 + n_job))
        score_df = pd.DataFrame()
        for tt in scores_with_candidates:
            sc = pd.DataFrame(tt)
            score_df = score_df.append(sc)
        score_df['aic'] = score_df.apply(lambda x: 2 * x['k'] - 2 * x['llr'], axis=1)
        score_df['bic'] = score_df.apply(lambda x: math.log(len(predict_train_data)) * x['k'] - 2 * x['llr'],
                                         axis=1)
        try:
            score_df['aic_v'] = score_df.apply(lambda x: 2 * x['k'] - 2 * x['llr_v'], axis=1)
            score_df['bic_v'] = score_df.apply(lambda x: math.log(len(predict_train_data)) * x['k'] - 2 * x['llr_v'],
                                               axis=1)
        except Exception as e:
            pass
        return score_df

    def func_var_clus(self, variable_list, predict_train_data, scorecarddf):
        variable_list = ['woe_' + x for x in variable_list]
        df = predict_train_data[variable_list]
        clus = VarClus()
        clus.decompose(dataframe=df)
        model_list = ['woe_' + x for x in list(set(scorecarddf['variable_name']))]
        h = clus.print_cluster_structure(model_variable=model_list, h_space=5)
        h = '算法来自https://github.com/jingmin1987/variable-clustering \n ****为入模变量 ---为未如模变量 \n\n' + h
        return h

    def reult_show_only(self, result_page):
        scorecard_result_ui(mainframe=result_page, project_path=self.project_path, node_name=self.node_name,
                            predict_train_data=self.predict_train_data, predict_vaild_data=self.predict_vaild_data,
                            predict_oot_data=self.predict_oot_data, predict_reject_data=self.predict_reject_data,
                            train_target=self.target_train, oot_target=self.target_oot,
                            reject_target=self.target_reject,
                            train_time_id=self.timeid_train, oot_time_id=self.timeid_oot,
                            reject_time_id=self.timeid_reject,
                            record_list=self.model_ppp[0], model=self.model_ppp[1], scorecarddf=self.f_scorecard,
                            f_group_report=self.IGN_f_group_report
                            , variable_list=self.vari_list, lasso_df=self.lasso_df,
                            model_var_type=self.par_variable_type, var_clus=self.var_clus)

    def add_delet_var(self, record_list, input_model, model_variable_df, modify_var, flag, par_variable_type, var_list,
                      n_job, predict_train_data, target_train, predict_vaild_data, par_intercept_flag):
        error2_f = Toplevel(self.master)
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()
        error2_f.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
        L2 = Label(error2_f, text="计算中。。。")
        L2.grid()
        self.master.update()
        if par_variable_type == 'WOE':
            cof = pd.DataFrame(input_model.params).reset_index().rename({'index': 'woe_variable_name', 0: 'coff'},
                                                                        axis=1)
            all_list = list(set(['woe_' + x for x in var_list]) | set(cof['woe_variable_name']))
            if flag == 'add':

                selected_list = list(cof['woe_variable_name']) + [modify_var]
            elif flag == 'del':

                selected_list = list(set(cof['woe_variable_name']) - set([modify_var]))

            else:

                selected_list = list(set(cof['woe_variable_name']))
            try:
                selected_list.remove('const')
            except:
                pass
            try:
                all_list.remove('const')
            except:
                pass
            lent = math.ceil(len(all_list) / n_job)

            def func(num):
                # summ = pd.DataFrame()
                summ = []
                for l in range((num - 1) * lent, min(len(all_list), num * lent)):
                    candidate = all_list[l]
                    # for candidate in all_list:
                    if candidate in selected_list:
                        model_list = list(set(selected_list) - set([candidate]))
                    else:
                        model_list = list(set(selected_list + [candidate]))
                    if par_intercept_flag:  # 是否有截距
                        logit_mod = sm.Logit(predict_train_data[target_train],
                                             sm.add_constant(predict_train_data[model_list]))
                    else:
                        logit_mod = sm.Logit(predict_train_data[target_train], predict_train_data[model_list])
                    result = logit_mod.fit(method='lbfgs', maxiter=100)
                    pvalue = max(result.pvalues)
                    fpr_t, tpr_t, threshold_t = roc_curve(predict_train_data[target_train],
                                                          result.predict())  ###计算真正率和假正率
                    roc_auc_t = auc(fpr_t, tpr_t)  ###计算auc的值
                    if predict_vaild_data.empty == False:
                        if par_intercept_flag:
                            pre_v = result.predict(sm.add_constant(predict_vaild_data[model_list]))
                        else:
                            pre_v = result.predict(predict_vaild_data[model_list])
                        fpr_v, tpr_v, threshold_v = roc_curve(predict_vaild_data[target_train], pre_v)  ###计算真正率和假正率
                        roc_auc_v = auc(fpr_v, tpr_v)  ###计算auc的值
                    else:
                        roc_auc_v = None
                    summ.append({'var': candidate, 'pvalue': pvalue, 'auc_t': roc_auc_t, 'auc_v': roc_auc_v})
                summ_df = pd.DataFrame(summ)
                return summ_df

            scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(
                delayed(func)(num) for num in range(1, 1 + n_job))
            score_df = pd.DataFrame()
            for tt in scores_with_candidates:
                sc = pd.DataFrame(tt)
                score_df = score_df.append(sc)
            # 现在模型
            if par_intercept_flag:  # 是否有截距
                logit_mod = sm.Logit(predict_train_data[target_train],
                                     sm.add_constant(predict_train_data[selected_list]))
            else:
                logit_mod = sm.Logit(predict_train_data[target_train], predict_train_data[selected_list])
            result = logit_mod.fit(method='lbfgs', maxiter=100)
            var = 'current_model'
            pvalue = max(result.pvalues)
            fpr_t, tpr_t, threshold_t = roc_curve(predict_train_data[target_train], result.predict())  ###计算真正率和假正率
            roc_auc_t = auc(fpr_t, tpr_t)  ###计算auc的值
            if predict_vaild_data.empty == False:
                if par_intercept_flag:
                    pre_v = result.predict(sm.add_constant(predict_vaild_data[selected_list]))
                else:
                    pre_v = result.predict(predict_vaild_data[selected_list])
                fpr_v, tpr_v, threshold_v = roc_curve(predict_vaild_data[target_train], pre_v)  ###计算真正率和假正率
                roc_auc_v = auc(fpr_v, tpr_v)  ###计算auc的值
            else:
                roc_auc_v = None
            current = pd.DataFrame([{'var': var, 'pvalue': pvalue, 'auc_t': roc_auc_t, 'auc_v': roc_auc_v}])

            score_df = score_df.append(current)

            score_df['use_or'] = score_df['var'].apply(lambda x: 'Y' if x in selected_list else 'N')

        else:
            # GRP
            group_varlist = ['f_group_' + x for x in var_list]
            df = predict_train_data.copy()
            df_v = predict_vaild_data
            for varable in group_varlist:
                mm = len(df[varable].unique())
                grouplist = list(
                    df[varable].groupby(df[varable]).agg({'count'}).reset_index().sort_values(by='count')[varable][
                    0:mm - 1])
                for value in grouplist:
                    df['%s_%s' % (varable, int(value))] = df[varable].apply(lambda x: 1 if x == value else 0)
                    df['%s_%s' % (varable, int(value))] = df['%s_%s' % (varable, int(value))].astype('int8')
                    if predict_vaild_data.empty == False:
                        df_v['%s_%s' % (varable, int(value))] = df_v[varable].apply(lambda x: 1 if x == value else 0)
                        df_v['%s_%s' % (varable, int(value))] = df_v['%s_%s' % (varable, int(value))].astype('int8')
            grp_model = input_model
            cof = pd.DataFrame(grp_model.params).reset_index().rename({'index': 'grp_variable_name', 0: 'coff'}, axis=1)
            variable_df = model_variable_df
            grp_score = pd.merge(variable_df, cof, how='outer', left_on='model_var', right_on='grp_variable_name')
            grp_score['variable'][grp_score['grp_variable_name'] == 'const'] = 'const'
            use = grp_score.groupby('variable')['coff'].max().reset_index()
            use_t = list(use[use['coff'].isnull() == False]['variable'])
            if flag == 'add':
                use = use_t + [modify_var]
            elif flag == 'del':
                use = list(set(use_t) - set([modify_var]))
            else:
                use = use_t
            try:
                use.remove(('const'))
            except:
                pass
            selected = ['f_group_' + x for x in use]
            result_list = []

            if selected != []:
                group_variable_select = variable_df[variable_df['variable'].isin(selected)]['model_var'].sum()
            else:
                group_variable_select = []
            for candidate in group_varlist:
                group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['model_var'].sum()
                if candidate in selected:
                    model_list = list(set(group_variable_select) - set(group_variable_candidate))
                else:
                    model_list = group_variable_select + group_variable_candidate
                if par_intercept_flag:  # 是否有截距
                    logit_mod = sm.Logit(df[target_train], sm.add_constant(df[model_list]))
                else:
                    logit_mod = sm.Logit(df[target_train], df[model_list])
                result = logit_mod.fit(method='lbfgs', maxiter=100)
                var = candidate
                pvalue = max(result.pvalues)
                fpr_t, tpr_t, threshold_t = roc_curve(predict_train_data[target_train], result.predict())  ###计算真正率和假正率
                roc_auc_t = auc(fpr_t, tpr_t)  ###计算auc的值
                if predict_vaild_data.empty == False:
                    if par_intercept_flag:
                        pre_v = result.predict(sm.add_constant(df_v[model_list]))
                    else:
                        pre_v = result.predict(df_v[model_list])
                    fpr_v, tpr_v, threshold_v = roc_curve(predict_vaild_data[target_train], pre_v)  ###计算真正率和假正率
                    roc_auc_v = auc(fpr_v, tpr_v)  ###计算auc的值
                else:
                    roc_auc_v = None
                result_list.append({'var': var, 'pvalue': pvalue, 'auc_t': roc_auc_t, 'auc_v': roc_auc_v})
            if par_intercept_flag:  # 是否有截距
                logit_mod = sm.Logit(df[target_train], sm.add_constant(df[group_variable_select]))
            else:
                logit_mod = sm.Logit(df[target_train], df[group_variable_select])
            result = logit_mod.fit(method='lbfgs', maxiter=100)
            var = 'current_model'
            pvalue = max(result.pvalues)
            fpr_t, tpr_t, threshold_t = roc_curve(predict_train_data[target_train], result.predict())  ###计算真正率和假正率
            roc_auc_t = auc(fpr_t, tpr_t)  ###计算auc的值
            if predict_vaild_data.empty == False:
                if par_intercept_flag:
                    pre_v = result.predict(sm.add_constant(df_v[group_variable_select]))
                else:
                    pre_v = result.predict(df_v[group_variable_select])
                fpr_v, tpr_v, threshold_v = roc_curve(predict_vaild_data[target_train], pre_v)  ###计算真正率和假正率
                roc_auc_v = auc(fpr_v, tpr_v)  ###计算auc的值
            else:
                roc_auc_v = None
            result_list.append({'var': var, 'pvalue': pvalue, 'auc_t': roc_auc_t, 'auc_v': roc_auc_v})
            score_df = pd.DataFrame(result_list)
            score_df['use_or'] = score_df['var'].apply(lambda x: 'Y' if x in ['f_group_' + x for x in use] else 'N')
        score_df = score_df.rename(
            {'var': '变量名称', 'pvalue': '调整后模型最大p值', 'auc_t': '调整后训练集auc', 'auc_v': '调整后验证集auc', 'use_or': '是否在模型中'},
            axis=1)
        score_df = score_df[['变量名称', '调整后训练集auc', '调整后验证集auc', '调整后模型最大p值', '是否在模型中']]
        score_df = score_df.sort_values(by=['是否在模型中', '调整后训练集auc'], ascending=[False, False])
        score_df = score_df[score_df['变量名称'] == 'current_model'].append(score_df[score_df['变量名称'] != 'current_model'])
        try:
            error2_f.destroy()
        except:
            pass
        return score_df, record_list, result, model_variable_df
