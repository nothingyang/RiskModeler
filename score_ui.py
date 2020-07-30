import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd

import pickle as pickle
from .func import binning
import math
from .model import lrmodel

lrmodel = lrmodel()
import datetime

binning = binning()
import statsmodels.api as sm

from .score_result_ui import score_result_ui

import os



class scoreing():
    def __init__(self, mianframe, project_info):
        self.master = mianframe
        # project参数
        self.save = 'N'
        self.node_type = 'scoring'
        self.project_info = project_info
        self.project_path = os.path.split(project_info[project_info['模块类型'] == 'project']['保存地址'][0])[0]
        self.node_name = 'Score'
        self.exist_data = list(project_info['模块名字'])
        self.load = 'N'
        self.finsh = 'N'
        # IGN参数
        self.IGN_f_group_report = pd.DataFrame()
        self.IGN_grouped_train_data = pd.DataFrame()
        self.predict_score_data=pd.DataFrame()
        self.IGN_par_traindatavariable_setting = None
        self.IGN_node_time = None

        self.timeid_train = None
        self.target_train = None
        self.timeid_score = None
        self.target_score=None
        # 模型参数
        self.par_intercept_flag = True
        self.par_variable_type = 'WOE'
        self.model_start_flag = 'N'

        # 分组过程参数
        # 评分卡变量
        self.predict_score_data=pd.DataFrame()
        self.predict_train_data = pd.DataFrame()
        self.IGN_grouping_data=pd.DataFrame()
        self.model_ppp = []
        self.scorecard_df = pd.DataFrame()
        self.lasso_df = pd.DataFrame()
        self.par_score_data=pd.DataFrame()
        self.pre_data()
        self.pre_model()

    # 模块参数
    def pre_data(self):
        dd = list(self.project_info[self.project_info['模块类型'] == 'DATA']['保存地址'])
        self.scoredf_list = []
        for add in dd:
            try:
                fr = open(add, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                data_role = node_info[0]['data_role']
                node_name = node_info[0]['node_name']
                if data_role == 'Score':
                    self.scoredf_list.append(node_name)
            except Exception as e:
                tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (add, e))
    def pre_model(self):
        try:
            dd = list(self.project_info[(self.project_info['模块类型'] == 'SCR') &(self.project_info['状态'] == 'Good')]['模块名字'])
            self.SCR_list = dd
        except Exception as e:
            tk.messagebox.showwarning('错误', e)
    def load_node(self,node_data,ac):
        print('ccc')
        self.finsh = 'Y'
        self.node_setting=node_data[0]
        self.predict_score_data=node_data[1]
        self.node_model_data=node_data[2]
        self.node_score_data=node_data[3]

        self.node_model_name=self.node_setting['model_nodename']
        self.par_score_dataname=self.node_setting['model_dataname']
        self.par_score_data=self.node_setting['score_data_node']
        self.node_save_path=self.node_setting['node_save_path']
        previous_node_name=self.node_setting['previous_node_name']
        previous_node_time=self.node_setting['previous_node_time']
        self.node_name=self.node_setting['node_name']

        self.par_scoredatavariable_setting = self.node_score_data[0]['data_variable_setting']
        self.par_score_dataname = self.node_score_data[0]['node_name']
        self.par_score_data_time = self.node_score_data[0]['time']
        self.par_score_data = self.node_score_data[1]
        if len(self.par_scoredatavariable_setting[self.par_scoredatavariable_setting['变量角色'] == 'TimeID']) == 1:
            self.flag_timeid_score = True
            self.timeid_score = \
            self.par_scoredatavariable_setting.loc[self.par_scoredatavariable_setting['变量角色'] == 'TimeID'][
                '变量名称'].values[0]
        if len(self.par_scoredatavariable_setting[self.par_scoredatavariable_setting['变量角色'] == '目标']) == 1:
            self.flag_target_score = True
            self.target_score = \
            self.par_scoredatavariable_setting.loc[self.par_scoredatavariable_setting['变量角色'] == '目标']['变量名称'].values[0]


        self.par_variable_type = self.node_model_data[0]['par_variable_type']
        self.predict_train_data = self.node_model_data[0]['predict_train_data']
        self.model_ppp = self.node_model_data[0]['model']
        self.f_scorecard = self.node_model_data[0]['scorecard_df']
        self.target_train = self.node_model_data[0]['report_para']['train_target']
        self.timeid_train = self.node_model_data[0]['report_para']['timeid_train']
        self.IGN_f_group_report = self.node_model_data[0]['report_para']['f_group_report']
        self.IGN_grouping_data = self.node_model_data[0]['IGN_grouping_data']
        if ac == 'setting':
            error_list=[]
            print('a')
            for i in range(0,2):
                if previous_node_name[i]!=None:
                    path_list = self.project_info[self.project_info['创建时间'] == previous_node_time[i]]['保存地址']
                    if len(path_list) == 0:
                        print(previous_node_time)
                        print({'name':previous_node_name[i],'time':previous_node_time[i]})
                        error_list=error_list+[{'name':previous_node_name[i],'time':previous_node_time[i]}]
            def continu(event):
                for child in self.master.winfo_children():
                    child.destroy()
                # 以前数据集更新了就重新更新结果
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
                self.load='Y'
                self.Start_UI()
                self.adjustsetting()
        else:
            print('b')
            self.reult_show_only(self.master)

    def import_data_node(self,event):
        path = self.project_info[self.project_info['模块名字'] == self.comboxlist_score_data.get()]['保存地址'][0]
        fr = open(path, 'rb')
        node_info = pickle.load(fr)
        fr.close()
        self.node_score_data=node_info
        self.par_scoredatavariable_setting = node_info[0]['data_variable_setting']
        self.par_score_dataname = node_info[0]['node_name']
        self.par_score_data_time = node_info[0]['time']
        self.par_score_data = node_info[1]
        # self.previous_reject_check_change = node_info[0]['check_change']
        self.previous_reject_node_usedlist = node_info[0]['use_node']
    def load_model_data(self, event):
        try:
            path = self.project_info[self.project_info['模块名字'] == self.comboxlist_SCR.get()]['保存地址'][0]
            fr = open(path, 'rb')
            node_data = pickle.load(fr)
            fr.close()
            self.node_model_data=node_data
            self.node_setting=node_data[0]
            self.node_type = node_data[0]['node_type']
            self.node_model_name=node_data[0]['node_name']
            self.node_model_time=node_data[0]['time']
            self.node_save_path=node_data[0]['node_save_path']
            self.par_variable_type=node_data[0]['par_variable_type']
            self.predict_train_data=node_data[0]['predict_train_data']
            self.model_ppp=node_data[0]['model']
            self.f_scorecard=node_data[0]['scorecard_df']
            self.target_train=node_data[0]['report_para']['train_target']
            self.timeid_train=node_data[0]['report_para']['timeid_train']
            self.IGN_f_group_report=node_data[0]['report_para']['f_group_report']
            self.IGN_grouping_data=node_data[0]['IGN_grouping_data']
        except  Exception as e:
            tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (self.comboxlist_SCR.get(), e))
    def Start_UI(self):
        self.start_window_base = self.master
        width = self.master.winfo_screenwidth() * 0.2
        height = self.master.winfo_screenheight() * 0.4
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()
        self.start_window_base.geometry(
            '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
        self.start_window_base.title('数据集打分')
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

        self.start_window_data = LabelFrame(self.start_window_base, text='导入打分数据:')
        L1 = Label(self.start_window_data, width=25, text="分组数据:", justify="left")
        L1.grid(column=0, row=0, sticky=(W))
        self.comboxlist_score_data = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_score_data["value"] = self.scoredf_list
        if self.par_score_data.empty != True:
            for i in range(len(self.scoredf_list)):
                if self.scoredf_list[i]==self.par_score_dataname:
                    self.comboxlist_score_data.current(i)
        self.comboxlist_score_data.bind("<<ComboboxSelected>>", lambda event: self.import_data_node(event))
        self.comboxlist_score_data.grid(column=1, row=0, sticky=(W))

        L3 = Label(self.start_window_data, width=25, text="导入模型:", justify="left")
        L3.grid(column=0, row=2, sticky=(W))
        self.comboxlist_SCR = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_SCR["value"] = self.SCR_list
        if self.IGN_grouping_data.empty != True:
            for i in range(len(self.SCR_list)):
                if self.SCR_list[i] == self.node_model_name:
                    self.comboxlist_SCR.current(i)
        self.comboxlist_SCR.grid(column=1, row=2, sticky=(W))
        self.comboxlist_SCR.bind("<<ComboboxSelected>>", lambda event: self.load_model_data(event))
        self.start_window_data.grid(columnspan=3, sticky=(W), padx=10, pady=10)

        self.button_setting_save = ttk.Button(self.start_window_base, text='（保存）退出')
        self.button_setting_save.grid(column=0, row=7, sticky=(W), padx=10, pady=10)
        self.button_setting_save.bind("<Button-1>", self.save_project)
        if (self.load == 'Y') | (self.finsh == 'Y'):
            self.check_result = ttk.Button(self.start_window_base, text='查看结果')
            self.check_result.grid(column=1, row=7, sticky=(W), padx=10, pady=10)
            self.check_result.bind("<Button-1>", self.scorecard_result_show_ui)
        if (self.load == 'N') & (self.finsh == 'N'):
            self.button_setting_run = ttk.Button(self.start_window_base, text='应用')
            self.button_setting_run.grid(column=2, row=7, sticky=(W))
            self.button_setting_run.bind("<Button-1>", self.Scoring)
        else:
            self.button_refresh_run = ttk.Button(self.start_window_base, text='刷新结果')
            self.button_refresh_run.grid(column=2, row=7, sticky=(W))
            self.button_refresh_run.bind("<Button-1>", self.Scoring)
            self.button_output = ttk.Button(self.start_window_base, text='导出数据集')
            self.button_output.grid(column=0, row=8, sticky=(W), padx=10, pady=10)
            self.button_output.bind("<Button-1>", self.out_dataset)
    def out_dataset(self, event):
        try:
            word = '导出数据集：\n Score数据集：%s/%s_train.csv \n' % (self.project_path, self.node_name)
            self.predict_score_data.to_csv(self.project_path + '/' + '%s_score.csv' % self.node_name, index=False,
                                           encoding='utf-8')
            tk.messagebox.showwarning('成功', word)
        except  Exception as e:
            tk.messagebox.showwarning('错误', e)
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
            pickle.dump([self.node_setting,self.predict_score_data,self.node_model_data,self.node_score_data], fw, protocol=4)
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
    def Scoring(self, event):
        # 检查各个数据集变量情况
        try:
            if self.model_start_flag=='N':
                self.model_start_flag='Y'
                variable_list = list(self.model_ppp[1].params.reset_index()['index'])
                try:
                    variable_list.remove('const')
                except:
                    pass
                variable_df=self.model_ppp[2]
                inmodel_df = variable_df[variable_df['model_var'].isin(variable_list)]
                include_var = list(inmodel_df[inmodel_df['ori_var'].isnull() == False]['ori_var'])
                if len(inmodel_df[inmodel_df['var_type'] == 'add']) > 0:
                    include_var = include_var + list(inmodel_df[inmodel_df['var_type'] == 'add']['vara'])
                    include_var = include_var + list(inmodel_df[inmodel_df['var_type'] == 'add']['varb'])
                unique_varable = list(set(include_var))
                model_var_num = list(set(self.IGN_grouping_data[(self.IGN_grouping_data['variable_name'].isin(unique_varable)) & (
                            self.IGN_grouping_data['variable_type'] == 'num')]['variable_name']))
                model_var_char = list(set(self.IGN_grouping_data[(self.IGN_grouping_data['variable_name'].isin(unique_varable)) & (
                            self.IGN_grouping_data['variable_type'] == 'char')]['variable_name']))

                if len(self.par_scoredatavariable_setting[self.par_scoredatavariable_setting['变量角色'] == 'TimeID']) == 1:
                    self.flag_timeid_score = True
                    self.timeid_score = self.par_scoredatavariable_setting.loc[self.par_scoredatavariable_setting['变量角色'] == 'TimeID']['变量名称'].values[0]
                if len(self.par_scoredatavariable_setting[self.par_scoredatavariable_setting['变量角色'] == '目标']) == 1:
                    self.flag_target_score = True
                    self.target_score = self.par_scoredatavariable_setting.loc[self.par_scoredatavariable_setting['变量角色'] == '目标']['变量名称'].values[0]

                self.varchar_score = list(self.par_scoredatavariable_setting[self.par_scoredatavariable_setting['变量类型'] == '字符型']['变量名称'])
                self.varnum_score = list(self.par_scoredatavariable_setting[self.par_scoredatavariable_setting['变量类型'] == '数值型']['变量名称'])

                if list(set(model_var_num) - set(self.varnum_score)) != []:
                    tk.messagebox.showwarning('错误',"打分集中没有如下数值型变量%s" % (list(set(model_var_num) - set(self.varnum_score))))
                    self.model_start_flag = 'N'
                elif list(set(model_var_char) - set(self.varchar_score )) != []:
                    tk.messagebox.showwarning('错误',"打分集中没有如下字符型变量%s" % (list(set(model_var_num) - set(self.varchar_score))))
                    self.model_start_flag = 'N'
                else:
                    self.grouped_score_data = binning.fit_bin_existing(data=self.par_score_data,
                                                                       varnum=model_var_num,
                                                                       varchar=model_var_char,
                                                                       target=self.target_train,
                                                                       group_info=self.IGN_grouping_data,
                                                                       data_only=True)
                    self.scorecard_data_pre(self.model_ppp)

                    node_save_path = self.project_path + '/' + '%s.model' % self.node_name
                    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    self.node_setting = {'node_type':'Scoring',
                                         'node_name':self.node_name,
                                         'model_nodename':self.node_model_name,
                                         'model_dataname':self.par_score_dataname,
                                         'score_data_node': self.par_score_data,
                                         'scored_data':self.predict_score_data,
                                         'time':nowTime,
                                         'node_save_path':node_save_path,
                                         'use_node':[self.node_name,self.node_model_name,self.par_score_dataname],
                                         'previous_node_name': [self.node_model_name,self.par_score_dataname],
                                         'previous_node_time': [self.node_model_time,self.par_score_data_time]
                                         }
                    self.finsh = 'Y'
                    for child in self.master.winfo_children():
                        child.destroy()
                    self.adjustsetting()
                    self.model_start_flag = 'N'
        except Exception as e:
            tk.messagebox.showwarning('错误', e)
            self.model_start_flag = 'N'
    def scorecard_result_show_ui(self, event):
        try:
            if self.result_page.state() == 'normal':
                tk.messagebox.showwarning('错误', "请先处理当前打开窗口")
        except:
            self.result_page = Toplevel(self.master)
            score_result_ui(mainframe=self.result_page,
                                predict_train_data=self.predict_train_data,
                                predict_score_data=self.predict_score_data,
                                train_target=self.target_train,
                                score_target=self.target_score,
                                train_time_id=self.timeid_train, score_time_id=self.timeid_score,
                                record_list=self.model_ppp[0], model=self.model_ppp[1], scorecarddf=self.f_scorecard,
                                f_group_report=self.IGN_f_group_report,
                                model_var_type=self.par_variable_type)
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
                    add_variabile_df = woe_score[(woe_score['var_type'] == 'add') & (woe_score['coff'].isnull() == False)]
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

            variable_df = model_re[2]
            woe_score = pd.merge(variable_df, cof, how='outer', left_on='model_var', right_on='woe_variable_name')


            # 给数据集打分
            self.predict_score_data = woe_predict(model=woe_model_re, intercept=self.par_intercept_flag,
                                                  df=self.grouped_score_data, woe_score=woe_score)
            self.predict_score_data = score_predict(self.f_scorecard, self.predict_score_data)
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
            B = self.par_odds_double_score / math.log(2)
            A = self.par_odds_score_ratio + B * math.log(self.par_odds_ratio)
            scorecard['SCORE'] = scorecard.apply(
                lambda x: A - B * x['coff'] if x['variable_name'] == 'const' else -B * x['coff'], axis=1)
            score_adjust = scorecard.groupby('variable_name')['SCORE'].min().reset_index().rename(
                {'SCORE': 'score_min'}, axis=1)
            adjust_num = score_adjust[score_adjust['score_min'] < 0]['score_min'].sum()
            score_adjust['score_min'][score_adjust['variable_name'] == 'const'] = -adjust_num
            f_scorecard = pd.merge(scorecard, score_adjust, how='left', on='variable_name')


            # 给数据集打分
            def grp_predict(model, intercept, df):
                input_list = list(pd.DataFrame(model.params).reset_index().rename({'index': 'grp_variable_name', 0: 'coff'}, axis=1)['grp_variable_name'])
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
            self.predict_score_data = grp_predict(model=grp_model, intercept=self.par_intercept_flag,
                                                  df=group_data_pre(self.grouped_score_data, f_scorecard))
            self.predict_score_data = score_predict(self.f_scorecard, self.predict_score_data)
    def reult_show_only(self, result_page):
        score_result_ui(mainframe=result_page,
                        predict_train_data=self.predict_train_data,
                        predict_score_data=self.predict_score_data,
                        train_target=self.target_train,
                        score_target=self.target_score,
                        train_time_id=self.timeid_train, score_time_id=self.timeid_score,
                        record_list=self.model_ppp[0], model=self.model_ppp[1], scorecarddf=self.f_scorecard,
                        f_group_report=self.IGN_f_group_report,
                        model_var_type=self.par_variable_type)

