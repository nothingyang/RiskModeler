import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
from pandastable import Table
from .func import binning
import math
from .base import group_func
import numpy as np
group_func =group_func()
binning =binning()
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class UserInterfacea():
    # Launch the df in a pandastable frame
    def __init__(self,mainfram ,s_data, group_info, f_data,
                 varchar, varnum, varchar_reject, varnum_reject ,varchar_oot, varnum_oot,
                 target ,target_reject ,target_oot,
                 train_data ,valid_data ,reject_data ,oot_data,
                 grouped_train_data ,grouped_valid_data ,grouped_reject_data ,grouped_oot_data,
                 vaild_flag ,reject_flag ,oot_flag,
                 flag_timeid_oot ,timeid_oot ,flag_timeid_reject ,timeid_reject ,flag_timeid_train
                 ,timeid_train,not_use,project_path,node_name, result_show=False):
        self.project_path=project_path
        self.node_name=node_name
        self.vaild_flag = vaild_flag
        self.reject_flag =reject_flag
        self.oot_flag =oot_flag
        self.result_show=result_show

        self.flag_timeid_oot =flag_timeid_oot
        self.timeid_oot =timeid_oot
        self.flag_timeid_reject =flag_timeid_reject
        self.timeid_reject =timeid_reject
        self.flag_timeid_train =flag_timeid_train
        self.timeid_train =timeid_train

        self.train_data = train_data
        self.valid_data = valid_data
        self.reject_data =reject_data
        self.oot_data =oot_data

        self.grouped_train_data =grouped_train_data
        self.grouped_valid_data =grouped_valid_data
        self.grouped_reject_data =grouped_reject_data
        self.grouped_oot_data =grouped_oot_data

        self.target = target
        self.target_reject =target_reject
        self.target_oot =target_oot

        self.varchar = varchar
        self.varnum = varnum
        self.varchar_reject =varchar_reject
        self.varnum_reject =varnum_reject
        self.varchar_oot =varchar_oot
        self.varnum_oot =varnum_oot


        self.f_group_data_modify = f_data
        self.s_group_data_modify = s_data
        self.group_info_modify = group_info

        self.var = 'all'
        self.flag_sum = False
        self.flag_detail = True

        self.detail, self.summary, self.modify = self.renamedata(s_group_data=s_data, f_group_data=f_data,
                                                                 validation=self.vaild_flag)
        self.detail_modify = self.detail
        self.summary_modify = self.summary
        self.not_use =not_use
        self.modify_modify = self.modify
        self.flag_s_needsave = False
        self.flag_f_needsave = False
        self.create_firstUI(mainfram)
        self.sort()
        self.refresh_df(df=self.detail_modify)

    def renamedata(self, s_group_data, f_group_data, validation):
        if 's_group' in list(s_group_data.columns):
            pass
        else:
            s_group_data['s_group'] = np.nan
        if validation == True:
            f_group_data = f_group_data.rename(
                {'pct_f_N_obs': '组占比', 'badrate_vaild': '验证集事件率', 'psi_n': '样本psi', 'psi_bad': '事件psi',
                 'variable_name': '变量名称', 'label': '注释', 'f_N_obs': '粗分组样本数', 'f_group': '组编号', 'iv': '信息熵',
                 'f_Bad_rate': '粗分组事件率', 'f_N_bad': '粗分组事件数', 'miss_rate': '缺失率'}, axis=1)
            detail = f_group_data[['变量名称', '注释', '粗分组样本数', '组编号', '信息熵',
                                   '粗分组事件率', '粗分组事件数', '缺失率', 'woe', '样本psi', '事件psi']]
            summary = detail[['变量名称', '信息熵', '缺失率', '样本psi']].drop_duplicates()
        else:
            f_group_data = f_group_data.rename(
                {'variable_name': '变量名称', 'label': '注释', 'f_N_obs': '粗分组样本数', 'f_group': '组编号', 'iv': '信息熵',
                 'f_Bad_rate': '粗分组事件率', 'f_N_bad': '粗分组事件数', 'miss_rate': '缺失率'}, axis=1)
            detail = f_group_data[['变量名称', '注释', '粗分组样本数', '组编号', '信息熵',
                                   '粗分组事件率', '粗分组事件数', '缺失率', 'woe']]
            summary = detail[['变量名称', '信息熵', '缺失率']].drop_duplicates()
        modify = s_group_data[
            ['variable_name', 'label', 's_group', 'f_group', 's_Bad_rate', 'f_Bad_rate', 'iv', 's_N_bad', 's_N_obs',
             'f_N_bad', 'f_N_obs', 'value']]
        modify = modify.rename(
            {'variable_name': '变量名称', 's_group': "细分组编号", 'label': '注释', 'f_group': '粗分组编号', 'iv': '信息熵',
             'f_Bad_rate': '粗分组事件率', 's_Bad_rate': '细分组事件率', 's_N_obs': '细分组样本数', 's_N_bad': '细分组事件数',
             'f_N_bad': '粗分组事件数', 'f_N_obs': '粗分组样本数', 'value': '值'}, axis=1)

        return detail, summary, modify
    def sort(self):
        self.summary_modify = self.summary_modify.sort_values(by=['信息熵'], ascending=[False])
        self.detail_modify = self.detail_modify.sort_values(by=['信息熵', '变量名称', '组编号'], ascending=[False, False, True])
    def create_firstUI(self ,mainfram):
        self.root = mainfram
        self.root.title("Univariate Analysis Report.")
        self.root.resizable(width=True, height=True)
    def create_modify(self ,type_c):
        if type_c =='first':
            value =[]
            if self.flag_timeid_train ==True:
                value.append('Trian')
                if self.vaild_flag==True:
                    value.append('Valid')
            if self.flag_timeid_oot==True:
                value.append('OOT')
            if self.flag_timeid_reject==True:
                value.append('Reject')
            value.append('NO')
            self.fist_value =value[0]
        else:
            self.fist_value =str(self.comboxlist_sample_select.get())
        if type_c !='refrsh_pic':
            bad_pct ,total_pct ,final ,node_type =self.groupplot_datainit(self.fist_value)
        else:
            bad_pct ,total_pct ,final ,node_type =self.groupplot_datarecalculate(self.fist_value)
        if type_c !='first':
            for child in self.top_modify.winfo_children():
                child.destroy()
        else:
            self.top_modify = Toplevel(self.root)
            self.top_modify.title('%s ' %self.modify_var)
            width = self.top_modify.winfo_screenwidth( ) *0.7
            height = self.top_modify.winfo_screenheight( ) *0.6

            # 获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
            screenwidth = self.top_modify.winfo_screenwidth()
            screenheight = self.top_modify.winfo_screenheight()
            alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth -width ) /2, (screenheight -height ) /2)
            self.top_modify.geometry(alignstr)
            self.old_iv =round(self.modify_modify_var['信息熵'].unique()[0] ,2)
        self.new_iv =round(self.modify_modify_var['信息熵'].unique()[0] ,2)
        self.top_modify.resizable(width=True, height=True)
        if self.result_show == False:
            button =Frame(self.top_modify)
            test_button4 = ttk.Button(button, text='保存')
            test_button4.pack(side=LEFT ,fill='x' ,anchor='w' ,padx=3 ,pady=3)
            test_button4.bind("<Button-1>", self.save_modify_var)

            test_button6 = ttk.Button(button, text='关闭')
            test_button6.pack( side=LEFT ,fill='x' ,anchor='w' ,padx=3 ,pady=3)
            test_button6.bind("<Button-1>" ,self.close_modify_var)

            test_button5 = ttk.Button(button, text='刷新图表' ,command=lambda: self.create_modify('refrsh_pic'))
            test_button5.pack( side=LEFT ,fill='x' ,anchor='w' ,padx=3 ,pady=3)


            labela =ttk.Label(button ,text='新IV值： %2f ' %self.new_iv)
            labela.pack(side=LEFT, anchor='n', padx=30, pady=3)
            labelb = ttk.Label(button, text='旧IV值：%2f ' % self.old_iv)
            labelb.pack(side=LEFT, anchor='n', padx=20, pady=3)

            self.comboxlist_variable_use = ttk.Combobox(button, width=15)
            self.comboxlist_variable_use["value"] = ['使用', '不使用']
            if self.modify_var in self.not_use:
                self.comboxlist_variable_use.current(1)
            else:
                self.comboxlist_variable_use.current(0)
            self.comboxlist_variable_use.pack(side=RIGHT, fill='x', anchor='e', padx=3, pady=3)
            labelc = ttk.Label(button, text='变量使用： ')
            labelc.pack(side=RIGHT, fill='x', anchor='e')
            button.pack(side=TOP, anchor='sw', fill='x', padx=5, pady=5)

        self.refresh_modify_data()

        self.plot_group(bad_pct, total_pct, final, node_type)
    def refresh_modify_data(self):
        try:
            if self.modify_var in self.varchar:

                self.modify_modify_var = self.modify_modify_var.drop(columns='细分组编号')
            else:

                pass
        except:

            pass
        if self.modify_var in self.varnum:
            self.modify_modify_var = self.modify_modify_var.sort_values(by=['细分组编号', '粗分组编号'], ascending=[True, True])
        else:
            self.modify_modify_var = self.modify_modify_var.sort_values(by=['粗分组编号'], ascending=[True])
        f_modify = LabelFrame(self.top_modify, text='分组信息')

        self.modify_table = self.ptm = Table(f_modify, dataframe=self.modify_modify_var)
        self.ptm.show()
        f_modify.pack(side=TOP, anchor='nw', fill='both', expand='YES', padx=5, pady=5)

        def handle_left_click_modify_add(event):
            try:
                self.comboxlist_modify_f_group.destroy()
                self.top_s_group_modify.destroy()
            except:
                pass
            self.modify_table.handle_left_click(event)

        if self.result_show == False:
            self.modify_table.bind("<Button-3>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Button-1>", handle_left_click_modify_add)
            self.modify_table.bind("<Button-2>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Double-Button-3>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Double-Button-1>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Double-Button-2>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Triple-Button-3>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Triple-Button-1>", self.handle_right_click_modify_add)
            self.modify_table.bind("<Triple-Button-2>", self.handle_right_click_modify_add)
    def close_modify_var(self, event):

        if self.flag_s_needsave == False:

            try:
                self.top_s_group_modify().destroy()
            except:
                pass
            try:
                self.comboxlist_modify_f_group().destroy()
            except:
                pass
            self.top_modify.destroy()

        else:
            def close_modify_var_yes(event):
                self.save_modify_var(event)
                try:
                    self.top_s_group_modify.destroy()
                except:
                    pass
                try:
                    self.comboxlist_modify_f_group.destroy()
                except:
                    pass

                close_modify_error1.destroy()
                self.top_modify.destroy()
                self.refresh_df_after_modify(event)

            def close_modify_var_no(event):
                try:
                    self.top_s_group_modify.destroy()
                except:
                    pass
                try:
                    self.comboxlist_modify_f_group.destroy()
                except:
                    pass
                close_modify_error1.destroy()
                self.top_modify.destroy()

            close_modify_error1 = Toplevel(self.top_modify)
            close_modify_error1.geometry('%sx%s+%s+%s' % (300, 50, event.x_root + 300, event.y_root))
            L2 = Label(close_modify_error1, text="是否保存已更改的分组")
            L2.grid(column=0, row=0, sticky=(W))
            test_button4 = ttk.Button(close_modify_error1, text='是')
            test_button4.grid(column=0, row=1, sticky=(W))
            test_button4.bind("<Button-1>", close_modify_var_yes)
            test_button5 = ttk.Button(close_modify_error1, text='否')
            test_button5.grid(column=1, row=1, sticky=(W))
            test_button5.bind("<Button-1>", close_modify_var_no)
    def close_master(self, event):
        def close(event):
            try:
                self.top_s_group_modify().destroy()
            except:
                pass
            try:
                self.comboxlist_modify_f_group().destroy()
            except:
                pass
            try:
                self.top_modify.destroy()
            except:
                pass
            self.root.destroy()

        def yes(event):
            self.flag_f_needsave = True
            close(event)

        close_master = Toplevel(self.root)
        close_master.geometry('%sx%s+%s+%s' % (300, 50, event.x_root - 300, event.y_root))
        close_master.title('')
        L2 = Label(close_master, text="是否保存已更改的分组")
        L2.grid(column=0, row=0, sticky=(W))
        test_button4 = ttk.Button(close_master, text='是')
        test_button4.grid(column=0, row=1, sticky=(W))
        test_button4.bind("<Button-1>", yes)
        test_button5 = ttk.Button(close_master, text='否')
        test_button5.grid(column=1, row=1, sticky=(W))
        test_button5.bind("<Button-1>", close)
    def refresh_df(self, df):
        for child in self.root.winfo_children():
            child.destroy()
        toplabel = LabelFrame(self.root, text='菜单栏')
        test_button = ttk.Button(toplabel, text='明细')
        test_button.pack(side=LEFT, fill='x', anchor='w', padx=3, pady=3)
        test_button.bind("<Button-1>", self.change_df_detail)

        test_button1 = ttk.Button(toplabel, text='总结')
        test_button1.pack(side=LEFT, fill='x', anchor='w', padx=3, pady=3)
        test_button1.bind("<Button-1>", self.change_df_sum)


        if self.result_show == False:
            test_button4 = ttk.Button(toplabel, text='刷新')
            test_button4.pack(side=LEFT, fill='x', anchor='w', padx=3, pady=3)
            test_button4.bind("<Button-1>", self.refresh_df_after_modify)

            test_button7 = ttk.Button(toplabel, text='导出')
            test_button7.pack(side=LEFT, fill='x', anchor='w', padx=3, pady=3)
            test_button7.bind("<Button-1>", self.output_excel)

            test_button5 = ttk.Button(toplabel, text='关闭')
            test_button5.pack(side=LEFT, fill='x', anchor='w', padx=3, pady=3)
            test_button5.bind("<Button-1>", self.close_master)
        toplabel.pack(side=TOP, anchor='w')
        f = LabelFrame(self.root, text='分组结果')
        screen_width = f.winfo_screenwidth() * 0.5
        screen_height = f.winfo_screenheight() * 0.8
        if self.flag_sum==True:
            self.summary_modify['是否使用'] = self.summary_modify.apply(lambda x: '不使用' if x['变量名称'] in self.not_use else '使用', axis=1)
            self.table_sum = self.pt = Table(f, dataframe=self.summary_modify, height=screen_height, width=screen_width)
        else:
            self.detail_modify['是否使用'] = self.detail_modify.apply(lambda x: '不使用' if x['变量名称'] in self.not_use else '使用', axis=1)
            self.table_sum = self.pt = Table(f, dataframe=self.detail_modify , height=screen_height, width=screen_width)

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
        f.pack(side=TOP, anchor='nw', fill='both', expand='YES', padx=5, pady=5)
    def change_df_detail(self, event):
        # Responds to button

        self.flag_sum = False
        self.flag_detail = True
        # self.sort()
        self.refresh_df(df=self.detail_modify)
    def change_df_sum(self, event):
        # Responds to button
        self.flag_sum = True
        self.flag_detail = False
        # self.sort()
        self.refresh_df(df=self.summary_modify)
    def refresh_df_after_modify(self, event):
        # self.sort()
        if self.flag_sum == True:
            self.refresh_df(df=self.summary_modify)
        else:
            self.refresh_df(df=self.detail_modify)
    def output_excel(self,event):
        try:
            if self.flag_sum == True:
                df = self.summary_modify
            else:
                df = self.detail_modify
            path=self.project_path + '/' + '%s_groupingresult.xlsx' % self.node_name
            df.to_excel(path)
            path1 = self.project_path + '/' + '%s_groupingresult_detail.xlsx' % self.node_name
            self.s_group_data_modify.to_excel(path1)
            tk.messagebox.showwarning('成功', "已经成功导出到%s \n已经成功导出到%s " %(path,path1))
        except  Exception as e:
            tk.messagebox.showwarning('错误', e)
    def handle_left_click(self, event):
        rowclicked = self.pt.get_row_clicked(event)
        if self.flag_sum == True:
            self.modify_var = self.summary_modify.iloc[rowclicked]['变量名称']
        else:
            self.modify_var = self.detail_modify.iloc[rowclicked]['变量名称']
        try:
            if self.top_modify.state() == 'normal':
                tk.messagebox.showwarning('错误', "错误：请先关闭已经打开窗口")
        except:
            self.modify_modify_var = self.modify_modify[self.modify_modify['变量名称'] == self.modify_var]
            self.summary_modify_var = self.summary_modify[self.summary_modify['变量名称'] == self.modify_var]
            self.detail_modify_var = self.detail_modify[self.detail_modify['变量名称'] == self.modify_var]
            self.group_info_modify_var = self.group_info_modify[
                self.group_info_modify['variable_name'] == self.modify_var]
            self.flag_s_needsave = False
            self.create_modify('first')
            self.root.wait_window(self.top_modify)
            self.refresh_df_after_modify(event)
    def handle_left_click_modify_add(self, event):
        try:
            self.comboxlist_modify_f_group.destroy()
            self.top_s_group_modify.destroy()
        except:
            pass
        self.modify_table.handle_left_click(event)
    def handle_right_click_modify_add(self, event):
        try:
            self.comboxlist_modify_f_group.destroy()
            self.top_s_group_modify.destroy()
        except:
            pass
        rowclicked = self.ptm.get_row_clicked(event)
        colclicked = self.ptm.get_col_clicked(event)
        if list(self.modify_modify_var.columns)[colclicked] == '细分组编号':
            try:
                if np.isnan(self.modify_modify_var.iloc[rowclicked]['值']) == True:
                    value = self.modify_modify_var.iloc[rowclicked]['细分组编号']
                    self.s_max = \
                        self.group_info_modify_var[self.group_info_modify_var['s_group'] == value]['s_max'].values[0]
                    self.s_min = \
                        self.group_info_modify_var[self.group_info_modify_var['s_group'] == value]['s_min'].values[0]
                    self.top_s_group_modify = Toplevel()
                    self.top_s_group_modify.title('增加新的分隔值')
                    self.top_s_group_modify.geometry('%sx%s+%s+%s' % (150, 100, event.x_root, event.y_root))
                    self.L1 = Label(self.top_s_group_modify, text="请输入新分隔值")
                    self.L1.grid()
                    name = StringVar()
                    self.E1 = Entry(self.top_s_group_modify, textvariable=name, bd=5)
                    self.E1.grid()
                    self.f1 = ttk.Button(self.top_s_group_modify, text='确认')
                    self.f1.grid()
                    self.f1.bind("<Button-1>", self.s_group_modify)
                    self.flag_s_needsave = True

            except:
                pass

        elif list(self.modify_modify_var.columns)[colclicked] == '粗分组编号':
            if self.modify_var in self.varnum:
                value = self.modify_modify_var.iloc[rowclicked]['细分组编号']
                self.modify_s_group_num = value
                list_group = self.modify_f_group_list(s_group=self.group_info_modify_var, s_g=self.modify_s_group_num)
            else:
                self.modify_s_group_num = self.modify_modify_var.iloc[rowclicked]['值']
                f_group = self.modify_modify_var.iloc[rowclicked]['粗分组编号']
                if len(self.group_info_modify_var[self.group_info_modify_var['f_group'] == f_group]) > 1:
                    list_group = list(self.group_info_modify_var['f_group'].unique())
                    list_group.append(max(list_group) + 1)
                else:
                    list_group = list(self.group_info_modify_var['f_group'].unique())
            self.comboxlist_modify_f_group = ttk.Combobox(self.top_modify)

            self.comboxlist_modify_f_group["value"] = list_group
            self.top_modify.update()
            self.comboxlist_modify_f_group.place(x=event.x_root - self.top_modify.winfo_rootx(),
                                                 y=event.y_root - self.top_modify.winfo_rooty())
            self.comboxlist_modify_f_group.bind("<<ComboboxSelected>>", self.modify_combo)
            self.flag_s_needsave = True
        else:
            pass
    def groupplot_datainit(self, select_data):
        grouped_traindata = self.grouped_train_data
        total_pct = pd.DataFrame(
            grouped_traindata.groupby(['f_group_%s' % self.modify_var])['f_group_%s' % self.modify_var].count()).rename(
            {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
        total_pct['Group'] = total_pct['f_group_%s' % self.modify_var]
        total_pct['pct_train'] = total_pct['num'] / total_pct['num'].sum()
        total_pct = total_pct.drop(columns=['f_group_%s' % self.modify_var])

        bad_pct = grouped_traindata.groupby(['f_group_%s' % self.modify_var]).agg(
            {self.target: ['sum', 'count']}).reset_index().rename({'f_group_%s' % self.modify_var: 'Group'}, axis=1)
        bad_pct['badrate_train'] = bad_pct[self.target]['sum'] / bad_pct[self.target]['count']
        bad_pct = bad_pct[['Group', 'badrate_train']]

        if self.vaild_flag == True:
            grouped_validdata = self.grouped_valid_data
            valid_pct = pd.DataFrame(grouped_validdata.groupby(['f_group_%s' % self.modify_var])[
                                         'f_group_%s' % self.modify_var].count()).rename(
                {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
            valid_pct['Group'] = valid_pct['f_group_%s' % self.modify_var]
            valid_pct['pct_valid'] = valid_pct['num'] / valid_pct['num'].sum()
            valid_pct = valid_pct.drop(columns=['f_group_%s' % self.modify_var])
            total_pct = pd.merge(total_pct, valid_pct, how='outer', on='Group')

            validbad_pct = grouped_validdata.groupby(['f_group_%s' % self.modify_var]).agg(
                {self.target: ['sum', 'count']}).reset_index().rename({'f_group_%s' % self.modify_var: 'Group'}, axis=1)
            validbad_pct['badrate_valid'] = validbad_pct[self.target]['sum'] / validbad_pct[self.target]['count']
            validbad_pct = validbad_pct[['Group', 'badrate_valid']]
            bad_pct = pd.merge(bad_pct, validbad_pct, how='outer', on='Group')

        if (self.reject_flag == True) & (self.modify_var in (self.varchar_reject+self.varnum_reject)):
            grouped_rejectdata = self.grouped_reject_data
            reject_pct = pd.DataFrame(grouped_rejectdata.groupby(['f_group_%s' % self.modify_var])['f_group_%s' % self.modify_var].count()).rename(
                {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
            reject_pct['Group'] = reject_pct['f_group_%s' % self.modify_var]
            reject_pct['pct_reject'] = reject_pct['num'] / reject_pct['num'].sum()
            reject_pct = reject_pct.drop(columns=['f_group_%s' % self.modify_var])
            total_pct = pd.merge(total_pct, reject_pct, how='outer', on='Group')
            if self.target_reject != None:
                rejectbad_pct = grouped_rejectdata.groupby(['f_group_%s' % self.modify_var]).agg(
                    {self.target_reject: ['sum', 'count']}).reset_index().rename(
                    {'f_group_%s' % self.modify_var: 'Group'}, axis=1)
                rejectbad_pct['badrate_reject'] = rejectbad_pct[self.target_reject]['sum'] / \
                                                  rejectbad_pct[self.target_reject]['count']
                rejectbad_pct = rejectbad_pct[['Group', 'badrate_reject']]
                bad_pct = pd.merge(bad_pct, rejectbad_pct, how='outer', on='Group')
        if (self.oot_flag == True)& (self.modify_var in (self.varchar_oot+self.varnum_oot)):
            grouped_ootdata = self.grouped_oot_data
            oot_pct = pd.DataFrame(self.grouped_oot_data.groupby(['f_group_%s' % self.modify_var])[
                                       'f_group_%s' % self.modify_var].count()).rename(
                {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
            oot_pct['Group'] = oot_pct['f_group_%s' % self.modify_var]
            oot_pct['pct_oot'] = oot_pct['num'] / oot_pct['num'].sum()
            oot_pct = oot_pct.drop(columns=['f_group_%s' % self.modify_var])
            total_pct = pd.merge(total_pct, oot_pct, how='outer', on='Group')
            if self.target_oot != None:
                ootbad_pct = self.grouped_oot_data.groupby(['f_group_%s' % self.modify_var]).agg(
                    {self.target_oot: ['sum', 'count']}).reset_index().rename({'f_group_%s' % self.modify_var: 'Group'},
                                                                              axis=1)
                ootbad_pct['badrate_oot'] = ootbad_pct[self.target_oot]['sum'] / ootbad_pct[self.target_oot]['count']
                ootbad_pct = ootbad_pct[['Group', 'badrate_oot']]
                bad_pct = pd.merge(bad_pct, ootbad_pct, how='outer', on='Group')
        if select_data == 'Trian':
            base_list=[]
            for gr in self.grouped_train_data['f_group_%s' % self.modify_var].unique():
                for ti in self.grouped_train_data[self.timeid_train].unique():
                    base_list.append({'Group':gr,self.timeid_train:ti})
            base_df=pd.DataFrame(base_list)
            valid_pct = pd.DataFrame(
                self.grouped_train_data.groupby(['f_group_%s' % self.modify_var, self.timeid_train])[self.target].agg(
                    {'count', 'sum'}).reset_index())
            total = pd.DataFrame(
                self.grouped_train_data.groupby([self.timeid_train])[self.target].agg({'count'}).reset_index().rename(
                    {'count': 'total_num'}, axis=1))
            final = pd.merge(total, valid_pct, how='outer', on=self.timeid_train)
            final['popu_pct'] = final['count'] / final['total_num']
            final['bad_rate'] = final['sum'] / final['count']
            final['Group'] = final['f_group_%s' % self.modify_var]
            final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_train])
            node_type = 'train'
        elif select_data == 'Valid':
            base_list=[]
            for gr in self.grouped_train_data['f_group_%s' % self.modify_var].unique():
                for ti in self.grouped_train_data[self.timeid_train].unique():
                    base_list.append({'Group':gr,self.timeid_train:ti})
            base_df = pd.DataFrame(base_list)
            valid_pct = pd.DataFrame(
                self.grouped_valid_data.groupby(['f_group_%s' % self.modify_var, self.timeid_train])[self.target].agg(
                    {'count', 'sum'}).reset_index())
            total = pd.DataFrame(
                self.grouped_valid_data.groupby([self.timeid_train])[self.target].agg({'count'}).reset_index().rename(
                    {'count': 'total_num'}, axis=1))
            final = pd.merge(total, valid_pct, how='outer', on=self.timeid_train)
            final['popu_pct'] = final['count'] / final['total_num']
            final['bad_rate'] = final['sum'] / final['count']
            final['Group'] = final['f_group_%s' % self.modify_var]
            final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_train])
            node_type = 'vaild'
        elif (select_data == 'Reject')& (self.modify_var in (self.varchar_reject+self.varnum_reject)):
            base_list=[]
            for gr in self.grouped_reject_data['f_group_%s' % self.modify_var].unique():
                for ti in self.grouped_reject_data[self.timeid_reject].unique():
                    base_list.append({'Group':gr,self.timeid_reject:ti})
            base_df=pd.DataFrame(base_list)
            if self.target_reject != None:
                valid_pct = pd.DataFrame(
                    self.grouped_reject_data.groupby(['f_group_%s' % self.modify_var, self.timeid_reject])[
                        self.target_reject].agg({'count', 'sum'}).reset_index())
                total = pd.DataFrame(self.grouped_reject_data.groupby([self.timeid_reject])[self.target_reject].agg(
                    {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_reject)
                final['popu_pct'] = final['count'] / final['total_num']
                final['bad_rate'] = final['sum'] / final['count']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_reject])
                node_type = 'reject'
            else:
                valid_pct = pd.DataFrame(
                    self.grouped_reject_data.groupby(['f_group_%s' % self.modify_var, self.timeid_reject])[
                        self.modify_var].agg({'count'}).reset_index())
                total = pd.DataFrame(
                    self.grouped_reject_data.groupby([self.timeid_reject])['f_group_%s' % self.modify_var].agg(
                        {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_reject)
                final['popu_pct'] = final['count'] / final['total_num']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_reject])
                node_type = 'reject'
        elif (select_data == 'OOT')& (self.modify_var in (self.varchar_oot+self.varnum_oot)):
            base_list=[]
            for gr in self.grouped_oot_data['f_group_%s' % self.modify_var].unique():
                for ti in self.grouped_oot_data[self.timeid_oot].unique():
                    base_list.append({'Group':gr,self.timeid_oot:ti})
            base_df=pd.DataFrame(base_list)
            if self.target_oot != None:
                valid_pct = pd.DataFrame(
                    self.grouped_oot_data.groupby(['f_group_%s' % self.modify_var, self.timeid_oot])[self.target_oot].agg(
                        {'count', 'sum'}).reset_index())
                total = pd.DataFrame(
                    self.grouped_oot_data.groupby([self.timeid_oot])[self.target_oot].agg({'count'}).reset_index().rename(
                        {'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_oot)
                final['popu_pct'] = final['count'] / final['total_num']
                final['bad_rate'] = final['sum'] / final['count']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final=pd.merge(base_df,final,how='outer',on=['Group',self.timeid_oot])
                node_type = 'oot'
            else:
                valid_pct = pd.DataFrame(
                    self.grouped_oot_data.groupby(['f_group_%s' % self.modify_var, self.timeid_oot])[self.modify_var].agg(
                        {'count'}).reset_index())
                total = pd.DataFrame(self.grouped_oot_data.groupby([self.timeid_oot])['f_group_%s' % self.modify_var].agg(
                    {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_oot)
                final['popu_pct'] = final['count'] / final['total_num']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_oot])
                node_type = 'oot'
        else:
            final = pd.DataFrame()
            node_type = 'no'
        print(bad_pct)
        bad_pct=pd.merge(bad_pct,total_pct[['Group']].drop_duplicates(),how='outer',on='Group')
        # bad_pct=bad_pct.fillna(0)
        # total_pct=total_pct.fillna(0)
        # final=final.fillna(0)
        return bad_pct, total_pct, final, node_type
    def groupplot_datarecalculate(self, select_data):
        if self.modify_var in self.varchar:
            grouped_traindata = group_func.charvarexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                        inputdata=self.train_data, col=self.modify_var,
                                                        target=self.target)
            self.tt_train = grouped_traindata.drop_duplicates()
            grouped_traindata = pd.merge(self.train_data, self.tt_train, how='left')
        else:
            grouped_traindata = group_func.numericexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                        inputdata=self.train_data, col=self.modify_var,
                                                        target=self.target, modify=False,
                                                        add_value=0)
            self.tt_train = grouped_traindata.drop_duplicates()
            grouped_traindata = pd.merge(self.train_data, self.tt_train, how='left')

        total_pct = pd.DataFrame(
            grouped_traindata.groupby(['f_group_%s' % self.modify_var])['f_group_%s' % self.modify_var].count()).rename(
            {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
        total_pct['Group'] = total_pct['f_group_%s' % self.modify_var]
        total_pct['pct_train'] = total_pct['num'] / total_pct['num'].sum()
        total_pct = total_pct.drop(columns=['f_group_%s' % self.modify_var])

        bad_pct = grouped_traindata.groupby(['f_group_%s' % self.modify_var]).agg(
            {self.target: ['sum', 'count']}).reset_index().rename({'f_group_%s' % self.modify_var: 'Group'}, axis=1)
        bad_pct['badrate_train'] = bad_pct[self.target]['sum'] / bad_pct[self.target]['count']
        bad_pct = bad_pct[['Group', 'badrate_train']]

        if self.vaild_flag == True:
            if self.modify_var in self.varchar:
                grouped_validdata = group_func.charvarexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                            inputdata=self.valid_data, col=self.modify_var,
                                                            target=self.target)
            else:
                grouped_validdata = group_func.numericexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                            inputdata=self.valid_data, col=self.modify_var,
                                                            target=self.target, modify=False,
                                                            add_value=0)
            self.tt_vaild = grouped_validdata.drop_duplicates()
            grouped_validdata = pd.merge(self.valid_data, self.tt_vaild, how='left')

            valid_pct = pd.DataFrame(grouped_validdata.groupby(['f_group_%s' % self.modify_var])[
                                         'f_group_%s' % self.modify_var].count()).rename(
                {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
            valid_pct['Group'] = valid_pct['f_group_%s' % self.modify_var]
            valid_pct['pct_valid'] = valid_pct['num'] / valid_pct['num'].sum()
            valid_pct = valid_pct.drop(columns=['f_group_%s' % self.modify_var])
            total_pct = pd.merge(total_pct, valid_pct, how='outer', on='Group')

            validbad_pct = grouped_validdata.groupby(['f_group_%s' % self.modify_var]).agg(
                {self.target: ['sum', 'count']}).reset_index().rename({'f_group_%s' % self.modify_var: 'Group'}, axis=1)
            validbad_pct['badrate_valid'] = validbad_pct[self.target]['sum'] / validbad_pct[self.target]['count']
            validbad_pct = validbad_pct[['Group', 'badrate_valid']]
            bad_pct = pd.merge(bad_pct, validbad_pct, how='outer', on='Group')

        if (self.reject_flag == True)& (self.modify_var in (self.varchar_reject+self.varnum_reject)):
            if self.modify_var in self.varchar_reject:
                grouped_rejectdata = group_func.charvarexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                             inputdata=self.reject_data, col=self.modify_var,
                                                             target=self.target)
            elif self.modify_var in self.varnum_reject:
                grouped_rejectdata = group_func.numericexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                             inputdata=self.reject_data, col=self.modify_var,
                                                             target=self.target, modify=False,
                                                             add_value=0)
            self.tt_reject = grouped_rejectdata.drop_duplicates()
            grouped_rejectdata = pd.merge(self.reject_data, self.tt_reject, how='left')

            reject_pct = pd.DataFrame(grouped_rejectdata.groupby(['f_group_%s' % self.modify_var])[
                                          'f_group_%s' % self.modify_var].count()).rename(
                {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
            reject_pct['Group'] = reject_pct['f_group_%s' % self.modify_var]
            reject_pct['pct_reject'] = reject_pct['num'] / reject_pct['num'].sum()
            reject_pct = reject_pct.drop(columns=['f_group_%s' % self.modify_var])
            total_pct = pd.merge(total_pct, reject_pct, how='outer', on='Group')
            if self.target_reject != None:
                rejectbad_pct = grouped_rejectdata.groupby(['f_group_%s' % self.modify_var]).agg(
                    {self.target_reject: ['sum', 'count']}).reset_index().rename(
                    {'f_group_%s' % self.modify_var: 'Group'}, axis=1)
                rejectbad_pct['badrate_reject'] = rejectbad_pct[self.target_reject]['sum'] / \
                                                  rejectbad_pct[self.target_reject]['count']
                rejectbad_pct = rejectbad_pct[['Group', 'badrate_reject']]
                bad_pct = pd.merge(bad_pct, rejectbad_pct, how='outer', on='Group')
        if (self.oot_flag == True ) & (self.modify_var in (self.varchar_oot+self.varnum_oot)):
            if self.modify_var in self.varchar_oot:
                grouped_ootdata = group_func.charvarexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                          inputdata=self.oot_data, col=self.modify_var,
                                                          target=self.target)
            elif self.modify_var in self.varnum_oot:
                grouped_ootdata = group_func.numericexist(group_info_old=self.group_info_modify_var, data_only=True,
                                                          inputdata=self.oot_data, col=self.modify_var,
                                                          target=self.target, modify=False,
                                                          add_value=0)

            self.tt_oot = grouped_ootdata.drop_duplicates()
            grouped_ootdata = pd.merge(self.oot_data, self.tt_oot, how='left')
            oot_pct = pd.DataFrame(grouped_ootdata.groupby(['f_group_%s' % self.modify_var])[
                                       'f_group_%s' % self.modify_var].count()).rename(
                {'f_group_%s' % self.modify_var: 'num'}, axis=1).reset_index()
            oot_pct['Group'] = oot_pct['f_group_%s' % self.modify_var]
            oot_pct['pct_oot'] = oot_pct['num'] / oot_pct['num'].sum()
            oot_pct = oot_pct.drop(columns=['f_group_%s' % self.modify_var])
            total_pct = pd.merge(total_pct, oot_pct, how='outer', on='Group')
            if self.target_oot != None:
                ootbad_pct = grouped_ootdata.groupby(['f_group_%s' % self.modify_var]).agg(
                    {self.target_oot: ['sum', 'count']}).reset_index().rename({'f_group_%s' % self.modify_var: 'Group'},
                                                                              axis=1)
                ootbad_pct['badrate_oot'] = ootbad_pct[self.target_oot]['sum'] / ootbad_pct[self.target_oot]['count']
                ootbad_pct = ootbad_pct[['Group', 'badrate_oot']]
                bad_pct = pd.merge(bad_pct, ootbad_pct, how='outer', on='Group')
        if select_data == 'Trian':
            base_list=[]
            for gr in grouped_traindata['f_group_%s' % self.modify_var].unique():
                for ti in grouped_traindata[self.timeid_train].unique():
                    base_list.append({'Group':gr, self.timeid_train:ti})
            base_df=pd.DataFrame(base_list)
            valid_pct = pd.DataFrame(
                grouped_traindata.groupby(['f_group_%s' % self.modify_var, self.timeid_train])[self.target].agg(
                    {'count', 'sum'}).reset_index())
            total = pd.DataFrame(
                grouped_traindata.groupby([self.timeid_train])[self.target].agg({'count'}).reset_index().rename(
                    {'count': 'total_num'}, axis=1))
            final = pd.merge(total, valid_pct, how='outer', on=self.timeid_train)
            final['popu_pct'] = final['count'] / final['total_num']
            final['bad_rate'] = final['sum'] / final['count']
            final['Group'] = final['f_group_%s' % self.modify_var]
            final.to_csv(r'D:\SynologyDrive\india_project\final.csv')
            base_df.to_csv(r'D:\SynologyDrive\india_project\base.csv')
            final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_train])
            node_type = 'train'
        elif select_data == 'Valid':
            base_list=[]
            for gr in grouped_validdata['f_group_%s' % self.modify_var].unique():
                for ti in grouped_validdata[self.timeid_train].unique():
                    base_list.append({'Group':gr, self.timeid_train:ti})
            base_df=pd.DataFrame(base_list)
            valid_pct = pd.DataFrame(
                grouped_validdata.groupby(['f_group_%s' % self.modify_var, self.timeid_train])[self.target].agg(
                    {'count', 'sum'}).reset_index())
            total = pd.DataFrame(
                grouped_validdata.groupby([self.timeid_train])[self.target].agg({'count'}).reset_index().rename(
                    {'count': 'total_num'}, axis=1))
            final = pd.merge(total, valid_pct, how='outer', on=self.timeid_train)
            final['popu_pct'] = final['count'] / final['total_num']
            final['bad_rate'] = final['sum'] / final['count']
            final['Group'] = final['f_group_%s' % self.modify_var]
            final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_train])
            node_type = 'train'
        elif (select_data == 'Reject')& (self.modify_var in (self.varchar_reject+self.varnum_reject)):
            base_list=[]
            for gr in grouped_rejectdata['f_group_%s' % self.modify_var].unique():
                for ti in grouped_rejectdata[self.timeid_reject].unique():
                    base_list.append({'Group':gr, self.timeid_reject:ti})
            base_df=pd.DataFrame(base_list)
            if self.target_reject != None:
                valid_pct = pd.DataFrame(
                    grouped_rejectdata.groupby(['f_group_%s' % self.modify_var, self.timeid_reject])[
                        self.target_reject].agg({'count', 'sum'}).reset_index())
                total = pd.DataFrame(grouped_rejectdata.groupby([self.timeid_reject])[self.target_reject].agg(
                    {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_reject)
                final['popu_pct'] = final['count'] / final['total_num']
                final['bad_rate'] = final['sum'] / final['count']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_reject])
                node_type = 'reject'
            else:
                valid_pct = pd.DataFrame(
                    grouped_rejectdata.groupby(['f_group_%s' % self.modify_var, self.timeid_reject])[
                        'f_group_%s' % self.modify_var].agg({'count'}).reset_index())
                total = pd.DataFrame(
                    grouped_rejectdata.groupby([self.timeid_reject])['f_group_%s' % self.modify_var].agg(
                        {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_reject)
                final['popu_pct'] = final['count'] / final['total_num']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_reject])
                node_type = 'reject'
        elif (select_data == 'OOT')& (self.modify_var in (self.varchar_oot+self.varnum_oot)):
            base_list=[]
            for gr in grouped_ootdata['f_group_%s' % self.modify_var].unique():
                for ti in grouped_ootdata[self.timeid_oot].unique():
                    base_list.append({'Group':gr,self.timeid_oot:ti})
            base_df=pd.DataFrame(base_list)
            if self.target_oot != None:
                valid_pct = pd.DataFrame(
                    grouped_ootdata.groupby(['f_group_%s' % self.modify_var, self.timeid_oot])[self.target_oot].agg(
                        {'count', 'sum'}).reset_index())
                total = pd.DataFrame(
                    grouped_ootdata.groupby([self.timeid_oot])[self.target_oot].agg({'count'}).reset_index().rename(
                        {'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_oot)
                final['popu_pct'] = final['count'] / final['total_num']
                final['bad_rate'] = final['sum'] / final['count']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_oot])
                node_type = 'oot'
            else:
                valid_pct = pd.DataFrame(
                    grouped_ootdata.groupby(['f_group_%s' % self.modify_var, self.timeid_oot])['f_group_%s' % self.modify_var, ].agg(
                        {'count'}).reset_index())
                total = pd.DataFrame(grouped_ootdata.groupby([self.timeid_oot])['f_group_%s' % self.modify_var].agg(
                    {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final = pd.merge(total, valid_pct, how='outer', on=self.timeid_oot)
                final['popu_pct'] = final['count'] / final['total_num']
                final['Group'] = final['f_group_%s' % self.modify_var]
                final = pd.merge(base_df, final, how='outer', on=['Group', self.timeid_oot])
                node_type = 'oot'
        else:
            final = pd.DataFrame()
            node_type = 'no'
        print(bad_pct)
        bad_pct=pd.merge(bad_pct,total_pct[['Group']],how='outer',on='Group')
        # bad_pct=bad_pct.fillna(0)
        # total_pct=total_pct.fillna(0)
        # final=final.fillna(0)

        return bad_pct, total_pct, final, node_type
    def plot_group(self, bad_pct, total_pct, final, node_type):
        if node_type=='train':
            time=self.timeid_train
        elif node_type=='valid':
            time=self.timeid_train
        elif node_type =='reject':
            time=self.timeid_reject
        elif node_type=='oot':
            time=self.timeid_oot
        else:
            time='NO'
        good = LabelFrame(self.top_modify, text='样本稳定性')
        self.comboxlist_sample_select = ttk.Combobox(good, width=15)
        value = []
        if self.flag_timeid_train == True:
            value.append('Trian')
        if self.vaild_flag==True:
            value.append('Valid')
        if self.flag_timeid_oot == True:
            value.append('OOT')
        if self.flag_timeid_reject == True:
            value.append('Reject')
        value.append('NO')
        self.comboxlist_sample_select["value"] = value
        for i in range(len(value)):
            if value[i] == self.fist_value:
                self.comboxlist_sample_select.current(i)
        if self.result_show==False:
            self.comboxlist_sample_select.pack(side=TOP, anchor='ne')
            self.comboxlist_sample_select.bind("<<ComboboxSelected>>",
                                               lambda event, ac='refrsh_pic': self.create_modify(ac))
        canvasg = tk.Canvas()
        try:
            plt.close()
        except:
            pass
        fig = plt.figure()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=1, hspace=None)
        ax1 = fig.add_subplot(1, 2, 1)
        x = total_pct['Group']
        y1 = total_pct['pct_train']
        bar_width = 0.2
        plt.bar(x=range(len(x)), height=y1, label='Train', color='steelblue', alpha=0.8, width=bar_width)
        if 'pct_valid' in list(total_pct.columns):
            y2 = total_pct['pct_valid']
            plt.bar(x=np.arange(len(x)) + bar_width, height=y2, label='valid', color='indianred', alpha=0.8,
                    width=bar_width)
        if 'pct_reject' in list(total_pct.columns):
            y3 = total_pct['pct_reject']
            plt.bar(x=np.arange(len(x)) + 2 * bar_width, height=y3, label='reject', color='powderblue', alpha=0.8,
                    width=bar_width)
        if 'pct_oot' in list(total_pct.columns):
            y4 = total_pct['pct_oot']
            plt.bar(x=np.arange(len(x)) + 3 * bar_width, height=y4, label='oot', color='pink', alpha=0.8,
                    width=bar_width)

        plt.legend()
        # 为两条坐标轴设置名称
        plt.xlabel(u"组")
        plt.ylabel(u"占比")
        badrate1 = bad_pct['badrate_train']
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(x)), badrate1, 'steelblue')
        if ('badrate_valid', '') in list(bad_pct.columns):
            badrate2 = bad_pct['badrate_valid']
            ax2.plot(np.arange(len(x)) + bar_width, badrate2, 'indianred')
        if ('badrate_reject', '') in list(bad_pct.columns):
            badrate3 = bad_pct['badrate_reject']
            ax2.plot(np.arange(len(x)) + bar_width * 2, badrate3, 'powderblue')
        if ('badrate_oot', '') in list(bad_pct.columns):
            badrate4 = bad_pct['badrate_oot']
            ax2.plot(np.arange(len(x)) + bar_width * 3, badrate4, 'pink')
        ax2.set_ylabel('坏账率')
        plt.xticks(np.arange(len(x)) + 0.15, x, fontsize=12, rotation=20)

        if node_type != 'no':
            ax3 = fig.add_subplot(1, 2, 2)
            bar_width = 0.1
            i = 0
            group_list = list(final['Group'].unique())
            group_list.sort(reverse=False)
            for group in group_list:
                tt = final[final['Group'] == group]
                y2 = tt['popu_pct']
                x = tt[time]
                plt.bar(x=np.arange(len(x)) + 0.08 * i, height=y2, label="组_%s" % group,  alpha=0.8,
                        width=bar_width)
                i = i + 1
            # 设置标题

            plt.legend()
            # 为两条坐标轴设置名称
            plt.xlabel("Time")
            plt.ylabel("样本占比")

            plt.xticks(np.arange(len(x)) + 0.15, x, fontsize=12, rotation=20)
            if 'bad_rate' in list(tt.columns):
                ax4 = ax3.twinx()  # this is the important function
                i = 0
                group_list = list(final['Group'].unique())
                group_list.sort(reverse=False)
                for group in group_list:
                    tt = final[final['Group'] == group]
                    y = tt['bad_rate']
                    x = tt[time]
                    ax4.plot(np.arange(len(x)) + 0.08 * i, y)
                    i = i + 1
                ax4.set_ylabel('坏账率')

        canvasg = FigureCanvasTkAgg(fig, good)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        good.pack(side=BOTTOM, fill='both', expand='yes')
    def modify_combo(self, event):
        # 粗分组改变
        combo_selection = self.comboxlist_modify_f_group.get()
        self.comboxlist_modify_f_group.destroy()
        if self.modify_var in self.varnum:

            self.mody = self.modify_f_group(s_group=self.group_info_modify_var, s_g=self.modify_s_group_num,
                                            input_group=combo_selection)
            self.group_info_modi = self.mody.drop(columns='f_group').rename({'f_group_modify': 'f_group'}, axis=1)
            self.group_info_modi['f_group'] = self.group_info_modi['f_group'].astype('int')
            self.group_info_modify_var, self.s_group_report_modify_var, self.f_group_report_modify_var = self.f_group_modify_recalculate(
                group_info_modi=self.group_info_modi, col=self.modify_var, validation=self.vaild_flag)
        else:

            self.group_info_modi = self.group_info_modify_var
            self.group_info_modi.loc[
                self.group_info_modi['value'] == self.modify_s_group_num, 'f_group'] = combo_selection
            self.group_info_modi['f_group'] = self.group_info_modi['f_group'].astype('int')
            self.group_info_modify_var, self.s_group_report_modify_var, self.f_group_report_modify_var = self.f_group_modify_recalculate(
                group_info_modi=self.group_info_modi, col=self.modify_var, validation=self.vaild_flag)
        self.detail_modify_var, self.summary_modify_var, self.modify_modify_var = self.renamedata(
            s_group_data=self.s_group_report_modify_var, f_group_data=self.f_group_report_modify_var,
            validation=self.vaild_flag)
        # 刷新当前更改
        self.create_modify('refresh_group')
    def s_group_modify(self, event):
        # 系分组改变
        try:
            value = float(self.E1.get())
            if (self.s_max > value) & (self.s_min < value):
                self.L1 = Label(self.top_s_group_modify, text="Running.....")
                self.L1.grid()
                self.top_s_group_modify.update()

                group_info_var, grouped_data = group_func.numericexist(inputdata=self.train_data, col=self.modify_var,
                                                                       group_info_old=self.group_info_modify_var
                                                                       , target=self.target, modify=True,
                                                                       add_value=value, data_only=False)

                group_info_var['miss_rate'] = group_info_var['miss_count'] / (
                        group_info_var['miss_count'] + group_info_var['count'])

                group_info_var['total_count'] = (group_info_var['miss_count'] + group_info_var['count'])

                self.group_info_modify_var, self.s_group_report_modify_var, self.f_group_report_modify_var = self.f_group_modify_recalculate(
                    group_info_modi=group_info_var, validation=self.vaild_flag, col=self.modify_var)

                self.detail_modify_var, self.summary_modify_var, self.modify_modify_var = self.renamedata(
                    s_group_data=self.s_group_report_modify_var, f_group_data=self.f_group_report_modify_var,
                    validation=self.vaild_flag)
                self.flag_s_needsave = True
                self.top_s_group_modify.destroy()
                # 刷新当前更改
                self.create_modify('refresh_group')
            else:
                tk.messagebox.showwarning('错误', "错误：超出区间数值")
        except Exception as e:
            tk.messagebox.showwarning('错误', "错误：请输入数值 %s" % e)
    def modify_f_group_list(self, s_group, s_g):
        if (s_g > s_group[s_group['miss_s'] == False]['s_group'].min()) & (
                s_g < s_group[s_group['miss_s'] == False]['s_group'].max()):
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            up_g = s_group[s_group['s_group'] == s_g + 1]['f_group'].values[0]
            down_g = s_group[s_group['s_group'] == s_g - 1]['f_group'].values[0]

            if (up_g != f_g) & (down_g != f_g):
                list_f = [up_g, down_g]
            elif (up_g == f_g) & (down_g == f_g):
                list_f = ['new group']
            elif (up_g == f_g) & (down_g != f_g):
                list_f = ['new group', down_g]
            else:
                list_f = ['new group', up_g]
        elif s_g == s_group[s_group['miss_s'] == False]['s_group'].min():
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            up_g = s_group[s_group['s_group'] == s_g + 1]['f_group'].values[0]
            if f_g == up_g:
                list_f = ['new group']
            else:
                list_f = [up_g]
        elif s_g == s_group[s_group['miss_s'] == False]['s_group'].max():
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            down_g = s_group[s_group['s_group'] == s_g - 1]['f_group'].values[0]
            if f_g == down_g:
                list_f = ['new group']
            else:
                list_f = [down_g]
        elif s_group[s_group['s_group'] == s_g]['miss_s'].values[0] == True:
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            if len(s_group[s_group['f_group'] == f_g]) > 1:
                list_f = list(s_group['f_group'].unique())
                list_f.append(min(list_f) - 1)
            else:
                list_f = list(s_group['f_group'].unique())
        return list_f
    def modify_f_group(self, s_group, s_g, input_group):
        modify_s_group = s_group.copy()
        modify_s_group['f_group_modify'] = modify_s_group['f_group']
        if (s_g > s_group[s_group['miss_s'] == False]['s_group'].min()) & (
                s_g < s_group[s_group['miss_s'] == False]['s_group'].max()):
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            up_g = s_group[s_group['s_group'] == s_g + 1]['f_group'].values[0]
            down_g = s_group[s_group['s_group'] == s_g - 1]['f_group'].values[0]

            if input_group == 'new group':

                if (up_g != f_g) & (down_g != f_g):
                    pass
                elif (up_g == f_g) & (down_g == f_g):
                    modify_s_group.loc[modify_s_group['s_group'] == s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] + 1
                    modify_s_group.loc[modify_s_group['s_group'] > s_g, 'f_group_modify'] = modify_s_group[
                                                                                                'f_group'] + 2
                elif (up_g == f_g) & (down_g != f_g):
                    modify_s_group.loc[modify_s_group['s_group'] > s_g, 'f_group_modify'] = modify_s_group[
                                                                                                'f_group'] + 1
                else:
                    modify_s_group.loc[modify_s_group['s_group'] >= s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] + 1
            elif int(input_group) == up_g:

                if (up_g != f_g) & (down_g != f_g):
                    modify_s_group.loc[modify_s_group['s_group'] > s_g, 'f_group_modify'] = modify_s_group[
                                                                                                'f_group'] - 1
                elif (up_g == f_g) & (down_g == f_g):
                    pass
                elif (up_g == f_g) & (down_g != f_g):
                    pass
                else:
                    modify_s_group.loc[modify_s_group['s_group'] == s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] + 1
            else:

                if (up_g != f_g) & (down_g != f_g):
                    modify_s_group.loc[modify_s_group['s_group'] >= s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] - 1
                elif (up_g == f_g) & (down_g == f_g):
                    pass
                elif (up_g == f_g) & (down_g != f_g):
                    modify_s_group.loc[modify_s_group['s_group'] == s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] - 1
                else:
                    pass

        elif s_g == s_group[s_group['miss_s'] == False]['s_group'].min():
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            up_g = s_group[s_group['s_group'] == s_g + 1]['f_group'].values[0]
            if input_group == 'new group':
                if f_g == up_g:
                    modify_s_group.loc[modify_s_group['s_group'] > s_g, 'f_group_modify'] = modify_s_group[
                                                                                                'f_group'] + 1
                else:
                    pass
            else:
                if f_g == up_g:
                    pass
                else:
                    modify_s_group.loc[modify_s_group['s_group'] > s_g, 'f_group_modify'] = modify_s_group[
                                                                                                'f_group'] - 1

        elif s_g == s_group[s_group['miss_s'] == False]['s_group'].max():
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            down_g = s_group[s_group['s_group'] == s_g - 1]['f_group'].values[0]
            if input_group == 'new group':
                if f_g == down_g:
                    modify_s_group.loc[modify_s_group['s_group'] == s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] + 1
                else:
                    pass
            else:
                if f_g == down_g:
                    pass
                else:
                    modify_s_group.loc[modify_s_group['s_group'] == s_g, 'f_group_modify'] = modify_s_group[
                                                                                                 'f_group'] - 1
        elif s_group[s_group['s_group'] == s_g]['miss_s'].values[0] == True:
            f_g = s_group[s_group['s_group'] == s_g]['f_group'].values[0]
            modify_s_group.loc[modify_s_group['s_group'] == s_g, 'f_group_modify'] = int(input_group)
        return modify_s_group
    def f_group_modify_recalculate(self, group_info_modi, validation, col):

        if col in self.varnum:
            varnum = [col]
            varchar = []
        else:
            varchar = [col]
            varnum = []

        base = group_info_modi[['count', 'max', 'mean', 'min',
                                'miss_count', 'miss_s', 's_Bad_rate', 's_N_bad', 's_N_obs',
                                's_group', 's_max', 's_min', 'value',
                                'variable_name', 'miss_rate', 'total_count', 'f_group']]

        f_data = group_info_modi.groupby('f_group').agg(
            {'s_N_bad': 'sum', 's_N_obs': 'sum', 's_max': 'max', 's_min': 'min', 'miss_s': 'max'}
        ).reset_index().rename(
            {'s_N_bad': 'f_N_bad', 's_N_obs': 'f_N_obs', 's_max': 'f_max', 's_min': 'f_min', 'miss_s': 'miss_f'},
            axis=1)

        total_bad = f_data['f_N_bad'].sum()
        total_good = f_data['f_N_obs'].sum() - f_data['f_N_bad'].sum()

        f_data['woe'] = f_data.apply(lambda x: math.log(
            (max(1, x['f_N_bad']) / total_bad) / (max(1, (x['f_N_obs'] - x['f_N_bad'])) / total_good)), axis=1)

        f_data['iv_g'] = ((f_data['f_N_bad'] / total_bad) - ((f_data['f_N_obs'] - f_data['f_N_bad']) / total_good)) * \
                         f_data['woe']
        iv = f_data['iv_g'].sum()
        f_data['iv'] = iv
        f_data['f_Bad_rate'] = f_data['f_N_bad'] / f_data['f_N_obs']
        group_info_modify_var = pd.merge(base, f_data, how='left', on='f_group')
        self.check1 = group_info_modify_var
        self.check2 = varnum
        self.check3 = varchar
        s_group_report_modify_var, f_group_report_modify_var = binning.report(group_info=group_info_modify_var,
                                                                              varnum=varnum, varchar=varchar)

        if validation == True:
            grouped_valid_data = binning.fit_bin_existing(data=self.valid_data, target=self.target, data_only=True,
                                                          varnum=varnum, varchar=varchar,
                                                          group_info=group_info_modify_var,
                                                          n_job=1)

            tt1 = grouped_valid_data.groupby(['f_group_%s' % col]).agg(
                {self.target: ['mean', 'count', 'sum']}).reset_index().rename(
                {'mean': 'badrate_vaild', 'count': 'n_vaild', 'sum': 'bad_n_vaild'}, axis=1)
            tt1['variable_name'] = col

            temp = pd.DataFrame()
            temp['variable_name'] = tt1['variable_name']
            temp['badrate_vaild'] = tt1[self.target]['badrate_vaild']
            temp['n_vaild'] = tt1[self.target]['n_vaild']
            temp['bad_n_vaild'] = tt1[self.target]['bad_n_vaild']
            temp['f_group'] = tt1['f_group_%s' % col]

            dd = pd.merge(f_group_report_modify_var[f_group_report_modify_var['variable_name'] == col], temp,
                          how='outer',
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

            s_group_report_modify_var = pd.merge(s_group_report_modify_var, dd, how='left',
                                                 on=['variable_name', 'f_group'])
            f_group_report_modify_var = pd.merge(f_group_report_modify_var, dd, how='left',
                                                 on=['variable_name', 'f_group'])

        return group_info_modify_var, s_group_report_modify_var, f_group_report_modify_var
    def save_modify_var(self, event):
        # 更新数据及更改
        self.group_info_modify = self.group_info_modify[
            self.group_info_modify['variable_name'] != self.modify_var]
        self.group_info_modify = self.group_info_modify.append(self.group_info_modify_var)

        self.summary_modify = self.summary_modify[
            self.summary_modify['变量名称'] != self.modify_var]
        self.summary_modify = self.summary_modify.append(self.summary_modify_var)
        if self.vaild_flag==True:
            self.summary_modify = self.summary_modify[['变量名称', '信息熵', '缺失率', '样本psi']].drop_duplicates()
        else:
            self.summary_modify = self.summary_modify[['变量名称', '信息熵', '缺失率']].drop_duplicates()
        self.detail_modify = self.detail_modify[
            self.detail_modify['变量名称'] != self.modify_var]
        self.detail_modify = self.detail_modify.append(self.detail_modify_var)
        if self.vaild_flag==True:
            self.detail_modify = self.detail_modify[['变量名称', '注释', '粗分组样本数', '组编号', '信息熵',
                                   '粗分组事件率', '粗分组事件数', '缺失率', 'woe', '样本psi', '事件psi']]
        else:
            self.detail_modify = self.detail_modify[['变量名称', '注释', '粗分组样本数', '组编号', '信息熵',
                                                     '粗分组事件率', '粗分组事件数', '缺失率', 'woe']]
        
        self.modify_modify = self.modify_modify[
            self.modify_modify['变量名称'] != self.modify_var]
        self.modify_modify = self.modify_modify.append(self.modify_modify_var)
        self.modify_modify = self.modify_modify[['变量名称', "细分组编号", '注释', '粗分组编号', '信息熵',
                                                 '粗分组事件率', '细分组事件率', '细分组样本数', '细分组事件数',
                                                 '粗分组事件数', '粗分组样本数', '值']]

        if str(self.comboxlist_variable_use.get()) != '使用':
            self.not_use.append(self.modify_var)
            list_t = list(set(self.not_use))
        else:
            list_t = list(set(self.not_use))
            try:
                list_t.remove(self.modify_var)
            except:
                pass
            self.not_use = list_t
        self.flag_s_needsave = False