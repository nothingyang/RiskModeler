import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
import pickle as pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib_venn import venn2
import os

class sample():
    def __init__(self, mainframe, project_info):
        self.master = mainframe
        self.project_info = project_info
        self.project_path = os.path.split(project_info[project_info['模块类型'] == 'project']['保存地址'][0])[0]
        self.train_data_list=[]
        self.method = ''
        self.train_pct = 0.7
        self.valid_pct = 0.00001
        self.seed = 123456
        self.bad_pct = 1
        self.sample_flag = '否'
        self.replace = 'False'
        self.bad_rate = 0.1
        self.par_train_data = pd.DataFrame()
        self.node_name = 'sample'
        self.exist_data = list(project_info['模块名字'])
        self.load = 'N'
        self.target = 'Y'
        self.save = 'N'

    def load_node(self, node_data, ac):
        # 重新进去页面
        self.load = 'Y'
        self.node_name = node_data[0]['node_name']
        self.method = node_data[0]['method']
        self.replace = node_data[0]['replace']
        self.seed = node_data[0]['seed']
        self.train_pct = node_data[0]['train_pct']
        self.data_role=node_data[0]['data_role']
        self.sample_flag = node_data[0]['adjuest_bad']
        self.bad_pct = node_data[0]['bad_sample_pct']
        self.bad_rate = node_data[0]['sample_bad_rate']
        self.previous_node_name=node_data[0]['previous_node_name']
        self.previous_node_time = node_data[0]['previous_node_time']
        self.check_list=node_data[0]['check_change']
        self.par_traindatavariable_setting=node_data[0]['data_variable_setting']
        self.trainpart_data = node_data[1]
        self.par_train_data =node_data[2]


        if ac=='setting':
            path_list = self.project_info[self.project_info['创建时间'] == self.previous_node_time]['保存地址']
            def continu(event):
                for child in self.master.winfo_children():
                    child.destroy()
                self.ui_start()
            def back(event):
                self.master.destroy()
            if len(path_list)==0:
                self.master.title('提示')
                L00 = Label(self.master, width=80, text="该模块引用的%s （创建于 %s)模块 没有在项目中找到，\n可能该模块已经更新，删除，"
                                                        "或未导入\n继续设置可能会导致以前结果丢失" %(self.previous_node_name,self.previous_node_time))
                L00.grid(column=0, row=0, columnspan=3, sticky=(W))
                button_contin = ttk.Button(self.master, text='继续设置')
                button_contin.grid(column=0, row=1, sticky=(W), padx=10, pady=10)
                button_contin.bind("<Button-1>", continu)
                button_back = ttk.Button(self.master, text='返回')
                button_back.grid(column=2, row=1, sticky=(W), padx=10, pady=10)
                button_back.bind("<Button-1>", back)
            else:
                path=path_list[0]
                try:
                    fr = open(path,'rb')
                    node_info = pickle.load(fr)
                    fr.close()
                    self.par_traindatavariable_setting = node_info[0]['data_variable_setting']
                    self.previous_check_change=node_info[0]['check_change']
                    self.previous_node_usedlist=node_info[0]['use_node']
                    self.previous_node_name=node_info[0]['node_name']
                    self.data_role=node_info[0]['data_role']
                    self.par_train_data=node_info[1]
                    self.ui_start()
                except Exception as e:
                    self.master.title('提示')
                    L00 = Label(self.master, width=80, text="导入%s （创建于 %s)模块 发生错误，\n可能该模块已经被破坏或删除，"
                                                            "\n%s" % (self.previous_node_name, self.previous_node_time,e))
                    L00.grid(column=0, row=0, columnspan=3, sticky=(W))
                    button_contin = ttk.Button(self.master, text='继续设置')
                    button_contin.grid(column=0, row=1, sticky=(W), padx=10, pady=10)
                    button_contin.bind("<Button-1>", continu)
                    button_back = ttk.Button(self.master, text='返回')
                    button_back.grid(column=2, row=1, sticky=(W), padx=10, pady=10)
                    button_back.bind("<Button-1>", back)
        else:
            self.result_ui(self.master,ac='re')

        # n = 0
        # for check_item in self.check_list:
        #     # 为了读取失败以后可以进入设置页面
        #     try:
        #         path = self.project_info[self.project_info['模块名字'] == check_item['node_name']]['保存地址'][0]
        #         fr = open(path, 'rb')
        #         node_info = pickle.load(fr)
        #         fr.close()
        #         if node_info[0]['time'] != check_item['node_time']:
        #             tk.messagebox.showwarning('错误', '之前引用的%s已经发生改变 \n 请重新设置更新结果' %check_item)
        #             #为了重新设置更新结果的时候不用再选一遍数据集了
        #             self.par_traindatavariable_setting = node_info[0]['data_variable_setting']
        #             self.previous_check_change=node_info[0]['check_change']
        #             self.previous_node_usedlist=node_info[0]['use_node']
        #             self.previous_node_name = node_info[0]['node_name']
        #             self.data_role=node_info[0]['data_role']
        #             self.par_train_data = node_info[1]
        #
        #             n = n + 1
        #         elif check_item['node_name'] == self.previous_node_name:
        #             self.par_traindatavariable_setting = node_info[0]['data_variable_setting']
        #             self.previous_check_change=node_info[0]['check_change']
        #             self.previous_node_usedlist=node_info[0]['use_node']
        #             self.previous_node_name = node_info[0]['node_name']
        #             self.data_role=node_info[0]['data_role']
        #             self.par_train_data = node_info[1]
        #         else:
        #             pass
        #     except Exception as e:
        #         n=n+1
        #         tk.messagebox.showwarning('错误', '%s error: %s' %(check_item['node_name'],e))
        # if n > 1:
        #     self.ui_start()
        # else:
        #     if ac == 'setting':
        #         self.ui_start()
        #     else:
        #         self.result_ui(self.master, ac='re')
    def pre_data(self):
        dd = list(self.project_info[(self.project_info['模块类型']=='DATA')&(self.project_info['状态']=='Good')]['保存地址'])
        for add in dd:
            try:
                fr = open(add, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                data_name = node_info[0]['data_name']
                self.train_data_list.append(data_name)
            except Exception as e:
                tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (add, e))
    def ui_start(self):
        # 初始页面设置
        if self.train_data_list==[]:
            self.pre_data()
        self.start_window_base = self.master
        width = self.master.winfo_screenwidth() * 0.15
        height = self.master.winfo_screenheight() * 0.3
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()
        self.start_window_base.geometry(
            '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))
        self.start_window_base.title('样本抽样参数设置')
        # 参数设置
        self.start_window_data = LabelFrame(self.start_window_base , text='参数设置')

        L00 = Label(self.start_window_data, width=20, text="名称")
        L00.grid(column=0, row=0, sticky=(W))
        if self.load == 'N':
            nodename = tk.StringVar(value=self.node_name)
            self.entry_node_name = Entry(self.start_window_data, textvariable=nodename, bd=1, width=18)
            self.entry_node_name.grid(column=1, row=0, sticky=(W))
        else:
            L01 = Label(self.start_window_data, width=20, text=self.node_name, bd=2)
            L01.grid(column=1, row=0, sticky=(W))
        L0 = Label(self.start_window_data, width=20, text="原始样本")
        L0.grid(column=0, row=1, sticky=(W))
        self.comboxlist_train_data = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_train_data["value"] = self.train_data_list
        #因为页面要根据选的数据集重新刷新所以要加一个数据集的判定
        if (self.load=='Y')|(self.par_train_data.empty==False):
            for i in range(len(self.train_data_list)):
                if self.train_data_list[i] == self.previous_node_name:
                    self.comboxlist_train_data.current(i)
        self.comboxlist_train_data.bind("<<ComboboxSelected>>", lambda event: self.load_data(event, datatype='train'))
        self.comboxlist_train_data.grid(column=1, row=1, sticky=(W))

        L1 = Label(self.start_window_data, width=20, text="抽样方法")
        L1.grid(column=0, row=2, sticky=(W))
        self.comboxlist_method = ttk.Combobox(self.start_window_data, width=15)
        if self.target == 'Y':
            self.comboxlist_method["value"] = ['简单随机', '分层（好坏）']
            if self.method=='简单随机':
                self.comboxlist_method.current(0)
            else:
                self.comboxlist_method.current(1)
        else:
            self.comboxlist_method["value"] = ['简单随机']
        if self.load=='Y':
            if self.method=='简单随机':
                self.comboxlist_method.current(0)
            else:
                try:
                    self.comboxlist_method.current(1)
                except:
                    self.comboxlist_method.current(0)
        else:
            self.comboxlist_method.current(0)

        self.comboxlist_method.grid(column=1, row=2, sticky=(W))

        L3 = Label(self.start_window_data, width=20, text="是否放回")
        L3.grid(column=0, row=3, sticky=(W))
        self.comboxlist_replace = ttk.Combobox(self.start_window_data, width=15)
        self.comboxlist_replace["value"] = ['False', 'True']

        if self.replace=='False':
            self.comboxlist_replace.current(0)
        else:
            self.comboxlist_replace.current(1)

        self.comboxlist_replace.grid(column=1, row=3, sticky=(W))

        L2 = Label(self.start_window_data, width=20, text="随机种子")
        L2.grid(column=0, row=4, sticky=(W))
        seed = tk.StringVar(value=self.seed)
        self.entry_seed = Entry(self.start_window_data, textvariable=seed, bd=1, width=18)
        self.entry_seed.grid(column=1, row=4, sticky=(W))
        self.entry_seed.bind('<Return>', lambda event: self.int_num_check(event, 'seed', 'int'))
        self.start_window_data.grid(column=0, row=0, columnspan=2, padx=10, pady=10)

        # 分配比例

        self.start_window_pct = LabelFrame(self.start_window_base, text='数据集抽样比例')
        L3 = Label(self.start_window_pct, width=20, text="抽样样本比例")
        L3.grid(column=0, row=0, sticky=(W))
        train_pct = tk.StringVar(value=self.train_pct)
        self.entry_train_pct = Entry(self.start_window_pct, textvariable=train_pct, bd=1, width=18)
        self.entry_train_pct.grid(column=1, row=0, sticky=(W))
        self.entry_train_pct.bind('<Return>', lambda event: self.int_num_check(event, 'train_pct', 'g'))
        self.start_window_pct.grid(column=0, row=1, columnspan=2, padx=10, pady=10)

        # 是否调整样本坏账率

        self.start_window_sample = LabelFrame(self.start_window_base, text='设置样本坏账率')

        L3 = Label(self.start_window_sample, width=20, text="是否调整样本坏账率")
        L3.grid(column=0, row=0, sticky=(W))
        if self.target == 'Y':
            self.comboxlist_sample_flag = ttk.Combobox(self.start_window_sample, width=15)
            self.comboxlist_sample_flag["value"] = ['否', '是']
            if self.sample_flag=='是':
                self.comboxlist_sample_flag.current(1)
            else:
                self.comboxlist_sample_flag.current(0)
            self.comboxlist_sample_flag.grid(column=1, row=0, sticky=(W))

        L3 = Label(self.start_window_sample, width=20, text="坏样本抽样比例")
        L3.grid(column=0, row=1, sticky=(W))
        if self.target == 'Y':
            bad_pct = tk.StringVar(value=self.bad_pct)
            self.entry_bad_pct = Entry(self.start_window_sample, textvariable=bad_pct, bd=1, width=18)
            self.entry_bad_pct.grid(column=1, row=1, sticky=(W))
            self.entry_bad_pct.bind('<Return>', lambda event: self.int_num_check(event, 'bad_pct', 'g'))

        L4 = Label(self.start_window_sample, width=20, text="抽样后坏账率")
        L4.grid(column=0, row=2, sticky=(W))
        if self.target == 'Y':
            valid_pct = tk.StringVar(value=self.bad_rate)
            self.entry_bad_rate = Entry(self.start_window_sample, textvariable=valid_pct, bd=1, width=18)
            self.entry_bad_rate.grid(column=1, row=2, sticky=(W))
            self.entry_bad_rate.bind('<Return>', lambda event: self.int_num_check(event, 'badrate', 'pct'))
            self.start_window_sample.grid(column=0, row=2, columnspan=2, padx=10, pady=10)

        if self.load == 'N':
            self.button_setting_save = ttk.Button(self.start_window_base, text='保存 确认')
            self.button_setting_save.grid(column=0, row=3, sticky=(W), padx=10, pady=10)
            self.button_setting_save.bind("<Button-1>", self.check_all_setting)
        else:
            self.button_setting_save = ttk.Button(self.start_window_base, text='更新结果')
            self.button_setting_save.grid(column=0, row=3, sticky=(W), padx=10, pady=10)
            self.button_setting_save.bind("<Button-1>", self.check_all_setting)

    def load_data(self, event, datatype):
        # 读取数据
        try:

            if (datatype == 'train') & (len(self.comboxlist_train_data.get()) >= 1):
                path = self.project_info[self.project_info['模块名字'] == self.comboxlist_train_data.get()]['保存地址'][0]
                fr = open(path, 'rb')
                node_info = pickle.load(fr)
                fr.close()
                self.par_traindatavariable_setting = node_info[0]['data_variable_setting']
                self.previous_check_change=node_info[0]['check_change']
                self.previous_node_usedlist=node_info[0]['use_node']
                self.previous_node_name = node_info[0]['node_name']
                self.previous_node_time = node_info[0]['time']
                self.previous_node_path = path
                self.data_role=node_info[0]['data_role']
                self.par_train_data = node_info[1]
                self.get_par()
                if len(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']) == 0:
                    self.get_par()
                    self.target = 'N'
                    for child in self.master.winfo_children():
                        child.destroy()
                    self.ui_start()
                else:
                    self.target = 'Y'
                    self.get_par()
                    for child in self.master.winfo_children():
                        child.destroy()
                    self.ui_start()
            elif len(self.comboxlist_train_data.get()) <= 1:

                self.par_train_data = pd.DataFrame()

            else:

                pass
        except Exception as e:
            self.par_train_data = pd.DataFrame()
            tk.messagebox.showwarning('错误', "%s数据集导入错误：%s" % (self.comboxlist_train_data.get(), e))

    def split_function(self):
        # 如果index有重复则重新reindex
        if self.par_train_data.index.is_unique == False:

            self.par_train_data = self.par_train_data.reset_index(drop=True)
        else:
            pass
        # 是否调整坏账率进行抽样
        if self.sample_flag == '是':
            target = list(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
            bad_data = self.par_train_data[self.par_train_data[target] == 1]
            if self.bad_pct <= 1:

                bad_part = bad_data.sample(frac=self.bad_pct, replace=False, random_state=self.seed)
            else:

                bad_part = bad_data.sample(frac=self.bad_pct, replace=True, random_state=self.seed)
            bad_num = len(bad_part)
            good_num = int(round((bad_num / self.bad_rate) * (1 - self.bad_rate), 0))
            good_data = self.par_train_data[self.par_train_data[target] == 0]
            if good_num > len(good_data):

                good_part = good_data.sample(n=good_num, replace=True, random_state=self.seed)
            else:

                good_part = good_data.sample(n=good_num, replace=False, random_state=self.seed)
            or_data = pd.concat([good_part, bad_part])
            self.trainpart_data = or_data
        else:
            print('gggg')
            or_data = self.par_train_data
            # 简单随机
            if self.method == '简单随机':
                # 过抽样
                if ((self.train_pct > 1) | (self.valid_pct > 1)) & (self.replace == 'False'):
                    tk.messagebox.showwarning('错误', '由于样本比例大于1 \n 进行有放回抽样')
                    self.replace = 'True'
                # 不放回抽样
                if (self.train_pct + self.valid_pct == 1) & (self.replace == 'False'):

                    self.trainpart_data = or_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)

                elif (self.train_pct + self.valid_pct < 1) & (self.replace == 'False'):

                    self.trainpart_data = or_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                elif (self.train_pct + self.valid_pct > 1) & (self.replace == 'False'):

                    self.trainpart_data = or_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                else:

                    self.trainpart_data = or_data.sample(frac=self.train_pct, replace=True, random_state=self.seed)
            else:
                # 根据好坏进行抽样
                target = \
                list(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
                good_data = or_data[or_data[target] == 0]
                bad_data = or_data[or_data[target] == 1]
                if ((self.train_pct > 1) | (self.valid_pct > 1)) & (self.replace == 'False'):
                    tk.messagebox.showwarning('错误', '由于抽样比例大于1 \n 进行有放回抽样')
                    self.replace = True
                if (self.train_pct + self.valid_pct == 1) & (self.replace == 'False'):

                    # good
                    trainpart_gdata = good_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                    # bad
                    trainpart_bdata = bad_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                elif (self.train_pct + self.valid_pct < 1) & (self.replace == 'False'):

                    # good
                    trainpart_gdata = good_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                    # bad
                    trainpart_bdata = bad_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                elif (self.train_pct + self.valid_pct > 1) & (self.replace == 'False'):

                    # good
                    trainpart_gdata = good_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                    # bad
                    trainpart_bdata = bad_data.sample(frac=self.train_pct, replace=False, random_state=self.seed)
                else:

                    # good
                    trainpart_gdata = good_data.sample(frac=self.train_pct, replace=True, random_state=self.seed)
                    # bad
                    trainpart_bdata = bad_data.sample(frac=self.train_pct, replace=True, random_state=self.seed)
                self.trainpart_data = pd.concat([trainpart_gdata, trainpart_bdata])
        try:
            self.tt.destroy()
        except:
            pass
        result_main_frame = Toplevel(self.master)
        self.result_ui(result_main_frame, ac='setting')
        self.master.wait_window(result_main_frame)

    def result_ui(self, mainframe, ac):

        self.tt = mainframe
        if ac != 're':
            self.button_result_save = ttk.Button(self.tt, text='保存 确认')
            self.button_result_save.grid(column=0, row=0, sticky=(W), padx=10, pady=10)
            self.button_result_save.bind("<Button-1>", self.save_data)

            self.button_reset = ttk.Button(self.tt, text='重新分区')
            self.button_reset.grid(column=3, row=0, sticky=(W), padx=10, pady=10)
            self.button_reset.bind("<Button-1>", self.all_reset)
        # 展示结果
        if self.target == 'Y':
            target = list(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
            t_data = {"data": '原始数据集', "obs": self.par_train_data[target].count(),
                      "bad_num": self.par_train_data[target].sum(),
                      "bad_rate": str(round(self.par_train_data[target].mean() * 100, 2)) + '%'}
            t_trian_data = {"data": '抽样数据集', "obs": self.trainpart_data[target].count(),
                            "bad_num": self.trainpart_data[target].sum(),
                            "bad_rate": str(round(self.trainpart_data[target].mean() * 100, 2)) + '%'}

            t = []
            t.append(t_data)
            t.append(t_trian_data)

            summ = LabelFrame(self.tt, text='分区结果', width=20, height=20)
            tree = ttk.Treeview(summ, columns=["data", 'obs', 'bad_num', 'bad_rate'], show="headings",
                                selectmode=tk.BROWSE)
            tree.column("data", width=80, minwidth=80, stretch=tk.NO, anchor="center")
            tree.column("obs", width=80, minwidth=80, stretch=tk.NO, anchor="center")
            tree.column("bad_num", width=80, minwidth=80, stretch=tk.NO, anchor="center")
            tree.column("bad_rate", width=80, minwidth=80, stretch=tk.NO, anchor="center")

            tree.heading("data", text="数据集")
            tree.heading("obs", text="样本数")
            tree.heading("bad_num", text="坏客户数")
            tree.heading("bad_rate", text="坏账率")

            i = 0
            for v in t:
                tree.insert('', i, values=(v.get("data"), v.get("obs"), v.get("bad_num"), v.get("bad_rate")))
                i += 1

            tree.grid()
            summ.grid(column=0, row=1,columnspan=4, padx=8, pady=8)
        else:
            t_data = {"data": '原始数据集', "obs": len(self.par_train_data)}
            t_trian_data = {"data": '抽样数据集', "obs": len(self.trainpart_data)}

            t = []
            t.append(t_data)
            t.append(t_trian_data)

            summ = LabelFrame(self.tt, text='分区结果', width=20, height=20)
            tree = ttk.Treeview(summ, columns=["data", 'obs', 'bad_num', 'bad_rate'], show="headings",
                                selectmode=tk.BROWSE)
            tree.column("data", width=80, minwidth=80, stretch=tk.NO, anchor="center")
            tree.column("obs", width=80, minwidth=80, stretch=tk.NO, anchor="center")

            tree.heading("data", text="数据集")
            tree.heading("obs", text="样本数")

            i = 0
            for v in t:
                tree.insert('', i, values=(v.get("data"), v.get("obs")))
                i += 1

            tree.grid()
            summ.grid(column=0, row=1,columnspan=4, padx=8, pady=8)
        # 总样本
        la = LabelFrame(self.tt, text='总样本分布')
        t = set(self.trainpart_data.index)

        t2 = set(self.par_train_data.index)
        canvas = tk.Canvas()
        g = plt.figure(figsize=(4, 4))
        pp = venn2(subsets=[t, t2], set_labels=('train', 'total'), set_colors=('r', 'g'))
        canvas = FigureCanvasTkAgg(g, la)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        la.grid(column=4, row=1,columnspan=4, padx=8, pady=8)
        if self.target == 'Y':
            # 好样本
            target = list(self.par_traindatavariable_setting[self.par_traindatavariable_setting['变量角色'] == '目标']['变量名称'])[0]
            good = LabelFrame(self.tt, text='好样本分布', width=20, height=20)
            tg = set(self.trainpart_data[self.trainpart_data[target] == 0].index)

            tg2 = set(self.par_train_data[self.par_train_data[target] == 0].index)
            canvasg = tk.Canvas()
            gg = plt.figure(figsize=(4, 4))
            pp = venn2(subsets=[tg, tg2], set_labels=('train', 'total'), set_colors=('r', 'g'))
            canvasg = FigureCanvasTkAgg(gg, good)
            canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            good.grid(column=0, row=2,columnspan=4, padx=8, pady=8)
            # 坏样本
            bad = LabelFrame(self.tt, text='坏样本分布', width=20, height=20)
            tg = set(self.trainpart_data[self.trainpart_data[target] == 1].index)

            tg2 = set(self.par_train_data[self.par_train_data[target] == 1].index)
            canvasg = tk.Canvas()
            gg = plt.figure(figsize=(4, 4))
            pp = venn2(subsets=[tg, tg2], set_labels=('train', 'total'), set_colors=('r', 'g'))
            canvasg = FigureCanvasTkAgg(gg, bad)
            canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            bad.grid(column=4, row=2,columnspan=4, padx=8, pady=8)

    def all_reset(self, event):
        # 返回重新分区
        try:
            self.tt.destroy()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)

    def save_data(self, event):
        # 保存数据
        try:
            node_save_path = self.project_path + '/' + '%s.sampling' % self.node_name
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.node_setting = {'node_type': 'SAMPLE',
                                 'node_name': self.node_name,
                                 'node_save_path': node_save_path,
                                 'data_role': self.data_role,
                                 'method': self.method,
                                 'replace': self.replace,
                                 'seed': self.seed,
                                 'train_pct': self.train_pct,
                                 'target': self.target,
                                 'adjuest_bad': self.sample_flag,
                                 'bad_sample_pct': self.bad_pct,
                                 'sample_bad_rate': self.bad_rate,
                                 'time': nowTime,
                                 'check_change': [{'node_name': self.node_name,
                                                   'node_time': nowTime}] + self.previous_check_change,
                                 'data_variable_setting': self.par_traindatavariable_setting,
                                 'previous_node_name': self.previous_node_name,
                                 'previous_node_time': self.previous_node_time,
                                 'previous_node_path': [self.previous_node_path],
                                 'use_node': [self.node_name] + self.previous_node_usedlist}

            data_save = (self.node_setting, self.trainpart_data,self.par_train_data)
            error2 = Toplevel(self.master)
            screenwidth = self.master.winfo_screenwidth()
            screenheight = self.master.winfo_screenheight()

            error2.geometry('%dx%d+%d+%d' % (150, 100, (screenwidth - 150) / 2, (screenheight - 100) / 2))
            L2 = Label(error2, text="保存中")
            L2.grid()
            self.master.update()

            filename = node_save_path
            fw = open(filename, 'wb')
            pickle.dump(data_save, fw, 1)
            fw.close()

            self.save = 'Y'
            try:
                error2.destroy()
            except:
                pass
            self.master.destroy()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)

    # 检查所有变量参数是否正确
    def check_all_setting(self, event):
        try:
            self.load_data(event, 'train')
            self.get_par()
            if (self.node_name in self.exist_data) & (self.load == 'N'):
                tk.messagebox.showwarning('错误', "该名称已经被占用，请更改")
            else:
                e = 0
                if self.par_train_data.empty == True:
                    tk.messagebox.showwarning('错误', "错误：训练样本为空")
                else:
                    if self.target == 'Y':
                        total = ['seed', 'train_pct', 'bad_pct', 'bad_rate']
                    else:
                        total = ['seed', 'train_pct']
                    for p in total:
                        if p in ['seed']:
                            flag = 'int'
                            entry_p = p
                            er = self.int_num_check(event, entry_p, flag)
                        elif p in ['bad_rate']:
                            flag = 'pct'
                            entry_p = p
                            er = self.int_num_check(event, entry_p, flag)
                        else:
                            flag = 'g'
                            entry_p = p
                            er = self.int_num_check(event, entry_p, flag)
                        e = e + er

                    if e == 0:
                        self.split_function()
        except Exception as e:
            tk.messagebox.showwarning('错误', e)

    def get_par(self):
        # 更新得到的设置
        if self.target == 'Y':
            self.sample_flag = self.comboxlist_sample_flag.get()
            self.bad_pct = float(self.entry_bad_pct.get())
            self.bad_rate = float(self.entry_bad_rate.get())
        self.train_pct = float(self.entry_train_pct.get())
        self.seed = int(self.entry_seed.get())
        self.replace = self.comboxlist_replace.get()

        self.par_train_data = self.par_train_data
        self.method = self.comboxlist_method.get()
        try:
            self.node_name = self.entry_node_name.get()
        except:
            pass

    def int_num_check(self, event, entry_p, flag):
        # 检查数字是否正确
        flag_er = 0
        if entry_p == 'seed':
            inputnum = self.entry_seed.get()
            tip = '随机种子'
        elif entry_p == 'train_pct':
            inputnum = self.entry_train_pct.get()
            tip = '抽样样本比例'

        elif (self.target == 'Y') & (entry_p == 'bad_rate'):
            inputnum = self.entry_bad_rate.get()
            tip = '整体样本坏账率'
        elif (self.target == 'Y') & (entry_p == 'bad_pct'):
            inputnum = self.entry_bad_pct.get()
            tip = '坏样本抽样比例'
        else:
            pass

        try:
            if float(inputnum) <= 0:

                tk.messagebox.showwarning('错误', '%s:输入值不能小于等于0' % tip)
                flag_er = flag_er + 1
            else:
                if flag == 'int':
                    try:
                        int(inputnum)
                    except Exception as e:
                        tk.messagebox.showwarning('错误','%s:%s'%(tip, e))
                        flag_er = flag_er + 1
                elif flag == 'pct':
                    try:
                        num = float(inputnum)
                        if num > 1:
                            tk.messagebox.showwarning('错误', '%s:输入值不能大于1' % tip)
                            flag_er = flag_er + 1
                        else:
                            pass
                    except Exception as e:
                        tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
                        flag_er = flag_er + 1
                else:
                    try:
                        num = float(inputnum)
                    except Exception as e:
                        tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
                        flag_er = flag_er + 1
        except Exception as e:
            tk.messagebox.showwarning('错误', '%s:%s' % (tip, e))
            flag_er = flag_er + 1
        return flag_er
