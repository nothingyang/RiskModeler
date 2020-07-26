import tkinter as tk
from tkinter import *
import tkinter.messagebox
from tkinter import ttk
from tkinter.scrolledtext import *
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        hscrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=canvas.xview)

        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        def _bound_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)  

        def _unbound_to_mousewheel( event):
            canvas.bind_all("<MouseWheel>") 
#         canvas.bind('<Enter>', _unbound_to_mousewheel)
#         canvas.bind('<Leave>', _bound_to_mousewheel)

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set,xscrollcommand=hscrollbar.set)
        hscrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        


class score_result_ui():
    def __init__(self,mainframe,
                 predict_train_data,predict_score_data,
                 train_target,score_target,
                 train_time_id,score_time_id,
                 record_list, model,scorecarddf,f_group_report,model_var_type
                ):
        self.master=mainframe
        self.predict_train_data=predict_train_data
        self.predict_score_data=predict_score_data
        self.train_target=train_target
        self.score_target=score_target

        self.train_time_id=train_time_id
        self.score_time_id=score_time_id
        self.record_list =record_list
        self.model=model
        self.scorecarddf=scorecarddf
        self.f_group_report=f_group_report
        self.model_var_type=model_var_type
        self.plot_tab(
                 predict_train_data=self.predict_train_data,
                 train_target=self.train_target,
                 train_time_id=self.train_time_id,score_time_id=self.score_time_id,predict_score_data=self.predict_score_data,
                 record_list=self.record_list, model=self.model,scorecarddf=self.scorecarddf,model_var_type=self.model_var_type,score_target=self.score_target)

    def plot_tab(self,predict_train_data,train_target,train_time_id,record_list, model,scorecarddf,model_var_type,predict_score_data,score_time_id,score_target):
        # root=self.master
        self.master.title('评分卡')#标题
        self.screenwidth = self.master.winfo_screenwidth()
        self.screenheight = self.master.winfo_screenheight()
        width=self.screenwidth*0.85
        height=self.screenheight*0.75
        print(width,height)
        alignstr = '%dx%d+%d+%d' % (width, height, (self.screenwidth -width ) /2, (self.screenheight -height ) /2)
        self.master.geometry(alignstr)

        self.master.resizable(width = True, height = True)#窗口大小

        def help1():
            try:
                self.master.destroy()
            except:
                pass

        def help2():
            tkinter.messagebox.showinfo(title = 'suggestion',message = '请加好友QQ：***')

        menubar = Menu(self.master)
        filemenu = Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = '菜单',menu = filemenu)
        filemenu.add_command(label='关闭',command = help1)
        filemenu.add_command(label='输出报告',command = help2)


        tabcontrol = ttk.Notebook(self.master)
        tab1 = ttk.Frame(tabcontrol)
        tabcontrol.add(tab1, text = '模型信息')
        tab2 = ttk.Frame(tabcontrol)
        tabcontrol.add(tab2, text = '模型表现')#Frame控件的具体用法
        tab3 = ttk.Frame(tabcontrol)
        tabcontrol.add(tab3, text = '稳定性表现')#Frame控件的具体用法
        tabcontrol.pack(expand=1, fill="both")
        #第一页模型信息   

        def processp(plist):
            x=''
            for m in plist:
                t=str(m)+'\n\n'
                x=x+t
            return x
#第一页模型表现
#   ----------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
        process_bar_t=LabelFrame(tab1)
        scorecarddf['ave_score'] = scorecarddf.apply(lambda x: float(x['scorecard']) * float(x['pct_f_N_obs'].replace('%', '')) / 100,
                                   axis=1)
        av_df = scorecarddf.groupby('variable_name')['ave_score'].sum().reset_index().rename({'ave_score': 'average'}, axis=1)
        imp = pd.merge(scorecarddf, av_df, how='left', on='variable_name')
        imp['dif'] = abs(imp['scorecard'] - imp['average'])
        imp['dif_abs'] = imp.apply(lambda x: float(x['dif']) * float(x['pct_f_N_obs'].replace('%', '')) / 100, axis=1)
        show_df = imp.groupby('variable_name')['dif_abs'].sum().reset_index()
        show_df['im_pct'] = round(show_df['dif_abs'] / show_df['dif_abs'].sum(),2)
        def split_variable_name(x):
            i=0
            name=''
            while i< len(x):
                name+=x[i:i+20]
                name+='\n'
                i+=20
            return name
        show_df['name_split']=show_df['variable_name'].apply(lambda x: split_variable_name(x))
        show_df=show_df.sort_values(by='im_pct',ascending=False)
        try:
            plt.close()
        except:
            pass

        imp_gg = plt.figure(figsize=(min(max(20,len(show_df)*4),400),7),dpi=int((self.screenwidth/50)))
        plt.gcf().subplots_adjust(bottom=0.15)
        x = show_df['name_split']
        plt.bar(np.arange(len(x)) ,show_df['im_pct'],width=0.15)
        plt.xticks(np.arange(len(x)) , x, size=24)
        plt.tick_params(labelsize=24)
        plt.tight_layout()


        imp_var_base = LabelFrame(process_bar_t )
        imp_var_s = ScrollableFrame(imp_var_base)
        imp_var=LabelFrame(imp_var_s.scrollable_frame, text='变量重要性')
        imp_var_s.pack(side=BOTTOM, anchor='nw', fill=tk.BOTH, expand=YES)
        canvasg=FigureCanvasTkAgg(imp_gg,imp_var)
        canvasg.get_tk_widget().pack( expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        imp_var.pack(side=TOP, fill='both',expand=1, anchor='ne')
        imp_var_base.pack(side=TOP, fill='both',expand=1, anchor='ne')

        process_bar=LabelFrame(process_bar_t,text='变量筛选过程')
        textPad1 = ScrolledText(process_bar,width =85,height =60)
        textPad1.insert(tkinter.constants.END, chars = processp(record_list) )
        textPad1.pack(side=tk.TOP,fill='both',expand=YES)
        process_bar.pack(side=TOP, fill='both',expand=NO, anchor='ne')
        process_bar_t.pack(side=RIGHT, fill='both',expand=YES, anchor='ne')
        var_list=max([len(x) for x in scorecarddf["variable_name"]])
        final=LabelFrame(tab1,text='最终模型')
        textPad = ScrolledText(final,width =var_list+80,height =30)
        textPad.insert(tkinter.constants.END, chars = str(model.summary()) )
        textPad.pack(fill='x',expand=NO)
        final.pack(side=TOP, anchor='w',fill='x',expand=1,  padx=5, pady=5)



        final2=LabelFrame(tab1,text='评分卡')
        columns = ('variable_name', 'f_group', 'label', 'f_Bad_rate', 'pct_f_N_obs',
                   'woe',
               'coff', 'scorecard')
        tree = ttk.Treeview(final2, show = "headings", 
                            columns = columns, selectmode = tk.BROWSE,height=70)
        tree.column("variable_name", anchor = "w",minwidth=0,width=150, stretch=NO)
        tree.column( 'f_group', anchor = "center",minwidth=0,width=50, stretch=NO)
        tree.column('label', anchor = "w",minwidth=0,width=200, stretch=NO)
        tree.column('f_Bad_rate', anchor = "center",minwidth=0,width=50, stretch=NO)
        tree.column('pct_f_N_obs', anchor = "center",minwidth=0,width=50, stretch=NO)
        tree.column('woe', anchor = "center",minwidth=0,width=50, stretch=NO)
        tree.column('coff', anchor = "center",minwidth=0,width=50, stretch=NO)
        tree.column('scorecard', anchor = "center",minwidth=0,width=50, stretch=NO)
        tree.heading("variable_name", text = "变量名称")
        tree.heading( 'f_group', text = "分组")
        tree.heading('label', text = "注释")
        tree.heading('f_Bad_rate', text = "坏账率")
        tree.heading('pct_f_N_obs', text = "样本占比")
        tree.heading('woe', text = "WOE")
        tree.heading('coff', text = "系数")
        tree.heading('scorecard', text = "分数")
        scorecarddf.apply(lambda x: tree.insert('',1, values = (x['variable_name'], x['f_group'], x['label'], x['f_Bad_rate'], x['pct_f_N_obs'],x['woe'],x['coff'], x['scorecard'])),axis=1)
        tree.pack(side=TOP,fill=tk.BOTH,expand=NO)
        final2.pack(side=TOP, anchor='w',fill='both',expand=NO,  padx=5, pady=5)

        preds_t=self.predict_train_data[self.train_target] 
        labels_t=self.predict_train_data['SCORECARD_LR_p_1']
        #train------------------------------------




        #oot------------------------------------------------------
        if (self.predict_score_data.empty==True)|(self.score_target==None):
            preds_s = ()
            labels_s = ()
            flag_score_target = False
            flag_score_data = False
        else:
            preds_s=self.predict_score_data[self.score_target]
            labels_s=self.predict_score_data['SCORECARD_LR_p_1']
            flag_score_target=True
            flag_score_data=True
#第二页模型表现
# ----------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
 #下1       -----------------------------------------------------------------------------
        middle_frame=Frame(tab2,width=self.screenwidth*0.72, height=self.screenheight*0.27)
        middle_frame.pack_propagate(0)
        ks=LabelFrame(middle_frame,text='KS')
            
        ks_mp=self.PlotKS(preds_t=preds_t, labels_t=labels_t, n=100, asc=0,preds_s=preds_s, labels_s=labels_s,
               flag_score_data=flag_score_data, flag_score_target=flag_score_target)
        canvasg=FigureCanvasTkAgg(ks_mp,ks)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        ks.pack(side=LEFT, anchor='nw',fill=tk.BOTH,expand=YES,  padx=5, pady=5)
    
        cali_data=LabelFrame(middle_frame,text='分数校准数据')
        if (flag_score_target==True) & (flag_score_data==True):
            columns = ('s_group','SCORECARD_LR_p_1',score_target,'count','SCORECARD_LR_p_1_s',score_target+'_s','count_s')
        else:
            columns = ('s_group','SCORECARD_LR_p_1',score_target,'count')
        tree = ttk.Treeview(cali_data, show = "headings", 
                            columns = columns, selectmode = tk.BROWSE)
        tree.column('s_group', anchor = "center",minwidth=0,width=80, stretch=NO)
        tree.column( 'SCORECARD_LR_p_1', anchor = "center",minwidth=0,width=60, stretch=NO)
        tree.column(score_target, anchor = "center",minwidth=0,width=60, stretch=NO)
        tree.column('count', anchor = "center",minwidth=0,width=60, stretch=NO)
        if (flag_score_target==True) & (flag_score_data==True):
            tree.column( 'SCORECARD_LR_p_1_s', anchor = "center",minwidth=0,width=60, stretch=NO)
            tree.column(score_target+'_s', anchor = "center",minwidth=0,width=60, stretch=NO)
            tree.column('count_s', anchor = "center",minwidth=0,width=60, stretch=NO)
        tree.heading('s_group', text = "分数段")
        tree.heading( 'SCORECARD_LR_p_1', text = "Trian预测值")
        tree.heading(score_target, text = "Trian坏账率")
        tree.heading('count', text = "Trian样本数")
        if (flag_score_target==True) & (flag_score_data==True):
            tree.heading( 'SCORECARD_LR_p_1_s', text = "Score预测值")
            tree.heading(score_target+'_s', text = "Score预测值坏账率")
            tree.heading('count_s', text = "Score预测值样本数")
        m = self.cali(df=predict_train_data,df_s=predict_score_data,score='SCORECARD_LR_p_1' ,target=train_target,
                      flag_score_data=flag_score_data, flag_score_target=flag_score_target, score_target=score_target)

        m[0]['s_group']=m[0].apply(lambda x: str(round(x['s_min'],4))+'-'+str(round(x['s_max'],4)) , axis=1)
        m[0]['SCORECARD_LR_p_1']=round(m[0]['SCORECARD_LR_p_1'],4)
        m[0][train_target]=round(m[0][train_target],4)
        if (flag_score_target==True) & (flag_score_data==True):
            m[0]['SCORECARD_LR_p_1_s']=round(m[0]['SCORECARD_LR_p_1_s'],4)
            m[0][score_target+'_s']=round(m[0][score_target+'_s'],4)
        final=m[0].sort_values(by='s_group',ascending = False)
        if (flag_score_target==True) & (flag_score_data==True):
            #score和train一样或者不一样得时候会出错
            final.apply(lambda x: tree.insert('',0, values = (x['s_group'], x['SCORECARD_LR_p_1'], x[train_target], x['count'],x['SCORECARD_LR_p_1_s'], x['%s_s'%score_target], x['count_s'])),axis=1)
        else:
            final.apply(lambda x: tree.insert('',0, values = (x['s_group'], x['SCORECARD_LR_p_1'], x[train_target], x['count'])),axis=1)
        tree.pack(fill='both',expand=YES)
        cali_data.pack(side=LEFT, anchor='nw',fill='y',expand=NO,  padx=5, pady=5)

        pltcali=LabelFrame(middle_frame,text='分数校准')
        # m = cali(df,'SCORECARD_LR_p_1' ,model1.target_train)
        canvasg=FigureCanvasTkAgg(m[1],pltcali)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=RIGHT, fill=tk.BOTH, expand=1)
        pltcali.pack(side=RIGHT, anchor='w',fill='x',expand=YES,  padx=5, pady=5)


        middle_frame.pack(side=TOP, anchor='w',fill='x',expand=YES,  padx=5, pady=5)

 #下2       -----------------------------------------------------------------------------
        right_frame=Frame(tab2,width=self.screenwidth*0.72, height=self.screenheight*0.22)
        right_frame.pack_propagate(0)
        pltauc=LabelFrame(right_frame,text='AUC')


        auc_m=self.AUC(preds_t=preds_t, labels_t=labels_t, preds_s=preds_s, labels_s=labels_s,flag_score_data=flag_score_data, flag_score_target=flag_score_target)
        canvasg=FigureCanvasTkAgg(auc_m,pltauc)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        pltauc.pack(side=LEFT, anchor='nw',fill=tk.BOTH,expand=YES,  padx=5, pady=5)
    
        lift_da=LabelFrame(right_frame,text='Lift数据')
        if (flag_score_target==True) & (flag_score_data==True):
            columns = ('total_pct','lift','total_pct_s','lift_s')
        else:
            columns = ('total_pct','lift')
        tree = ttk.Treeview(lift_da, show = "headings", columns = columns, selectmode = tk.BROWSE)
        tree.column('total_pct', anchor = "center",minwidth=0,width=80, stretch=NO)
        tree.column( 'lift', anchor = "center",minwidth=0,width=80, stretch=NO)
        if (flag_score_target==True) & (flag_score_data==True):
            tree.column('total_pct_s', anchor = "center",minwidth=0,width=80, stretch=NO)
            tree.column( 'lift_s', anchor = "center",minwidth=0,width=80, stretch=NO)
        tree.heading('total_pct', text = "Train样本累计占比")
        tree.heading( 'lift', text = "Train提升率")
        if (flag_score_target==True) & (flag_score_data==True):
            tree.heading('total_pct_s', text = "Score样本累计占比")
            tree.heading( 'lift_s', text = "Score提升率")
        temp2=m[0].sort_values(by='s_group',ascending = False)
        temp2['total']=temp2['count'].cumsum(axis=0)
        temp2['bad']=round(temp2['count']*temp2[train_target],0)
        temp2['total_pct']=temp2['total']/temp2['count'].sum()
        temp2['lift']=temp2[train_target]/(temp2['bad'].sum()/temp2['count'].sum())
        temp2['total_pct']=round(temp2['total_pct'],2)
        temp2['lift']=round(temp2['lift'],2)
        if (flag_score_target==True) & (flag_score_data==True):
            temp2['total_s']=temp2['count_s'].cumsum(axis=0)
            temp2['bad_s']=round(temp2['count_s']*temp2[score_target+'_s'],0)
            temp2['total_pct_s']=temp2['total_s']/temp2['count_s'].sum()
            temp2['lift_s']=temp2[score_target+'_s']/(temp2['bad_s'].sum()/temp2['count_s'].sum())
            temp2['total_pct_s']=round(temp2['total_pct_s'],2)
            temp2['lift_s']=round(temp2['lift_s'],2)
  
        lift_data=temp2.sort_values(by='total_pct',ascending = False)
        if (flag_score_target==True) & (flag_score_data==True):
            lift_data.apply(lambda x: tree.insert('',0, values = (x['total_pct'], x['lift'],x['total_pct_s'], x['lift_s'])),axis=1)
        else:
            lift_data.apply(lambda x: tree.insert('',0, values = (x['total_pct'], x['lift'])),axis=1)
            
        tree.pack(fill='both',expand=YES)
        lift_da.pack(side=LEFT, anchor='nw',fill='y',expand=NO,  padx=5, pady=5)
        pltlift=LabelFrame(right_frame,text='Lift')

        try:
            plt.close()
        except:
            pass
        m=plt.figure()
        X=temp2['total_pct']
        Y=temp2['lift']#定义折线图的X，Y坐标
        plt.plot(X, Y,label='训练集') #折线图
        if (flag_score_target==True) & (flag_score_data==True):
            X_s=temp2['total_pct_s']
            Y_s=temp2['lift_s']#定义折线图的X，Y坐标
            plt.plot(X_s, Y_s,label='Score集') #折线图
        i=0 
        for a, b in zip(X, Y):
            if (i % 10 == 0) |(i==5):
                plt.text(a, b,'%.2f' %a+'--'+ '%.2f' %b , ha='center', va='bottom', fontsize=10)#每个点的数值
            i=i+1
        plt.legend()#显示每根折线的label
        plt.title('lift')#显示图名
        canvasg=FigureCanvasTkAgg(m,pltlift)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        pltlift.pack(side=RIGHT, anchor='w',fill='x',expand=YES,  padx=5, pady=5)
        right_frame.pack(side=TOP, fill=tk.BOTH,expand=YES,  padx=5, pady=5)

 #下3       -----------------------------------------------------------------------------
        bottom_frame=Frame(tab2,width=self.screenwidth*0.72, height=self.screenheight*0.22)
        bottom_frame.pack_propagate(0)
        pltdistri=LabelFrame(bottom_frame,text='分数分布')
        dis_m = self.plotdis(predict_train_data,train_target,'SCORECARD_LR_p_1')
        canvasg=FigureCanvasTkAgg(dis_m,pltdistri)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        pltdistri.pack(side=LEFT, anchor='nw',fill='y',expand=NO,  padx=5, pady=5)

    
        disgb=LabelFrame(bottom_frame,text='训练集分数区间分布')
        


        if self.score_target==None:
            score_target_flag=False
        else:
            score_target_flag=True

        auc_m= self.disallfb(df=self.predict_train_data,df_s=predict_score_data,score_target_flag=score_target_flag,
                                 train_target=self.train_target,score_target=score_target)
        canvasg=FigureCanvasTkAgg(auc_m,disgb)
        canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        disgb.pack(side=TOP, anchor='nw',fill=tk.BOTH,expand=YES,  padx=5, pady=5)
        bottom_frame.pack(side=TOP, fill=tk.BOTH,expand=YES,  padx=5, pady=5)

#第三页模型表现
#   ----------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

        if (train_time_id!=None ) or (score_time_id!=None) :
            left_frame_3=Frame(tab3,width=self.screenwidth*0.32, height=self.screenheight*0.27)
            self.left_frame_3_width=self.screenwidth*0.32
            def calculateginitime(gini_score):
                ste_da=pd.DataFrame()
                if train_time_id!= None:
                    for period in list(predict_train_data[train_time_id].unique()):
                        fpr, tpr, threshold = roc_curve(predict_train_data[predict_train_data[train_time_id]==period][train_target]
                                                        ,predict_train_data[predict_train_data[train_time_id]==period][gini_score])
                        roc_auc = auc(fpr, tpr)
                        num=predict_train_data[predict_train_data[train_time_id]==period][train_time_id].count()
                        bad_rate=predict_train_data[predict_train_data[train_time_id]==period][train_target].mean()
                        dd={'timeid':period,'auc':roc_auc,'count':num,'bad_rate':bad_rate}
                        ste_da=ste_da.append(pd.DataFrame(dd,index=[1]))
                        ste_da['auc']=round(ste_da['auc'],3)
                        ste_da['bad_rate']=round(ste_da['bad_rate'],3)

                if (self.score_target!=None) and (self.score_time_id!=None):
                    temp_o=pd.DataFrame()
                    for period in list(predict_score_data[score_time_id].unique()):
                        fpr, tpr, threshold = roc_curve(predict_score_data[predict_score_data[score_time_id]==period][score_target]
                                                        ,predict_score_data[predict_score_data[score_time_id]==period][gini_score])  ###计算真正率和假正率
                        roc_auc = auc(fpr, tpr)
                        num=predict_score_data[predict_score_data[score_time_id]==period][score_target].count()
                        bad_rate=predict_score_data[predict_score_data[score_time_id]==period][score_target].mean()
                        dd={'timeid':period,'auc':roc_auc,'count':num,'bad_rate':bad_rate}
                        temp_o=temp_o.append(pd.DataFrame(dd,index=[1]))
                        temp_o['auc']=round(temp_o['auc'],3)
                        temp_o['bad_rate']=round(temp_o['bad_rate'],3)
                    ste_da=pd.merge(ste_da,temp_o,how='outer',on=['timeid'],suffixes=('','_score'))

                ste_da = ste_da.sort_values(by='timeid', ascending=False)
                return ste_da
            model_ginitime=calculateginitime('SCORECARD_LR_p_1')


    #-------------------------------------------------------------------------------------------
            steab_data=LabelFrame(left_frame_3,text='模型表现稳定性')
            col=[]
            if train_time_id != None:
                col=['timeid','auc','count','bad_rate']
            if self.score_target!=None and self.score_time_id!=None :
                col=col+['auc_score','count_score','bad_rate_score']
            columns = col
            tree = ttk.Treeview(steab_data, show = "headings",
                                columns = columns, selectmode = tk.BROWSE)
            tree.column('timeid', anchor = "center",minwidth=0,width=80, stretch=NO)
            if train_time_id != None:
                tree.column( 'auc', anchor = "center",minwidth=0,width=60, stretch=NO)
                tree.column('count', anchor = "center",minwidth=0,width=60, stretch=NO)
                tree.column('bad_rate', anchor = "center",minwidth=0,width=60, stretch=NO)

            if self.score_target!=None and self.score_time_id!=None :
                tree.column( 'auc_score', anchor = "center",minwidth=0,width=60, stretch=NO)
                tree.column('count_score', anchor = "center",minwidth=0,width=60, stretch=NO)
                tree.column('bad_rate_score', anchor = "center",minwidth=0,width=60, stretch=NO)

            tree.heading('timeid', text = "时间ID")
            if train_time_id != None:
                tree.heading( 'auc', text = "Trian AUC")
                tree.heading('count', text = "Trian样本数")
                tree.heading('bad_rate', text = "Trian坏账率")
            if self.score_target!=None and self.score_time_id!=None :
                tree.heading('auc_score', text = "Score AUC")
                tree.heading('count_score', text = "Score样本数")
                tree.heading('bad_rate_score', text = "Score坏账率")
            def inter(x):
                re=(x['timeid'],)
                if train_time_id != None:
                    re=re+(x['auc'],x['count'],x['bad_rate'],)
                if self.score_target!=None and self.score_time_id!=None :
                    re=re+(x['auc_score'],x['count_score'],x['bad_rate_score'],)
                return re
            model_ginitime.apply(lambda x: tree.insert('',0, values = inter(x)),axis=1)
            tree.pack(fill='both',expand=YES)
            steab_data.pack(side=TOP, anchor='nw',fill='y',expand=NO,  padx=5, pady=5)
    #-------------------------------------------------------------------------------------------
            steab_fig=LabelFrame(left_frame_3,text='模型表现稳定性')
            try:
                plt.close()
            except:
                pass
            gg_time = plt.figure(figsize=(10,3),dpi=int((self.left_frame_3_width/10)))
            ax1 = gg_time.add_subplot(1, 1, 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=1, hspace=None)
            model_ginitime = model_ginitime.sort_values(by='timeid', ascending=True)
            x = model_ginitime['timeid']

            bar_width = 0.1
            if train_time_id != None:
                y1 = model_ginitime['bad_rate']
                plt.bar(x=range(len(x)), height=y1, label='Train',  width=bar_width)
            if self.score_target!=None and self.score_time_id!=None :
                y3 = model_ginitime['bad_rate_score']
                plt.bar(x=np.arange(len(x)) +bar_width, height=y3, label='Score',
                        width=bar_width)
            plt.legend()
            # 为两条坐标轴设置名称
            plt.xlabel(u"时间",fontsize=12)
            plt.ylabel(u"坏账率",fontsize=12)

            ax2 = ax1.twinx()
            if train_time_id != None:
                auc_t = model_ginitime['auc']
                ax2.plot(np.arange(len(x)), auc_t)
            if self.score_target!=None and self.score_time_id!=None :
                auc2 =  model_ginitime['auc_score']
                ax2.plot(np.arange(len(x)) + bar_width, auc2)

            ax2.set_ylabel('AUC',fontsize=12)
            plt.xticks(np.arange(len(x)) + 0.15, x, fontsize=12, rotation=45)
            plt.tight_layout()
            canvasg=FigureCanvasTkAgg(gg_time,steab_fig)
            canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)
            canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=0)
            steab_fig.pack(side=TOP, anchor='nw',fill=tk.BOTH,expand=0,  padx=5, pady=5)
            steab_dis = LabelFrame(left_frame_3, text='分布稳定性')
            gg_dis=self.distimeall(df=self.predict_train_data, df_s=predict_score_data, score='SCORECARD_LR_p_1', train_time_id=self.train_time_id, score_time_id=score_time_id)
            canvasg=FigureCanvasTkAgg(gg_dis,steab_dis)
            canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)
            canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=0)
            steab_dis.pack(side=TOP, anchor='nw',fill=tk.BOTH,expand=0,  padx=5, pady=5)
            left_frame_3.pack(side=LEFT, anchor='nw',fill=tk.BOTH,expand=0,  padx=5, pady=5)

    #第三页右边----------------------------------------------------------------------------------------------------

            right_frame_b3=Frame(tab3,width=self.screenwidth*0.5, height=self.screenheight*0.27)
            self.right_frame_b3_width=self.screenwidth*0.4

            def plot_variable_sta(event):
                try:
                    self.s_frame.pack_forget()
                    self.s_frame = ScrollableFrame(right_frame_b3)
                except:
                    self.s_frame = ScrollableFrame(right_frame_b3)
                var_in=self.comboxlist_variable_use.get()
                if var_in !='const':
                    if list(scorecarddf[scorecarddf['variable_name']==var_in]['var_type'])[0]=='add':
                        if model_var_type=='WOE':
                            model_ginitime = calculateginitime(var_in)
                        else:
                            model_ginitime=calculateginitime('f_group_%s'%var_in)
                    else:
                        model_ginitime=calculateginitime('woe_%s'%var_in)
                    var_in_frame=LabelFrame(self.s_frame.scrollable_frame,text=var_in)
                    col=['timeid']
                    if train_time_id != None:
                        col=col+['auc','count','bad_rate']
                    if self.score_target!=None and self.score_time_id!=None :
                        col=col+['auc_score','count_score','bad_rate_score']
                    columns = col
                    tree = ttk.Treeview(var_in_frame, show = "headings",
                                        columns = columns, selectmode = tk.BROWSE)
                    tree.column('timeid', anchor = "center",minwidth=0,width=80, stretch=NO)
                    if train_time_id != None:
                        tree.column( 'auc', anchor = "center",minwidth=0,width=60, stretch=NO)
                        tree.column('count', anchor = "center",minwidth=0,width=60, stretch=NO)
                        tree.column('bad_rate', anchor = "center",minwidth=0,width=60, stretch=NO)
                    if self.score_target!=None and self.score_time_id!=None :
                        tree.column( 'auc_score', anchor = "center",minwidth=0,width=60, stretch=NO)
                        tree.column('count_score', anchor = "center",minwidth=0,width=60, stretch=NO)
                        tree.column('bad_rate_score', anchor = "center",minwidth=0,width=60, stretch=NO)
                    tree.heading('timeid', text = "时间ID")
                    if train_time_id != None:
                        tree.heading( 'auc', text = "Trian AUC")
                        tree.heading('count', text = "Trian样本数")
                        tree.heading('bad_rate', text = "Trian坏账率")
                    if self.score_target!=None and self.score_time_id!=None :
                        tree.heading('auc_score', text = "Score AUC")
                        tree.heading('count_score', text = "Score样本数")
                        tree.heading('bad_rate_score', text = "Score坏账率")
                    def inter(x):
                        re=(x['timeid'],)
                        if train_time_id != None:
                            re=re+(x['auc'],x['count'],x['bad_rate'],)
                        if self.score_target!=None and self.score_time_id!=None :
                            re=re+(x['auc_score'],x['count_score'],x['bad_rate_score'],)
                        return re
                    model_ginitime.apply(lambda x: tree.insert('',0, values = inter(x)),axis=1)
                    tree.pack(fill='both',expand=YES,  padx=5, pady=5)
    #报告----------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------
                    if list(scorecarddf[scorecarddf['variable_name']==var_in]['var_type'])[0]=='add':
                        report=scorecarddf[scorecarddf['variable_name']==var_in][ ['f_group','label', 'f_Bad_rate', 'f_N_obs']]
                        report['iv']=None
                        report['miss_rate']=0
                    else:
                        report=self.f_group_report[self.f_group_report['variable_name']==var_in]
                    columns=[ 'f_group','label','iv', 'f_Bad_rate', 'f_N_obs','miss_rate']
                    tree_rep = ttk.Treeview(var_in_frame, show = "headings",
                                        columns = columns, selectmode = tk.BROWSE)
                    tree_rep.column( 'f_group', anchor = "center",minwidth=0,width=60, stretch=NO)
                    tree_rep.column('label', anchor = "center",minwidth=0,width=280, stretch=NO)
                    tree_rep.column('iv', anchor = "center",minwidth=0,width=60, stretch=NO)
                    tree_rep.column('f_Bad_rate', anchor = "center",minwidth=0,width=60, stretch=NO)
                    tree_rep.column('f_N_obs', anchor = "center",minwidth=0,width=60, stretch=NO)
                    tree_rep.column('miss_rate', anchor = "center",minwidth=0,width=60, stretch=NO)
                    tree_rep.heading( 'f_group', text = "组数")
                    tree_rep.heading('label', text = "标识")
                    tree_rep.heading('iv', text = "信息熵")
                    tree_rep.heading('f_Bad_rate', text = "组坏账率")
                    tree_rep.heading('f_N_obs', text = "组坏账数")
                    tree_rep.heading('miss_rate', text = "缺失率")
                    report.apply(lambda x: tree_rep.insert('',0, values =(x['f_group'],x['label'],x['iv'], x['f_Bad_rate'], x['f_N_obs'],x['miss_rate'])),axis=1)
                    tree_rep.pack(fill='both',expand=YES,  padx=5, pady=5)

            #woe------group--------
                    ccc=self.groupplot_datainit(modify_var=var_in)
                    self.ccc=ccc
                    re_fig=self.plot_group(bad_pct=ccc[0], total_pct=ccc[1], final=ccc[2])
                    canvasg=FigureCanvasTkAgg(re_fig,var_in_frame)
                    canvasg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    canvasg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    var_in_frame.pack(side=TOP, anchor='nw',fill=tk.BOTH,expand=YES,  padx=5, pady=5)
                    self.s_frame.pack(side=BOTTOM, anchor='nw',fill=tk.BOTH,expand=YES)
            var_list=list(set(scorecarddf['variable_name']))
            try:
                var_list.remove('const')
            except:
                pass
            toplbel=LabelFrame(right_frame_b3)
            self.comboxlist_variable_use = ttk.Combobox(toplbel, width=45)
            self.comboxlist_variable_use["value"] = var_list
    #         self.comboxlist_variable_use.current(0)
            self.comboxlist_variable_use.bind("<<ComboboxSelected>>",plot_variable_sta )
            self.comboxlist_variable_use.pack(side=RIGHT, fill='x', anchor='ne', padx=3, pady=3)
            labelc = ttk.Label(toplbel, text='变量选择：')
            labelc.pack(side=RIGHT, fill='x', anchor='ne', padx=3, pady=3)
            toplbel.pack(side=TOP, fill='x', anchor='ne', padx=3, pady=3)
            right_frame_b3.pack(side=RIGHT, anchor='nw',fill=tk.BOTH,expand=YES,  padx=5, pady=5)
        self.master.config(menu = menubar)

    def PlotKS(self,preds_t, labels_t, n, asc,
                   preds_s, labels_s,flag_score_data, flag_score_target,):
        # preds is score: asc=1
        # preds is prob: asc=0
        def calculate_data(preds, labels, n, asc):
            gg=plt.figure()
            pred = preds  # 预测值
            bad = labels  # 取1为bad, 0为good
            ksds = pd.DataFrame({'bad': bad, 'pred': pred})
            ksds['good'] = 1 - ksds.bad

            if asc == 1:
                ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
            elif asc == 0:
                ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
            ksds1.index = range(len(ksds1.pred))
            ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
            ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

            if asc == 1:
                ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
            elif asc == 0:
                ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
            ksds2.index = range(len(ksds2.pred))
            ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
            ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

            # ksds1 ksds2 -> average
            ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
            ksds['cumsum_good2'] = ksds2['cumsum_good2']
            ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
            ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
            ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

            # ks
            ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
            ksds['tile0'] = range(1, len(ksds.ks) + 1)
            ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

            qe = list(np.arange(0, 1, 1.0 / n))
            qe.append(1)
            qe = qe[1:]

            ks_index = pd.Series(ksds.index)
            ks_index = ks_index.quantile(q=qe)
            ks_index = np.ceil(ks_index).astype(int)
            ks_index = list(ks_index)

            ksds = ksds.loc[ks_index]
            ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
            ksds0 = np.array([[0, 0, 0, 0]])
            ksds = np.concatenate([ksds0, ksds], axis=0)
            ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

            ks_value = ksds.ks.max()
            ks_pop = ksds.tile[ksds.ks.idxmax()]
            return ksds,ks_pop,ks_value

        ksds_t,ks_pop_t,ks_value_t=calculate_data(preds_t, labels_t, n, asc)

        if flag_score_data==True and flag_score_target==True:
            ksds_s,ks_pop_s,ks_value_s=calculate_data(preds_s, labels_s, n, asc)
        try:
            plt.close()
        except:
            pass
        gg=plt.figure()
        # chart
        plt.plot(ksds_t.tile, ksds_t.cumsum_good, label='训练累计好占比',
                 color='blue', linestyle='-', linewidth=2)

        plt.plot(ksds_t.tile, ksds_t.cumsum_bad, label='训练累计坏占比',
                 color='red', linestyle='-', linewidth=2)

        plt.plot(ksds_t.tile, ksds_t.ks, label='训练ks曲线',
                 color='green', linestyle='-', linewidth=2)

        if flag_score_data==True and flag_score_target==True:
            plt.plot(ksds_s.tile, ksds_s.cumsum_good, label='Score累计好占比',
              linestyle='--', linewidth=2)

            plt.plot(ksds_s.tile, ksds_s.cumsum_bad, label='Score累计坏占比',
                      linestyle='--', linewidth=2)

            plt.plot(ksds_s.tile, ksds_s.ks, label='Score ks曲线',
                      linestyle='--', linewidth=2)
        plt.legend()
        plt.axvline(ks_pop_t, color='gray', linestyle='--')
        tiltle='训练KS=%s ' % np.round(ks_value_t, 4) +'at Pop=%s' % np.round(ks_pop_t, 4)
        if flag_score_data==True and flag_score_target==True:
            tiltle=tiltle+'\nScoreKS=%s ' % np.round(ks_value_s, 4) +'at Pop=%s' % np.round(ks_pop_s, 4)
        plt.title(tiltle, fontsize=12)
        plt.tight_layout()
        return gg
    def AUC(self,preds_t, labels_t, preds_s,labels_s,flag_score_data,flag_score_target):
        try:
            plt.close()
        except:
            pass
        fpr_t, tpr_t, threshold_t = roc_curve(preds_t,labels_t)  ###计算真正率和假正率
        roc_auc_t = auc(fpr_t, tpr_t)  ###计算auc的值

        if flag_score_data==True and flag_score_target==True:
            fpr_s, tpr_s, threshold_s = roc_curve(preds_s,labels_s)  ###计算真正率和假正率
            roc_auc_s = auc(fpr_s, tpr_s)  ###计算auc的值

        gg=plt.figure()
        lw = 2
        plt.plot(fpr_t, tpr_t, color='darkorange',lw=lw , label='训练ROC curve (area = %0.3f)' % roc_auc_t)  ###假正率为横坐标，真正率为纵坐标做曲线

        if flag_score_data==True and flag_score_target==True:
            plt.plot(fpr_s, tpr_s, lw=lw ,linestyle='--', label='Score ROC curve (area = %0.3f)' % roc_auc_s)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('累计坏占比',fontsize=12)
        plt.ylabel('累计好占比',fontsize=12)
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.tight_layout()
        return gg
    def plotdis(self,df,target,score):
        try:
            plt.close()
        except:
            pass
        gg=plt.figure()
        n=plt.hist(df[df[target]==1][score], bins=100, density=1, alpha=0.5,
                  facecolor='blue',
                                     edgecolor='black',
                  histtype='bar',label='逾期客户'
                )
        m=plt.hist(df[df[target]==0][score], bins=100, density=1, alpha=0.5,
                  facecolor='green',
                                     edgecolor='black',
                  histtype='bar',label='正常客户'
                )
        m=plt.hist(df[score], bins=100, density=1, alpha=0.5,
                  facecolor='yellow',edgecolor='black',histtype='bar',label='所有客户')
        plt.legend()
        plt.title('分数分布区间')
        plt.tight_layout()
        return gg
    def cali(self,df,df_s,score ,target,flag_score_data,flag_score_target,score_target):
        try:
            plt.close()
        except:
            pass
        sp=self.num_finebin_group(df, score, 100,[])
        temp=self.binning(sp, df, score, 's_max', 's_min', 's_group', [])
        qq=temp.groupby([ 's_max', 's_min']).agg({score:'mean',target:'mean','s_group':'count'}).reset_index().rename({'s_group':'count'},axis=1)
        # QQ图和趋势线
        gg=plt.figure()
        z1 = np.polyfit(qq[score], qq[target], 1) # 用4次多项式拟合
        p1 = np.poly1d(z1)
        r=pow(qq[[score,target]].corr().iloc[1,0],2)
        note='训练集 y= %sx+%s'%(round(p1[1],3),round(p1[0],4)) +'\n R^2 = %s' %round(r,3)
         # 在屏幕上打印拟合多项式
        yvals=p1(qq[score]) # 也可以使用yvals=np.polyval(z1,x)
        
        if flag_score_data==True and flag_score_target==True:
            
            temp_s=self.binning(sp, df_s, score, 's_max', 's_min', 's_group', [])
            qq_s=temp_s.groupby([ 's_max', 's_min']).agg({score:'mean',score_target:'mean','s_group':'count'}).reset_index().rename({'s_group':'count'},axis=1)
            # QQ图和趋势线

            z1_s = np.polyfit(qq_s[score], qq_s[score_target], 1) # 用4次多项式拟合
            p1_s = np.poly1d(z1_s)
            r_s=pow(qq_s[[score,score_target]].corr().iloc[1,0],2)
            note_s='\nScore集y= %sx+%s'%(round(p1_s[1],3),round(p1_s[0],4)) +'\n R^2 = %s' %round(r,3)
             # 在屏幕上打印拟合多项式
            note=note+note_s
            yvals_s=p1(qq_s[score]) # 也可以使用yvals=np.polyval(z1,x)
            qq_s=qq_s.rename({score_target:'%s_s'%score_target},axis=1)
            qq=pd.merge(qq,qq_s,how='left',on=['s_max','s_min'],suffixes=('','_s'))
        plot1=plt.plot(qq[score], qq[target], '*')
        plot2=plt.plot(qq[score], yvals, 'r',label='训练集趋势线')

        if flag_score_data==True and flag_score_target==True:
            plot1=plt.plot(qq[score], qq['%s_s'%score_target], 'x')
            plot2=plt.plot(qq[score], yvals_s, 'b',label='Score趋势线')
        plt.xlabel('分数',fontsize=12)
        plt.ylabel('坏账率',fontsize=12)
        plt.text(0, 0,note , fontsize=12, color = "r", style = "italic", weight = "light",alpha=0.5,)
        plt.legend() # 指定legend的位置,读者可以自己help它的用法
        plt.title('Calibration')
        return qq, gg
    def binning(self,group_data, inputdata, col, inputmax, inputmin, inputgroup, specialcode_list):
        s_group_data = pd.DataFrame()
        group_data = group_data.reset_index()
        if specialcode_list != []:
            inputdata = inputdata[inputdata.isin({col: specialcode_list}) == False]
        inputdata = inputdata.loc[inputdata[col].isnull() == False]
        for lins in range(len(group_data)):
            temp = inputdata.copy()
            temp[inputgroup] = group_data.loc[lins, inputgroup]
            temp[inputmin] = group_data.loc[lins, inputmin]
            temp[inputmax] = group_data.loc[lins, inputmax]
            temp_data = temp[
                ((temp[col] <= temp[inputmax]) & (temp[col] > temp[inputmin]))
            ]
            s_group_data = pd.concat([s_group_data, temp_data])
        del temp
        return s_group_data
    def num_finebin_group(self,inputdata, col, s_bin_num, specialcode_list):
        if specialcode_list != []:
            inputdata = inputdata[inputdata.isin({col: specialcode_list}) == False]
        sort_df = inputdata[col][inputdata[col].isnull() == False]
        if len(sort_df.unique()) == 1:
            old_list = set([float('-inf'), float('inf')])
        elif len(sort_df.unique()) <= s_bin_num:
            old_list = set(list(sort_df.unique()))
            old_list.remove(max(old_list))
            old_list.remove(min(old_list))
            old_list.add(float('-inf'))
            old_list.add(float('inf'))
        else:
            old_list = set([float('-inf'), float('inf')])
            num = sort_df.size
            sort_df = sort_df.sort_values(ascending=True).reset_index().drop(columns='index')
            for i in range(1, s_bin_num):
                loca = int((i / s_bin_num) * num)
                value = sort_df.iloc[loca].values[0]
                old_list.add(value)
        new_list = list(old_list)
        new_list.sort()
        new_s_group = []
        for i in range(len(new_list) - 1):
            temp = {'s_group': i, 's_min': new_list[i], 's_max': new_list[i + 1]}
            new_s_group.append(temp)
        s_group_map = pd.DataFrame(new_s_group)
        return s_group_map
    def distimeall(self,df,df_s,score,train_time_id,score_time_id):
        try:
            plt.close()
        except:
            pass
        sp=self.num_finebin_group(df, score, 10,[])
        final_df=pd.DataFrame(columns=['time_id','s_max','s_min','s_group'])
        if train_time_id != None:
            temp=self.binning(sp, df, score, 's_max', 's_min', 's_group', [])
            qq=temp.groupby([train_time_id, 's_max', 's_min', 's_group']).agg({score:'count'}).reset_index().rename({train_time_id:'time_id',score:'count'},axis=1)
            qq_t = temp.groupby(train_time_id).agg({'s_group': 'count'}).reset_index().rename({train_time_id: 'time_id', 's_group': 'count_total'}, axis=1)
            qq=pd.merge(qq,qq_t,how='outer',on='time_id')
            qq['pct']=qq['count']/qq['count_total']
            final_df=pd.merge(final_df,qq,how='outer',on=['time_id','s_max','s_min', 's_group'])

        if score_time_id!=None:
            temp=self.binning(sp, df_s, score, 's_max', 's_min', 's_group', [])
            qq=temp.groupby([score_time_id, 's_max', 's_min', 's_group']).agg({score:'count'}).reset_index().rename({score_time_id:'time_id',score:'count'},axis=1)
            qq_s = temp.groupby(score_time_id).agg({'s_group': 'count'}).reset_index().rename({score_time_id: 'time_id', 's_group': 'count_total'}, axis=1)
            qq=pd.merge(qq,qq_s,how='outer',on='time_id')
            qq['pct_s']=qq['count']/qq['count_total']
            final_df=pd.merge(final_df,qq,how='outer',on=['time_id','s_max','s_min', 's_group'])
        final_df=final_df.fillna(0)
        #补上所有time_id 和 group 维度
        temp=final_df[['s_max', 's_min', 's_group']].drop_duplicates()
        plot_df=pd.DataFrame()
        for Time_id in set(final_df['time_id']):
            temp['time_id']=Time_id
            plot_df=plot_df.append(temp)
        plot_df=pd.merge(plot_df,final_df,how='left',on=['time_id','s_max','s_min', 's_group'])
        plot_df = plot_df.fillna(0)
        plot_df=plot_df.sort_values(by=['time_id','s_group'],ascending=[True,True])

        gg = plt.figure()
        ind = np.arange(len(set(plot_df['time_id'])))
        color_list=['gray','brown','orange','olive','green','cyan','blue','purple','pink','red','black']
        if train_time_id != None:
            BT = [0 for _ in range(len(set(plot_df['time_id'])))]
            c=0
            label_list = []
            pt = []

            for group in set(plot_df['s_group']):
                sort_df=plot_df[plot_df['s_group'] == group].sort_values(by='time_id',ascending=True)
                g=list(sort_df['pct'])
                x = list(sort_df['time_id'])
                s_min=round(list(plot_df[plot_df['s_group']==group]['s_min'])[0],2)
                s_max=round(list(plot_df[plot_df['s_group']==group]['s_max'])[0],2)
                p = plt.bar(ind, g, width=0.145, bottom=BT, color=color_list[c])
                label_list.append('%s-%s'%(s_min,s_max))
                pt.append(p)
                c=c+1
                d=[]
                for i in range(0, len(g)):
                    sum = BT[i] + g[i]
                    d.append(sum)
                BT=d
            plt.text(ind[0], 1.11, str('Train'), ha='center', rotation=90, fontsize=6)
        if 'pct_s' in plot_df.columns:
            BT = [0 for _ in range(len(set(plot_df['time_id'])))]
            c = 0
            label_list = []
            pt = []
            for group in set(plot_df['s_group']):
                sort_df = plot_df[plot_df['s_group'] == group].sort_values(by='time_id', ascending=True)
                g = list(sort_df['pct_s'])
                x = list(sort_df['time_id'])
                s_min=round(list(plot_df[plot_df['s_group']==group]['s_min'])[0],2)
                s_max=round(list(plot_df[plot_df['s_group']==group]['s_max'])[0],2)
                p=plt.bar(ind+(train_time_id != None)*0.15, g, width=0.145, bottom=BT,  color=color_list[c])
                label_list.append('%s-%s' % (s_min, s_max))
                pt.append(p)
                c = c + 1
                d=[]
                for i in range(0, len(g)):
                    sum = BT[i] + g[i]
                    d.append(sum)
                BT=d
            plt.text(ind[0]+(train_time_id != None)*0.155, 1.11, str('Score'), ha='center',rotation=90,fontsize=8)
        plt.ylabel('样本占比',fontsize=12)
        plt.title('时间分布图',fontsize=12)
        plt.xticks(ind, x,rotation=30)

        plt.legend([x[0] for x in pt], label_list,bbox_to_anchor=(0, -0.1), loc="upper left", prop={'size': 9},
                   borderaxespad=0, ncol=5)
        plt.tight_layout()
        return gg
    def disallfb(self,df,df_s,score_target_flag,train_target,score_target):
        try:
            plt.close()
        except:
            pass
        full=pd.DataFrame(columns=['plot_calibraition_flag'])
        full['plot_calibraition_flag']=('0.05','0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45','0.50','0.55','0.60','0.65','0.70','0.75','0.80', '0.85','0.90','0.95','1.00')

        def cal(x):
            for i in range(1,21):
                if x <= i*(1/20):
                    return '%4.2f' %(i*(1/20))
                    break
        df['plot_calibraition_flag'] = df['SCORECARD_LR_p_1'].apply(lambda x: cal(x)) 
        pre_plot=df.groupby('plot_calibraition_flag')[train_target].agg({'sum','count'}).reset_index()
        pre_plot['bad_pct']=pre_plot['sum']/pre_plot['count'].sum()
        pre_plot['total_pct']=pre_plot['count']/pre_plot['count'].sum()
        pre_plot['good_pct']=pre_plot['total_pct']-pre_plot['bad_pct']
        pre_plot=pd.merge(pre_plot,full,how='outer',on='plot_calibraition_flag').fillna(0)



        if score_target_flag==True:
            df_s['plot_calibraition_flag'] = df_s['SCORECARD_LR_p_1'].apply(lambda x: cal(x))
            pre_plot_s=df_s.groupby('plot_calibraition_flag')[score_target].agg({'sum','count'}).reset_index()
            pre_plot_s['bad_pct']=pre_plot_s['sum']/pre_plot_s['count'].sum()
            pre_plot_s['total_pct']=pre_plot_s['count']/pre_plot_s['count'].sum()
            pre_plot_s['good_pct']=pre_plot_s['total_pct']-pre_plot_s['bad_pct']
        else:
            df_s['plot_calibraition_flag'] = df_s['SCORECARD_LR_p_1'].apply(lambda x: cal(x))
            pre_plot_s=df_s.groupby('plot_calibraition_flag')['plot_calibraition_flag'].agg({'count'}).reset_index()
            pre_plot_s['total_pct']=pre_plot_s['count']/pre_plot_s['count'].sum()
        pre_plot_s=pd.merge(pre_plot_s,full,how='outer',on='plot_calibraition_flag').fillna(0)



        gg=plt.figure()
        N = len(pre_plot)
        GT = pre_plot['good_pct']
        BT = pre_plot['bad_pct']

        if score_target_flag==True:
            GS = pre_plot_s['good_pct']
            BS = pre_plot_s['bad_pct']
        else:
            Score=pre_plot_s['total_pct']
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, GT, width=0.15,label='训练好客户' )#, yerr=menStd)
        p2 = plt.bar(ind, BT, width=0.15, bottom=GT,label='训练坏客户')#, yerr=womenStd)

        if score_target_flag==True:
            p5 = plt.bar(ind+0.15, GS, width=0.15, label='Score好客户')#, yerr=menStd)
            p6 = plt.bar(ind+0.15, BS, width=0.15, bottom=GS,label='Score坏客户')#, yerr=womenStd)
        else:
            p7 = plt.bar(ind+0.15, Score, width=0.15,label='Score客户')

        plt.ylabel('样本占比',fontsize=12)
        plt.title('分数区间分布',fontsize=12)
        plt.xticks(ind, pre_plot['plot_calibraition_flag'])

        plt.legend()
        plt.tight_layout()
        return gg
    def groupplot_datainit(self,modify_var):
        grouped_traindata = self.predict_train_data

        grouped_scoredata = self.predict_score_data

        total_pct = pd.DataFrame(
            grouped_traindata.groupby(['f_group_%s' % modify_var])['f_group_%s' % modify_var].count()).rename(
            {'f_group_%s' % modify_var: 'num'}, axis=1).reset_index()
        total_pct['Group'] = total_pct['f_group_%s' % modify_var]
        total_pct['pct_train'] = total_pct['num'] / total_pct['num'].sum()
        total_pct = total_pct.drop(columns=['f_group_%s' % modify_var])

        bad_pct = grouped_traindata.groupby(['f_group_%s' % modify_var]).agg(
            {self.train_target: ['sum', 'count']}).reset_index().rename({'f_group_%s' % modify_var: 'Group'}, axis=1)
        bad_pct['badrate_train'] = bad_pct[self.train_target]['sum'] / bad_pct[self.train_target]['count']
        bad_pct = bad_pct[['Group', 'badrate_train']]

        if  self.score_target!=None and self.score_time_id!=None :

            score_pct = pd.DataFrame(grouped_scoredata.groupby(['f_group_%s' % modify_var])[ 'f_group_%s' % modify_var].count()).rename(
                {'f_group_%s' % modify_var: 'num'}, axis=1).reset_index()
            score_pct['Group'] = score_pct['f_group_%s' % modify_var]
            score_pct['pct_score'] = score_pct['num'] / score_pct['num'].sum()
            score_pct = score_pct.drop(columns=['f_group_%s' % modify_var])
            total_pct = pd.merge(total_pct, score_pct, how='outer', on='Group')
            if self.score_target != None:
                scorebad_pct = grouped_scoredata.groupby(['f_group_%s' % modify_var]).agg(
                    {self.score_target: ['sum', 'count']}).reset_index().rename({'f_group_%s' % modify_var: 'Group'}, axis=1)
                scorebad_pct['badrate_score'] = scorebad_pct[self.score_target]['sum'] / scorebad_pct[self.score_target]['count']
                scorebad_pct = scorebad_pct[['Group', 'badrate_score']]
                bad_pct = pd.merge(bad_pct, scorebad_pct, how='outer', on='Group')

#时间稳定性
        base_list = []
        for gr in grouped_traindata['f_group_%s' % modify_var].unique():
            for ti in grouped_traindata[self.train_time_id].unique():
                base_list.append({'Group': gr, 'timeid': ti})
        base_df = pd.DataFrame(base_list)
        valid_pct = pd.DataFrame(
            grouped_traindata.groupby(['f_group_%s' % modify_var, self.train_time_id])[self.train_target].agg(
                {'count', 'sum'}).reset_index())
        total = pd.DataFrame(
            grouped_traindata.groupby([self.train_time_id])[self.train_target].agg({'count'}).reset_index().rename(
                {'count': 'total_num'}, axis=1))
        final = pd.merge(total, valid_pct, how='outer', on=self.train_time_id)
        final['popu_pct'] = final['count'] / final['total_num']
        final=final.rename({self.train_time_id:'timeid'},axis=1)
        final['bad_rate'] = final['sum'] / final['count']
        final['Group'] = final['f_group_%s' % modify_var]
        final = pd.merge(base_df, final, how='outer', on=['Group', 'timeid'])


        if self.score_time_id != None:
            base_list=[]
            for gr in grouped_scoredata['f_group_%s' % modify_var].unique():
                for ti in grouped_scoredata[self.score_time_id].unique():
                    base_list.append({'Group':gr, 'timeid':ti})
            base_df=pd.DataFrame(base_list)
            if self.score_target != None:
                valid_pct = pd.DataFrame(
                    grouped_scoredata.groupby(['f_group_%s' % modify_var, self.score_time_id])[
                        self.score_target].agg({'count', 'sum'}).reset_index())
                total = pd.DataFrame(grouped_scoredata.groupby([self.score_time_id])[self.score_target].agg(
                    {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final_r = pd.merge(total, valid_pct, how='outer', on=self.score_time_id)
                final_r['popu_pct'] = final_r['count'] / final_r['total_num']
                final_r['bad_rate'] = final_r['sum'] / final_r['count']
                final_r=final_r.rename({self.score_time_id:'timeid'},axis=1)
                final_r['Group'] = final_r['f_group_%s' % modify_var]
                final=pd.merge(final,final_r,how='outer',on=['Group','timeid'],suffixes=('','_score'))
                final = pd.merge(base_df, final, how='outer', on=['Group', 'timeid'])
            else:
                valid_pct = pd.DataFrame(
                    grouped_scoredata.groupby(['f_group_%s' % modify_var, self.score_time_id])[
                        modify_var].agg({'count'}).reset_index())
                total = pd.DataFrame(
                    grouped_scoredata.groupby([self.score_time_id])['f_group_%s' % modify_var].agg(
                        {'count'}).reset_index().rename({'count': 'total_num'}, axis=1))
                final_r = pd.merge(total, valid_pct, how='outer', on=self.score_time_id)
                final_r['popu_pct'] = final_r['count'] / final_r['total_num']
                final_r=final_r.rename({self.score_time_id:'timeid'},axis=1)
                final_r['Group'] = final_r['f_group_%s' % modify_var]
                final=pd.merge(final,final_r,how='outer',on=['Group','timeid'],suffixes=('','_score'))
                final = pd.merge(base_df, final, how='outer', on=['Group', 'timeid'])
        return bad_pct, total_pct, final
    def plot_group(self, bad_pct, total_pct, final):
        try:
            plt.close()
        except:
            pass
        total_pic=1
        final=final.sort_values(by='timeid',ascending=True)
        if  'popu_pct'  in list(final.columns) :
            total_pic=total_pic+1
        if 'popu_pct_score'  in list(final.columns) :
            total_pic=total_pic+1
        fig = plt.figure(figsize=(10,6.3*total_pic),dpi=int((self.right_frame_b3_width/10)))
        
#      
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.3, hspace=0.3)
        num_pic=1

    #-----------------------------------------------------
        # 组分布
        ax1 = fig.add_subplot(total_pic, 1, num_pic)
        x = total_pct['Group']
        y1 = total_pct['pct_train']
        bar_width = 0.2
        plt.bar(x=range(len(x)), height=y1, label='Train', color='steelblue',  width=bar_width)

        if 'pct_score' in list(total_pct.columns):
            y3 = total_pct['pct_score']
            plt.bar(x=np.arange(len(x)) + 2 * bar_width, height=y3, label='Score', color='powderblue',
                    width=bar_width)

        plt.legend(loc='upper center',ncol=5)


        plt.xlabel(u"组",fontsize=12)
        plt.ylabel(u"样本占比",fontsize=12)
        plt.title('分组分布',fontsize=12)
        badrate1 = bad_pct['badrate_train']
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(x)), badrate1, 'steelblue')

        if ('badrate_score', '') in list(bad_pct.columns):
            badrate3 = bad_pct['badrate_score']
            ax2.plot(np.arange(len(x)) + bar_width * 2, badrate3, 'powderblue')
        ax2.set_ylabel('坏账率',fontsize=12)
        plt.xticks(np.arange(len(x)) + 0.15, x, fontsize=12, rotation=20)
        num_pic=num_pic+1
    #-----------------------------------------------------
        # Train

        if 'popu_pct'  in list(final.columns) :
            ax3 = fig.add_subplot(total_pic, 1, num_pic)
            bar_width = 0.1
            i = 0
            group_list = list(final['Group'].unique())
            group_list.sort(reverse=False)
            for group in group_list:
                tt = final[final['Group'] == group]
                y2 = tt['popu_pct']
                x = tt['timeid']
                plt.bar(x=np.arange(len(x)) + 0.08 * i, height=y2, label="组_%s" % group,
                        width=bar_width)
                i = i + 1
            plt.legend(loc='upper center',ncol=5)

            plt.xlabel("Time",fontsize=12)
            plt.ylabel("样本占比",fontsize=12)
            plt.title('Train',fontsize=12)
            plt.xticks(np.arange(len(x)) + 0.15, x, fontsize=12, rotation=45)
            if 'bad_rate' in list(tt.columns):
                ax4 = ax3.twinx()  # this is the important function
                i = 0
                group_list = list(final['Group'].unique())
                group_list.sort(reverse=False)
                for group in group_list:
                    tt = final[final['Group'] == group]
                    y = tt['bad_rate']
                    x = tt['timeid']
                    ax4.plot(np.arange(len(x)) + 0.08 * i, y)
                    i = i + 1
                ax4.set_ylabel('坏账率',fontsize=12)
            num_pic=num_pic+1


    #-----------------------------------------------------
        # reject 
        if 'popu_pct_score'  in list(final.columns) :
            ax6 = fig.add_subplot(total_pic, 1, num_pic)
            bar_width = 0.1
            i = 0
            group_list=list(final['Group'].unique())
            group_list.sort(reverse=False)
            for group in group_list:
                tt = final[final['Group'] == group]
                y2 = tt['popu_pct_score']
                x = tt['timeid']
                plt.bar(x=np.arange(len(x)) + 0.08 * i, height=y2, label="组_%s" % group,  
                        width=bar_width)
                i = i + 1
            plt.legend(loc='upper center',ncol=5)

            plt.xlabel("Time",fontsize=12)
            plt.ylabel("样本占比",fontsize=12)
            plt.title('score',fontsize=12)
            plt.xticks(np.arange(len(x)) + 0.15, x, fontsize=12, rotation=45)
            if 'bad_rate_score' in list(tt.columns):
                ax7 = ax6.twinx()  # this is the important function
                i = 0
                for group in list(final['Group'].unique()):
                    tt = final[final['Group'] == group]
                    y = tt['bad_rate_score']
                    x = tt['timeid']
                    ax7.plot(np.arange(len(x)) + 0.08 * i, y)
                    i = i + 1
                ax7.set_ylabel('坏账率',fontsize=12)
            num_pic=num_pic+1

    #-----------------------------------------------------
        plt.tight_layout()
        return fig
    def output_report(self):
        pass