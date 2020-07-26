import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from joblib import Parallel, delayed

import joblib

from tkinter import *

class lrmodel():
    def __init__(self,):


        pass
    def woe_logistic_regression(self,mianframe, inditor_pct, inditor_sample, var, p_value_entry,p_value_stay, add_inditor,
                                intercept, criterion, df, response, direction, show_step, apply_restrict, n_job=None,
                                flag_IGN=True):
        tip = Toplevel(mianframe)
        self.text = StringVar()
        self.label_list=''
        self.label_list=self.label_list+'start...'
        self.text.set(self.label_list)
        lb = Label(tip, textvariable=self.text)
        lb.pack()
        mianframe.update()
        record_list = []
        modelvar_match_df = pd.DataFrame()
        if n_job == None:
            n_job = joblib.cpu_count() - 1
        flag_next = True
        indicator_df = pd.DataFrame()
        if direction=='NO':
            add_inditor=False
        if flag_IGN == True:
            woe_varlist = ['woe_' + x for x in var]
            modelvar_match_df['ori_var'] = var
            modelvar_match_df['model_var'] = woe_varlist
            modelvar_match_df['var_type'] = 'ori'
        else:
            inditor_pct == False
            woe_varlist = var
        if apply_restrict:
            remove = []
            for p in range(len(woe_varlist) - 1):
                if len(df[woe_varlist[p]].unique()) < 2:
                    remove = remove + [woe_varlist[p]]
            var_clearn_t = set(woe_varlist) - set(remove)
            var_clearn_t = list(var_clearn_t)
            corr_data = df[var_clearn_t].corr()
            for col in corr_data.columns:
                if len(corr_data[corr_data[col] > 0.99]) > 1:
                    corr_data[corr_data.index == col] = 0
                    remove = remove + [col]
            var_clearn = set(woe_varlist) - set(remove)
            var_clearn = list(var_clearn)
            print('those varable will be not involve modeling because the high corr(>0.99) or zero performance', remove)
            record_list.append(
                'those varable will be not involve modeling because the high corr(>0.99) or zero performance')
            record_list.append(remove)
        else:
            var_clearn = woe_varlist
        # 检查可能添加的辅助变量
        if add_inditor == True:
            tlist = []
            for m in range(len(var_clearn)):
                for p in range(m + 1, len(var_clearn)):
                    vara = var[m]
                    varb = var[p]
                    listt = [vara, varb]
                    tlist.append(listt)
            lent = math.ceil(len(tlist) / n_job)

            def func(num):
                summary = pd.DataFrame()
                for l in range((num - 1) * lent, min(len(tlist), num * lent)):
                    pair = tlist[l]
                    vara = pair[0]
                    varb = pair[1]
                    target = response
                    var_a = df.groupby(['f_group_%s' % vara])[target].mean().reset_index().rename(
                        {target: 'badrate_vara'},
                        axis=1)
                    var_b = df.groupby(['f_group_%s' % varb])[target].mean().reset_index().rename(
                        {target: 'badrate_varb'},
                        axis=1)
                    tt = df.groupby(['f_group_%s' % vara, 'f_group_%s' % varb]).agg(
                        {target: ['mean', 'count']}).reset_index()
                    tt.columns = ['f_group_%s' % vara, 'f_group_%s' % varb, 'badrate', 'count']
                    tt = pd.merge(tt, var_a, how='left', on='f_group_%s' % vara)
                    tt = pd.merge(tt, var_b, how='left', on='f_group_%s' % varb)
                    tt['vara'] = vara
                    tt['varb'] = varb
                    tt = tt.rename({'f_group_%s' % vara: 'group_a', 'f_group_%s' % varb: 'group_b'}, axis=1)
                    summary = summary.append(tt)
                return summary
            scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(
                delayed(func)(num) for num in range(1, 1 + n_job))
            score_df = pd.DataFrame()
            for tt in scores_with_candidates:
                sc = pd.DataFrame(tt)
                score_df = score_df.append(sc)
            score_df['model_var'] = score_df.apply(lambda x: 'ind_f_group_%s_%s_f_group_%s_%s'% (x['vara'], int(x['group_a']), x['varb'], int(x['group_b'])),axis=1)
            score_df['var_type'] = 'add'
            modelvar_match_df = modelvar_match_df.append(score_df)
            che = score_df[((score_df['badrate'] < score_df['badrate_vara']) & (score_df['badrate'] < score_df['badrate_varb'])) | ((score_df['badrate'] > score_df['badrate_vara']) & (score_df['badrate'] > score_df['badrate_varb']))]
            che['h'] = che.apply(lambda x: min(abs(x['badrate'] - max(x['badrate_vara'], x['badrate_varb'])),abs(x['badrate'] - min(x['badrate_vara'], x['badrate_varb']))), axis=1)
            che_fin = che[(che['h'] > inditor_pct) & (che['count'] > len(df) * inditor_sample)&(che['count'] > 800)]
            remaining_inditor = che_fin
            indicator_df = che_fin
        else:
            remaining_inditor=pd.DataFrame()
        # 向前回归
        if direction == 'forward':
            remaining_list = var_clearn
            selected = []
            best_score = np.inf
            if show_step:
                print('\nforward_stepwise starting:\n')
                record_list.append('\nforward_stepwise starting:\n')
                self.label_list = self.label_list + '\nforward_stepwise starting:\n'
                self.text.set(self.label_list)
                mianframe.update()
            # 当变量未剔除完，并且当前评分更新时进行循环
            while remaining_list != [] and flag_next == True:
                lent = math.ceil(len(remaining_list) / n_job)
                def func(num):
                    result_list = []
                    for i in range(num * lent, min((num + 1) * lent, len(remaining_list))):
                        candidate = remaining_list[i]
                        if intercept:  # 是否有截距
                            logit_mod = sm.Logit(df[response], sm.add_constant(df[selected + [candidate]]))
                        else:
                            logit_mod = sm.Logit(df[response], df[selected + [candidate]])
                        result = logit_mod.fit(method='lbfgs', maxiter=100)
                        var = candidate
                        pvalue = result.pvalues[candidate]
                        score = eval('result.' + criterion)
                        result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                    return result_list
                scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(num) for num
                                                                                            in range(n_job))

                score_df = pd.DataFrame()
                for tt in scores_with_candidates:
                    sc = pd.DataFrame(tt)
                    score_df = score_df.append(sc)
                # 这几个指标取最小值进行优化
                if criterion == 'llr':
                    score_df['score'] = -score_df['score']
                score_df = score_df[score_df['pvalue'] <= p_value_entry]
                score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行升序排序
                if len(score_df) > 0:
                    current_score = score_df.iloc[0]['score']
                    if current_score < best_score:
                        best_score = current_score
                        selected.append(score_df.iloc[0]['var'])
                        remaining_list.remove(score_df.iloc[0]['var'])
                        if (add_inditor == True) & (remaining_inditor.empty == False):
                            remaining_inditor['add'] = remaining_inditor.apply(
                                lambda x: 'Y' if ('woe_' + x['vara'] in selected) & (
                                        'woe_' + x['varb'] in selected) else 'N', axis=1)
                            add = remaining_inditor[remaining_inditor['add'] == 'Y']
                            remaining_inditor = remaining_inditor[remaining_inditor['add'] == 'N']

                            def add_indictor(vara, varb, groupa, groupb):
                                df['ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb))] = df.apply(lambda x: 1 if (x['f_group_%s' % vara] == groupa) & (x['f_group_%s' % varb] == groupb) else 0, axis=1)
                                remaining_list.append('ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb)))
                            if len(add) > 0:
                                add.apply(lambda x: add_indictor(vara=x['vara'], varb=x['varb'], groupa=x['group_a'],
                                                                 groupb=x['group_b']), axis=1)
                        flag_next = True
                        if show_step:  # 是否显示逐步回归过程
                            if criterion == 'llr':
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            else:
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            if intercept:  # 是否有截距
                                logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))

                            else:
                                logit_mod = sm.Logit(df[response], df[selected])
                            model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
                            record_list.append(model.summary2())
                    else:
                        flag_next = False
                else:
                    flag_next = False

            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))

            else:
                logit_mod = sm.Logit(df[response], df[selected])

            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程

                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
        # 逐步回归
        if direction == 'stepwise':
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            include_var = []
            flag_nonew = 0
            best_score = np.inf
            if show_step:
                print('\nstepwise starting:\n')
                record_list.append('\nstepwise starting:\n')
                self.label_list = self.label_list + '\nstepwise starting:\n'
                self.text.set(self.label_list)
                mianframe.update()
            # 当变量未剔除完，并且当前评分更新时进行循环
            while remaining_list != [] and flag_next == True:

                lent = math.ceil(len(remaining_list) / n_job)

                def func(num):
                    result_list = []
                    for i in range(num * lent, min((num + 1) * lent, len(remaining_list))):
                        candidate = remaining_list[i]
                        if intercept:  # 是否有截距
                            logit_mod = sm.Logit(df[response], sm.add_constant(df[selected + [candidate]]))
                        else:
                            logit_mod = sm.Logit(df[response], df[selected + [candidate]])
                        result = logit_mod.fit(method='lbfgs', maxiter=100)
                        var = candidate
                        pvalue = result.pvalues[candidate]
                        score = eval('result.' + criterion)
                        result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                    return result_list

                scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(num) for num
                                                                                            in range(n_job))
                score_df = pd.DataFrame()
                for tt in scores_with_candidates:
                    sc = pd.DataFrame(tt)
                    score_df = score_df.append(sc)
                # 这几个指标取最小值进行优化
                if criterion == 'llr':
                    score_df['score'] = -score_df['score']
                score_df = score_df[score_df['pvalue'] <= p_value_entry]
                score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行降序排序
                if len(score_df) > 0:
                    current_score = score_df.iloc[0]['score']
                    if current_score < best_score:
                        best_score = current_score
                        selected.append(score_df.iloc[0]['var'])
                        remaining_list.remove(score_df.iloc[0]['var'])
                        if score_df.iloc[0]['var'] in include_var:
                            flag_nonew = flag_nonew + 1
                            print('Limited steps')
                        else:
                            flag_nonew = 0
                            include_var.append(score_df.iloc[0]['var'])
                        if (add_inditor == True) & (remaining_inditor.empty == False):
                            remaining_inditor['add'] = remaining_inditor.apply(lambda x: 'Y' if (('woe_' + x['vara'] in selected) & ('woe_' + x['varb'] in selected)) else 'N', axis=1)
                            add = remaining_inditor[remaining_inditor['add'] == 'Y']
                            remaining_inditor = remaining_inditor[remaining_inditor['add'] == 'N']
                            def add_indictor(vara, varb, groupa, groupb):
                                df['ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb))] = df.apply(lambda x: 1 if (x['f_group_%s' % vara] == groupa) & (x['f_group_%s' % varb] == groupb) else 0, axis=1)
                                remaining_list.append('ind_f_group_%s_%s_f_group_%s_%s' % (vara, int(groupa), varb, int(groupb)))
                            if len(add) > 0:
                                add.apply(lambda x: add_indictor(vara=x['vara'], varb=x['varb'], groupa=x['group_a'],groupb=x['group_b']), axis=1)
                        flag_next = True
                        if show_step:  # 是否显示逐步回归过程
                            if criterion == 'llr':
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            else:
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            if intercept:  # 是否有截距
                                logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))

                            else:
                                logit_mod = sm.Logit(df[response], df[selected])
                            model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
                            record_list.append(model.summary2())
                        flag_e = True
                        while (flag_e == True) and (len(selected)>1):
                            if intercept:  # 是否有截距
                                logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))
                            else:
                                logit_mod = sm.Logit(df[response], df[selected])
                            result_full = logit_mod.fit(method='lbfgs', maxiter=100)
                            #                 def func(num):
                            result_list = []
                            for i in range(0, len(selected)):
                                candidate = selected[i]
                                if intercept:  # 是否有截距
                                    logit_mod = sm.Logit(df[response],
                                                         sm.add_constant(df[list(set(selected) - set([candidate]))]))
                                else:
                                    logit_mod = sm.Logit(df[response], df[list(set(selected) - set([candidate]))])
                                result = logit_mod.fit(method='lbfgs', maxiter=100)
                                var = candidate
                                pvalue = result_full.pvalues[candidate]
                                score = eval('result.' + criterion)
                                result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                            score_df = pd.DataFrame(result_list)
                            if criterion == 'llr':
                                score_df['score'] = -score_df['score']
                            # 这几个指标取最小值进行优化
                            score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行降序排序
                            if len(score_df) > 0:
                                current_score = score_df.iloc[0]['score']
                                if current_score < best_score:
                                    best_score = current_score
                                    selected.remove(score_df.iloc[0]['var'])
                                    remaining_list.append(score_df.iloc[0]['var'])
                                    flag_next = True
                                    if show_step:  # 是否显示逐步回归过程
                                        if criterion == 'llr':
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                        else:
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                        if intercept:  # 是否有截距
                                            logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))
                                        else:
                                            logit_mod = sm.Logit(df[response], df[selected])
                                        model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
                                        record_list.append(model.summary2())

                                    flag_e = True
                                elif len(score_df[score_df['pvalue'] > p_value_stay]) > 0:
                                    score_df = score_df.sort_values(by='pvalue', ascending=False)
                                    best_score = score_df.iloc[0]['score']
                                    p_big = score_df.iloc[0]['pvalue']
                                    selected.remove(score_df.iloc[0]['var'])
                                    remaining_list.append(score_df.iloc[0]['var'])

                                    if show_step:  # 是否显示逐步回归过程
                                        print('Delet %s, pvalue = %.5f' % (score_df.iloc[0]['var'], p_big))
                                        record_list.append('Delet %s, pvalue = %.5f' % (score_df.iloc[0]['var'], p_big))
                                        self.label_list = self.label_list + '\nDelet %s, pvalue = %.5f' % (score_df.iloc[0]['var'], p_big)
                                        self.text.set(self.label_list)
                                        mianframe.update()
                                        if intercept:  # 是否有截距
                                            logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))

                                        else:
                                            logit_mod = sm.Logit(df[response], df[selected])
                                        model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
                                        record_list.append(model.summary2())
                                    flag_e = True
                                else:
                                    if flag_nonew >= 3:
                                        flag_next = False
                                    flag_e = False
                            else:
                                if flag_nonew >= 3:
                                    flag_next = False
                                flag_e = False
                        else:
                            pass
                    else:
                        flag_next = False
                else:
                    flag_next = False

            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[selected]))

            else:
                logit_mod = sm.Logit(df[response], df[selected])
            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程

                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
        if direction == 'NO':
            remaining_list = var_clearn
            selected = []
            if show_step:
                print('\nLR starting:\n')
                record_list.append('\nLR starting:\n')
                self.label_list = self.label_list + '\nLR starting:\n'
                self.text.set(self.label_list)
                mianframe.update()
            # 当变量未剔除完，并且当前评分更新时进行循环
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[var_clearn]))
            else:
                logit_mod = sm.Logit(df[response], df[var_clearn])
            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程

                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
            tip.destroy()
        if direction == 'exist':
            remaining_list = var_clearn
            selected = []
            if show_step:
                print('\nLR starting:\n')
                record_list.append('\nLR starting:\n')
                self.label_list = self.label_list + '\nLR starting:\n'
                self.text.set(self.label_list)
                mianframe.update()
            # 当变量未剔除完，并且当前评分更新时进行循环
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[var_clearn]))
            else:
                logit_mod = sm.Logit(df[response], df[var_clearn])
            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程

                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
            tip.destroy()

        return [record_list, stepwise_model, modelvar_match_df]

    def grp_logistic_regression(self, mianframe,var, p_value_entry,p_value_stay, intercept, criterion, df, response, direction, show_step,
                                apply_restrict, n_job=None):
        tip = Toplevel(mianframe)
        self.text = StringVar()
        self.label_list=''
        self.label_list=self.label_list+'start...'
        self.text.set(self.label_list)
        lb = Label(tip, textvariable=self.text)
        lb.pack()
        mianframe.update()
        record_list = []
        modelvar_match_df = pd.DataFrame()
        if n_job == None:
            n_job = joblib.cpu_count() - 1
        flag_next = True
        group_varlist = ['f_group_' + x for x in var]
        select_variable = []
        for varable in group_varlist:
            mm = len(df[varable].unique())
            grouplist = list(df[varable].groupby(df[varable]).agg({'count'}).reset_index().sort_values(by='count')[varable][0:mm - 1])
            remind_group = \
                list(df[varable].groupby(df[varable]).agg({'count'}).reset_index().sort_values(by='count')[varable][mm - 1:mm])[0]
            temp = df[[varable]]
            variabellist = []
            for value in grouplist:
                df['%s_%s' % (varable, int(value))] = df[varable].apply(lambda x: 1 if x == value else 0)
                df['%s_%s' % (varable, int(value))] = df['%s_%s' % (varable, int(value))].astype('int8')
                variabellist.append('%s_%s' % (varable, int(value)))
                modelvar_match_df = modelvar_match_df.append(pd.DataFrame({'ori_var':varable[8:],'variable': varable, 'model_var': '%s_%s' % (varable, int(value)), 'group': int(value), 'var_type': 'ori'},
                    index=[1]))
            modelvar_match_df = modelvar_match_df.append(pd.DataFrame(
                {'ori_var':varable[8:],'variable': varable, 'model_var': '%s_%s' % (varable, remind_group), 'group': remind_group,
                 'var_type': 'ori'}, index=[1]))
            dic = {'variable': varable, 'list': variabellist, 'remind_group': remind_group}
            select_variable.append(dic)

        variable_df = pd.DataFrame(select_variable)

        if apply_restrict:
            remove = []
            for p in range(len(group_varlist) - 1):
                if len(df[group_varlist[p]].unique()) < 2:
                    remove = remove + [group_varlist[p]]
            var_clearn_t = set(group_varlist) - set(remove)
            var_clearn_t = list(var_clearn_t)
            corr_data = df[var_clearn_t].corr()
            for col in corr_data.columns:
                if len(corr_data[corr_data[col] > 0.99]) > 1:
                    corr_data[corr_data.index == col] = 0

                    remove = remove + [col]
            var_clearn = set(group_varlist) - set(remove)
            var_clearn = list(var_clearn)
            print('those varable will be not involve modeling because the high corr(>0.99) or zero performance', remove)
            record_list.append(
                'those varable will be not involve modeling because the high corr(>0.99) or zero performance')
            record_list.append(remove)
        else:
            var_clearn = group_varlist
        if direction == 'forward':
            print('\nforward_stepwise starting:\n')
            record_list.append('\nforward_stepwise starting:\n')
            self.label_list = self.label_list + '\nforward_stepwise starting:\n'
            self.text.set(self.label_list)
            mianframe.update()
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            best_score = np.inf
            if show_step:
                print('\nforward_stepwise starting:\n')
                record_list.append('\nforward_stepwise starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            while remaining_list != [] and flag_next == True:
                lent = math.ceil(len(remaining_list) / n_job)

                def func(num):
                    result_list = []
                    for i in range(num * lent, min((num + 1) * lent, len(remaining_list))):
                        candidate = remaining_list[i]
                        group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['list'].sum()
                        if selected != []:
                            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                        else:
                            group_variable_select = []
                        if intercept:  # 是否有截距
                            logit_mod = sm.Logit(df[response],
                                                 sm.add_constant(df[group_variable_select + group_variable_candidate]))
                        else:
                            logit_mod = sm.Logit(df[response], df[group_variable_select + group_variable_candidate])
                        result = logit_mod.fit(method='lbfgs', maxiter=100)
                        var = candidate
                        pvalue_df = pd.DataFrame(result.pvalues).reset_index()
                        pvalue = pvalue_df[pvalue_df['index'].isin(group_variable_candidate)][0].min()
                        #                     pvalue = result.pvalues[candidate]
                        score = eval('result.' + criterion)
                        result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                    return result_list

                scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(num) for num
                                                                                            in range(n_job))

                score_df = pd.DataFrame()
                for tt in scores_with_candidates:
                    sc = pd.DataFrame(tt)
                    score_df = score_df.append(sc)
                # 这几个指标取最小值进行优化
                if criterion == 'llr':
                    score_df['score'] = -score_df['score']
                score_df = score_df[score_df['pvalue'] <= p_value_entry]
                score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行升序排序
                if len(score_df) > 0:
                    current_score = score_df.iloc[0]['score']
                    if current_score < best_score:
                        best_score = current_score
                        selected.append(score_df.iloc[0]['var'])
                        remaining_list.remove(score_df.iloc[0]['var'])
                        flag_next = True
                        if show_step:  # 是否显示逐步回归过程
                            if criterion == 'llr':
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            else:
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                self.text.set(self.label_list)
                                mianframe.update()

                    else:
                        flag_next = False
                else:
                    flag_next = False
            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[group_variable_select]))

            else:
                logit_mod = sm.Logit(df[response], df[group_variable_select])

            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程
                print()
                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print()
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
        if direction == 'stepwise':
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            include_var = []
            flag_nonew = 0
            best_score = np.inf
            if show_step:
                print('\nstepwise starting:\n')
                record_list.append('\nstepwise starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            self.label_list = self.label_list +'\nstepwise starting:\n'
            self.text.set(self.label_list)
            mianframe.update()
            while remaining_list != [] and flag_next == True:
                lent = math.ceil(len(remaining_list) / n_job)
                def func(num):
                    result_list = []
                    for i in range(num * lent, min((num + 1) * lent, len(remaining_list))):
                        candidate = remaining_list[i]
                        group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['list'].sum()
                        if selected != []:
                            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                        else:
                            group_variable_select = []

                        if intercept:  # 是否有截距
                            logit_mod = sm.Logit(df[response],
                                                 sm.add_constant(df[group_variable_select + group_variable_candidate]))
                        else:
                            logit_mod = sm.Logit(df[response], df[group_variable_select + group_variable_candidate])
                        result = logit_mod.fit(method='lbfgs', maxiter=100)
                        var = candidate

                        pvalue_df = pd.DataFrame(result.pvalues).reset_index()
                        pvalue = pvalue_df[pvalue_df['index'].isin(group_variable_candidate)][0].min()
                        score = eval('result.' + criterion)
                        result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                    return result_list
                scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(num) for num
                                                                                            in range(n_job))
                score_df = pd.DataFrame()
                for tt in scores_with_candidates:
                    sc = pd.DataFrame(tt)
                    score_df = score_df.append(sc)
                # 这几个指标取最小值进行优化
                if criterion == 'llr':
                    score_df['score'] = -score_df['score']
                score_df = score_df[score_df['pvalue'] <= p_value_entry]
                score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行降序排序
                if len(score_df) > 0:
                    current_score = score_df.iloc[0]['score']
                    if current_score < best_score:
                        best_score = current_score
                        selected.append(score_df.iloc[0]['var'])
                        remaining_list.remove(score_df.iloc[0]['var'])

                        if score_df.iloc[0]['var'] in include_var:
                            flag_nonew = flag_nonew + 1
                            print('Limited steps')
                        else:
                            flag_nonew = 0
                            include_var.append(score_df.iloc[0]['var'])

                        flag_next = True
                        if show_step:  # 是否显示逐步回归过程
                            if criterion == 'llr':
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            else:
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                        flag_e = True
                        while (flag_e == True) and (len(selected)>1):
                            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                            if intercept:  # 是否有截距
                                logit_mod = sm.Logit(df[response], sm.add_constant(df[group_variable_select]))
                            else:
                                logit_mod = sm.Logit(df[response], df[group_variable_select])
                            result_full = logit_mod.fit(method='lbfgs', maxiter=100)
                            result_list = []
                            for i in range(0, len(selected)):
                                candidate = selected[i]
                                group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['list'].sum()
                                if selected != []:
                                    group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                                else:
                                    group_variable_select = []

                                if intercept:  # 是否有截距
                                    logit_mod = sm.Logit(df[response], sm.add_constant(df[list(set(group_variable_select) - set(group_variable_candidate))]))
                                else:
                                    logit_mod = sm.Logit(df[response], df[list(set(group_variable_select) - set(group_variable_candidate))])
                                result = logit_mod.fit(method='lbfgs', maxiter=100)
                                var = candidate
                                pvalue_df = pd.DataFrame(result_full.pvalues).reset_index()
                                pvalue = pvalue_df[pvalue_df['index'].isin(group_variable_candidate)][0].min()
                                score = eval('result.' + criterion)
                                result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                            score_df = pd.DataFrame(result_list)

                            # 这几个指标取最小值进行优化
                            if criterion == 'llr':
                                score_df['score'] = -score_df['score']
                            score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行降序排序
                            if len(score_df) > 0:
                                current_score = score_df.iloc[0]['score']
                                if current_score < best_score:
                                    best_score = current_score
                                    selected.remove(score_df.iloc[0]['var'])
                                    remaining_list.append(score_df.iloc[0]['var'])
                                    flag_next = True
                                    flag_e = True
                                    if show_step:  # 是否显示逐步回归过程
                                        if criterion == 'llr':
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                        else:
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                elif len(score_df[score_df['pvalue'] > p_value_stay]) > 0:
                                    score_df = score_df.sort_values(by='pvalue', ascending=False)
                                    best_score = score_df.iloc[0]['score']
                                    selected.remove(score_df.iloc[0]['var'])
                                    remaining_list.append(score_df.iloc[0]['var'])
                                    flag_e = True
                                    if show_step:  # 是否显示逐步回归过程
                                        if criterion == 'llr':
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                        else:
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                else:
                                    if flag_nonew >= 3:
                                        flag_next = False
                                    flag_e = False
                            else:
                                if flag_nonew >= 3:
                                    flag_next = False
                                flag_e = False
                    else:
                        flag_next = False
                else:
                    flag_next = False
            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[group_variable_select]))

            else:
                logit_mod = sm.Logit(df[response], df[group_variable_select])

            stepwise_model = logit_mod.fit(method='lbfgs')  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程
                print()
                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print()
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
        if direction == 'NO':
            print('\nLR starting:\n')
            record_list.append('\nforward_stepwise starting:\n')
            self.label_list = self.label_list + '\nforward_stepwise starting:\n'
            self.text.set(self.label_list)
            mianframe.update()
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            if show_step:
                print('\nLR starting:\n')
                record_list.append('\nLR starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            group_variable_select = variable_df[variable_df['variable'].isin(var_clearn)]['list'].sum()
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[group_variable_select]))
            else:
                logit_mod = sm.Logit(df[response], df[group_variable_select])
            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合

            if show_step:  # 是否显示逐步回归过程
                print()
                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print()
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
            tip.destroy()
        return [record_list, stepwise_model, modelvar_match_df]
    def grp_ind_logistic_regression(self, mianframe,var, p_value_entry,p_value_stay, intercept, criterion, df, response, direction, show_step,
                                apply_restrict, n_job=None):
        tip = Toplevel(mianframe)
        self.text = StringVar()
        self.label_list=''
        self.label_list=self.label_list+'start...'
        self.text.set(self.label_list)
        lb = Label(tip, textvariable=self.text)
        lb.pack()
        mianframe.update()
        record_list = []
        modelvar_match_df = pd.DataFrame()
        if n_job == None:
            n_job = joblib.cpu_count() - 1
        flag_next = True
        group_varlist = ['f_group_' + x for x in var]
        select_variable = []
        for varable in group_varlist:
            mm = len(df[varable].unique())
            grouplist = list(df[varable].groupby(df[varable]).agg({'count'}).reset_index().sort_values(by='count')[varable][0:mm - 1])
            remind_group = \
                list(df[varable].groupby(df[varable]).agg({'count'}).reset_index().sort_values(by='count')[varable][mm - 1:mm])[0]
            temp = df[[varable]]
            variabellist = []
            for value in grouplist:
                df['%s_%s' % (varable, int(value))] = df[varable].apply(lambda x: 1 if x == value else 0)
                df['%s_%s' % (varable, int(value))] = df['%s_%s' % (varable, int(value))].astype('int8')
                variabellist.append('%s_%s' % (varable, int(value)))
                modelvar_match_df = modelvar_match_df.append(pd.DataFrame(
                    {'ori_var':varable[8:],'variable': varable, 'model_var': '%s_%s' % (varable, int(value)), 'group': int(value), 'var_type': 'ori'},
                    index=[1]))
            modelvar_match_df = modelvar_match_df.append(pd.DataFrame(
                {'ori_var':varable[8:],'variable': varable, 'model_var': '%s_%s' % (varable, remind_group), 'group': remind_group,
                 'var_type': 'ori'}, index=[1]))
            dic = {'variable': varable, 'list': variabellist, 'remind_group': remind_group}
            select_variable.append(dic)

        variable_df = pd.DataFrame(select_variable)

        if apply_restrict:
            remove = []
            for p in range(len(group_varlist) - 1):
                if len(df[group_varlist[p]].unique()) < 2:
                    remove = remove + [group_varlist[p]]
            var_clearn_t = set(group_varlist) - set(remove)
            var_clearn_t = list(var_clearn_t)
            corr_data = df[var_clearn_t].corr()
            for col in corr_data.columns:
                if len(corr_data[corr_data[col] > 0.99]) > 1:
                    corr_data[corr_data.index == col] = 0

                    remove = remove + [col]
            var_clearn = set(group_varlist) - set(remove)
            var_clearn = list(var_clearn)
            print('those varable will be not involve modeling because the high corr(>0.99) or zero performance', remove)
            record_list.append(
                'those varable will be not involve modeling because the high corr(>0.99) or zero performance')
            record_list.append(remove)
        else:
            var_clearn = group_varlist
        if direction == 'forward':
            print('\nforward_stepwise starting:\n')
            record_list.append('\nforward_stepwise starting:\n')
            self.label_list = self.label_list + '\nforward_stepwise starting:\n'
            self.text.set(self.label_list)
            mianframe.update()
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            best_score = np.inf
            if show_step:
                print('\nforward_stepwise starting:\n')
                record_list.append('\nforward_stepwise starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            while remaining_list != [] and flag_next == True:
                lent = math.ceil(len(remaining_list) / n_job)

                def func(num):
                    result_list = []
                    for i in range(num * lent, min((num + 1) * lent, len(remaining_list))):
                        candidate = remaining_list[i]
                        group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['list'].sum()
                        if selected != []:
                            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                        else:
                            group_variable_select = []
                        if intercept:  # 是否有截距
                            logit_mod = sm.Logit(df[response],
                                                 sm.add_constant(df[group_variable_select + group_variable_candidate]))
                        else:
                            logit_mod = sm.Logit(df[response], df[group_variable_select + group_variable_candidate])
                        result = logit_mod.fit(method='lbfgs', maxiter=100)
                        var = candidate
                        pvalue_df = pd.DataFrame(result.pvalues).reset_index()
                        pvalue = pvalue_df[pvalue_df['index'].isin(group_variable_candidate)][0].min()
                        #                     pvalue = result.pvalues[candidate]
                        score = eval('result.' + criterion)
                        result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                    return result_list

                scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(num) for num
                                                                                            in range(n_job))

                score_df = pd.DataFrame()
                for tt in scores_with_candidates:
                    sc = pd.DataFrame(tt)
                    score_df = score_df.append(sc)
                # 这几个指标取最小值进行优化
                if criterion == 'llr':
                    score_df['score'] = -score_df['score']
                score_df = score_df[score_df['pvalue'] <= p_value_entry]
                score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行升序排序
                if len(score_df) > 0:
                    current_score = score_df.iloc[0]['score']
                    if current_score < best_score:
                        best_score = current_score
                        selected.append(score_df.iloc[0]['var'])
                        remaining_list.remove(score_df.iloc[0]['var'])
                        flag_next = True
                        if show_step:  # 是否显示逐步回归过程
                            if criterion == 'llr':
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            else:
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                    else:
                        flag_next = False
                else:
                    flag_next = False
            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[group_variable_select]))

            else:
                logit_mod = sm.Logit(df[response], df[group_variable_select])

            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程
                print()
                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print()
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
        if direction == 'stepwise':
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            include_var = []
            remove_grp_variable=[]
            flag_nonew = 0
            best_score = np.inf
            if show_step:
                print('\nstepwise starting:\n')
                record_list.append('\nstepwise starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            self.label_list = self.label_list +'\nstepwise starting:\n'
            self.text.set(self.label_list)
            mianframe.update()
            while remaining_list != [] and flag_next == True:
                lent = math.ceil(len(remaining_list) / n_job)
                def func(num):
                    result_list = []
                    for i in range(num * lent, min((num + 1) * lent, len(remaining_list))):
                        candidate = remaining_list[i]
                        group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['list'].sum()
                        if selected != []:
                            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                        else:
                            group_variable_select = []

                        if intercept:  # 是否有截距
                            logit_mod = sm.Logit(df[response],sm.add_constant(df[list(set(group_variable_select + group_variable_candidate)-set(remove_grp_variable))]))
                        else:
                            logit_mod = sm.Logit(df[response], df[list(set(group_variable_select + group_variable_candidate)-set(remove_grp_variable))])
                        result = logit_mod.fit(method='lbfgs', maxiter=100)
                        var = candidate

                        pvalue_df = pd.DataFrame(result.pvalues).reset_index()
                        pvalue = pvalue_df[pvalue_df['index'].isin(group_variable_candidate)][0].min()
                        score = eval('result.' + criterion)
                        result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                    return result_list
                scores_with_candidates = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(num) for num
                                                                                            in range(n_job))
                score_df = pd.DataFrame()
                for tt in scores_with_candidates:
                    sc = pd.DataFrame(tt)
                    score_df = score_df.append(sc)
                # 这几个指标取最小值进行优化
                if criterion == 'llr':
                    score_df['score'] = -score_df['score']
                score_df = score_df[score_df['pvalue'] <= p_value_entry]
                score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行降序排序
                if len(score_df) > 0:
                    current_score = score_df.iloc[0]['score']
                    if current_score < best_score:
                        best_score = current_score
                        selected.append(score_df.iloc[0]['var'])
                        remaining_list.remove(score_df.iloc[0]['var'])

                        if score_df.iloc[0]['var'] in include_var:
                            flag_nonew = flag_nonew + 1
                            print('Limited steps')
                        else:
                            flag_nonew = 0
                            include_var.append(score_df.iloc[0]['var'])

                        flag_next = True
                        if show_step:  # 是否显示逐步回归过程
                            if criterion == 'llr':
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                            else:
                                print('Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                record_list.append(
                                    'Adding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                self.label_list = self.label_list + '\nAdding %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                self.text.set(self.label_list)
                                mianframe.update()
                        flag_e = True
                        while (flag_e == True) and (len(selected)>1):
                            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                            if intercept:  # 是否有截距
                                logit_mod = sm.Logit(df[response], sm.add_constant(df[list(set(group_variable_select) - set(remove_grp_variable))]))
                            else:
                                logit_mod = sm.Logit(df[response], df[list(set(group_variable_select) - set(remove_grp_variable))])
                            result_full = logit_mod.fit(method='lbfgs', maxiter=100)
                            result_list = []
                            for i in range(0, len(list(set(group_variable_select) - set(remove_grp_variable)))):
                                candidate = list(set(group_variable_select) - set(remove_grp_variable))[i]
                                #group_variable_candidate = variable_df[variable_df['variable'].isin([candidate])]['list'].sum()
                                # if selected != []:
                                #     group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
                                # else:
                                #     group_variable_select = []

                                if intercept:  # 是否有截距
                                    logit_mod = sm.Logit(df[response], sm.add_constant(df[list(set(group_variable_select)- set([candidate])-set(remove_grp_variable))]))
                                else:
                                    logit_mod = sm.Logit(df[response], df[list(set(group_variable_select) - set([candidate])-set(remove_grp_variable))])
                                result = logit_mod.fit(method='lbfgs', maxiter=100)
                                var = candidate
                                pvalue =result_full.pvalues[candidate]
                                score = eval('result.' + criterion)
                                result_list.append({'var': var, 'pvalue': pvalue, 'score': score})
                            score_df = pd.DataFrame(result_list)

                            # 这几个指标取最小值进行优化
                            if criterion == 'llr':
                                score_df['score'] = -score_df['score']
                            score_df = score_df.sort_values(by='score', ascending=True)  # 对评分列表进行降序排序
                            if len(score_df) > 0:
                                current_score = score_df.iloc[0]['score']
                                if current_score < best_score:
                                    best_score = current_score
                                    remove_grp_variable.append(score_df.iloc[0]['var'])
                                    # selected.remove(score_df.iloc[0]['var'])
                                    # remaining_list.append(score_df.iloc[0]['var'])
                                    flag_next = True
                                    flag_e = True
                                    if show_step:  # 是否显示逐步回归过程
                                        if criterion == 'llr':
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                        else:
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                elif len(score_df[score_df['pvalue'] > p_value_stay]) > 0:
                                    score_df = score_df.sort_values(by='pvalue', ascending=False)
                                    best_score = score_df.iloc[0]['score']
                                    remove_grp_variable.append(score_df.iloc[0]['var'])
                                    # selected.remove(score_df.iloc[0]['var'])
                                    # remaining_list.append(score_df.iloc[0]['var'])
                                    flag_e = True
                                    if show_step:  # 是否显示逐步回归过程
                                        if criterion == 'llr':
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, -best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                        else:
                                            print('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            record_list.append('Delet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score))
                                            self.label_list = self.label_list + '\nDelet %s, %s = %.3f' % (score_df.iloc[0]['var'], criterion, best_score)
                                            self.text.set(self.label_list)
                                            mianframe.update()
                                else:
                                    if flag_nonew >= 3:
                                        flag_next = False
                                    flag_e = False
                            else:
                                if flag_nonew >= 3:
                                    flag_next = False
                                flag_e = False
                    else:
                        flag_next = False
                else:
                    flag_next = False
            group_variable_select = variable_df[variable_df['variable'].isin(selected)]['list'].sum()
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response],
                                     sm.add_constant(df[list(set(group_variable_select) - set(remove_grp_variable))]))
            else:
                logit_mod = sm.Logit(df[response], df[list(set(group_variable_select) - set(remove_grp_variable))])
            stepwise_model = logit_mod.fit(method='lbfgs')  # 最优模型拟合
            tip.destroy()
            if show_step:  # 是否显示逐步回归过程
                print()
                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print()
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
        if direction == 'NO':
            print('\nLR starting:\n')
            record_list.append('\nforward_stepwise starting:\n')
            self.label_list = self.label_list + '\nforward_stepwise starting:\n'
            self.text.set(self.label_list)
            mianframe.update()
            remaining_list = var_clearn
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            if show_step:
                print('\nLR starting:\n')
                record_list.append('\nLR starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            group_variable_select = variable_df[variable_df['variable'].isin(var_clearn)]['list'].sum()
            if intercept:  # 是否有截距
                logit_mod = sm.Logit(df[response], sm.add_constant(df[group_variable_select]))
            else:
                logit_mod = sm.Logit(df[response], df[group_variable_select])
            stepwise_model = logit_mod.fit(method='lbfgs', maxiter=100)  # 最优模型拟合

            if show_step:  # 是否显示逐步回归过程
                print()
                print('模型变量共', len(selected), '个')
                record_list.append(['模型变量共', len(selected), '个'])
                print()
                print('模型变量列表是：', selected)
                record_list.append('模型变量列表是：')
                record_list.append(selected)
                print('\n', stepwise_model.summary2())
                record_list.append(stepwise_model.summary2())
            tip.destroy()
        return [record_list, stepwise_model, modelvar_match_df]