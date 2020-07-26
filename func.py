import pandas as pd
from .base import group_func
import numpy as np
from joblib import Parallel, delayed
import joblib

import math
group_func=group_func()

class binning():
    def fit_bin(self, data, varnum, varchar, target, s_bin_num=20, special_code=pd.DataFrame(), min_num=500, min_pct=0.05, n_job=None, criterion='entropy', splitter='best', max_depth=6, min_samples_leaf=500,max_leaf_nodes=9):
        if n_job == None:
            n_job = joblib.cpu_count() - 1
        #        colnum=data[colmns].select_dtypes(include=['float','int8','int16','int32','int64']).columns.values.tolist()
        #        colchar=data[colmns].select_dtypes(include=['object']).columns.values.tolist()
        column = varnum + varchar
        lenp = math.ceil(len(column) / (n_job))

        def func(part):
            temp1 = pd.DataFrame()
            temp2 = []
            for i in range(lenp * part, min(lenp * (part + 1), len(column))):
                col = column[i]
                if special_code.empty == False:
                    specialcode_list = list(special_code[special_code['variable'] == col]['value'])
                else:
                    specialcode_list = []
                inputdata = data[[col, target]]
                if col in varchar:
                    out, outdata = group_func.charvar(inputdata=inputdata, col=col, min_num=min_num, min_pct=min_pct,
                                                      target=target, criterion=criterion, splitter=splitter,
                                                      max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                      max_leaf_nodes=max_leaf_nodes)
                else:
                    out, outdata = group_func.numericvar(inputdata=inputdata, col=col,
                                                         specialcode_list=specialcode_list, s_bin_num=s_bin_num,
                                                         target=target, criterion=criterion, splitter=splitter,
                                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                         max_leaf_nodes=max_leaf_nodes)
                out = out.drop_duplicates()
                outdata = outdata.drop_duplicates()
                temp1 = pd.concat([temp1, out])
                temp2.append(outdata)
            return temp1, temp2

        #         if len(data.columns)*len(data)>3*10**8:
        #             print('your dataset is quiet large Parallel may not work , n_job=1 is recommend')
        results = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(part) for part in range(n_job))
        self.group_info = pd.DataFrame()
        self.group_info_data = data.copy()
        for i in range(len(results)):
            te = results[i][0]
            self.group_info = pd.concat([self.group_info, te], sort=True)
            tf = results[i][1]
            for pp in range(len(tf)):
                one = tf[pp]
                self.group_info_data = pd.merge(self.group_info_data, one, how='left')
        return self.group_info, self.group_info_data

    def fit_bin_existing(self, data, varnum, varchar, target, group_info, data_only=False, n_job=None):

        if n_job == None:
            n_job = joblib.cpu_count() - 2
        data_col = list(data.columns)
        group_col = varnum + varchar
        column = list(set(data_col).intersection(set(group_col)))
        lenp = math.ceil(len(column) / (n_job))


        def func(part):
            temp1 = pd.DataFrame()
            temp2 = []
            for i in range(lenp * part, min(lenp * (part + 1), len(column))):
                col = column[i]
                if data_only == False:
                    inputdata = data[[col, target]]
                else:
                    inputdata = data[[col]]
                group_info_col = group_info[group_info['variable_name'] == col]
                if data_only == False:
                    if col in varchar:

                        out, outdata = group_func.charvarexist(group_info_old=group_info_col, data_only=data_only,
                                                               inputdata=inputdata, col=col, target=target)
                    else:
                        out, outdata = group_func.numericexist(group_info_old=group_info_col, data_only=data_only,
                                                               inputdata=inputdata, col=col, target=target, modify=False,
                                                               add_value=0)
                    out = out.drop_duplicates()
                    outdata = outdata.drop_duplicates()
                    temp1 = pd.concat([temp1, out])
                    temp2.append(outdata)
                else:
                    if col in varchar:

                        outdata = group_func.charvarexist(group_info_old=group_info_col, data_only=data_only,
                                                          inputdata=inputdata, col=col, target=target)
                    else:
                        outdata = group_func.numericexist(group_info_old=group_info_col, data_only=data_only,
                                                          inputdata=inputdata, col=col, target=target, modify=False,
                                                          add_value=0)

                    outdata = outdata.drop_duplicates()
                    temp2.append(outdata)
                    # return outdata
            if data_only==False:
                return temp1, temp2
            else :
                return temp2

        results = Parallel(n_jobs=n_job, max_nbytes=None, verbose=5)(delayed(func)(part) for part in range(n_job))
        if data_only == False:
            self.group_info = pd.DataFrame()
            self.group_info_data = data.copy()
            for i in range(len(results)):
                te = results[i][0]
                self.group_info = pd.concat([self.group_info, te], sort=True)
                tf = results[i][1]
                for pp in range(len(tf)):
                    one = tf[pp]
                    self.group_info_data = pd.merge(self.group_info_data, one, how='left')
            return self.group_info, self.group_info_data
        else:

            self.group_info_data = data.copy()
            for i in range(len(results)):
                tf = results[i]
                for pp in range(len(tf)):
                    one = tf[pp]
                    self.group_info_data = pd.merge(self.group_info_data, one, how='left')
            return self.group_info_data


    def report(self, group_info, varnum, varchar):
        col=varnum+varchar

        group_info['miss_rate'] = group_info['miss_count'] / (group_info['miss_count'] + group_info['count'])
        group_info['total_count'] = (group_info['miss_count'] + group_info['count'])
        group_info = group_info[group_info['variable_name'].isin(col)]
        if len(varnum) < 1:
            base = group_info[['variable_name', 'f_group', 'f_Bad_rate', 'f_N_bad', 'f_N_obs', 'woe', 'iv',
                               'miss_rate']].drop_duplicates()

        else:
            base = group_info[
                ['variable_name', 'f_group', 'f_Bad_rate', 'f_N_bad', 'f_N_obs', 'woe', 'iv', 'miss_rate']].drop_duplicates()
        if len(varnum) >= 1:
            num_data_report = group_info[group_info['variable_name'].isin(varnum)]
            label_num = num_data_report.groupby(['variable_name', 'f_group']).agg(
                {'s_min': 'min', 's_max': 'max', 'miss_s': 'max'}).reset_index().rename(
                {'s_min': 'f_minlabel', 's_max': 'f_maxlabel', 'miss_s': 'miss_f'},
                axis=1)
            label_num['f_minlabel'] = round(label_num['f_minlabel'], 4)
            label_num['f_maxlabel'] = round(label_num['f_maxlabel'], 4)
            label_num['labelA'] = label_num.apply(
                lambda x: '%s < %s <= %s ' % (x['f_minlabel'], x['variable_name'], x['f_maxlabel']), axis=1)
            if len(num_data_report[num_data_report['value'].isnull() == False])>0:
                label_num_miss = num_data_report[num_data_report['value'].isnull() == False]
                label_num_miss['value'] = label_num_miss['value'].astype('str')
                label_num_miss['value'] = label_num_miss['value'] + ' ; '
                label_num_miss = label_num_miss.groupby(['variable_name', 'f_group'])['value'].sum().reset_index().rename(
                    {'value': 'labelB'}, axis=1)
                label = pd.merge(label_num, label_num_miss, how='left', on=['variable_name', 'f_group'])
                label = label.astype({'labelB': 'str'})
            else:
                label = label_num
            label['label'] = label.apply(
                lambda x: x['labelB'] if (np.isnan(x['f_maxlabel'])) & (np.isnan(x['f_minlabel'])) & (
                            x['miss_f'] == True)
                else x['labelA'] if (np.isnan(x['f_maxlabel']) == False) & (np.isnan(x['f_minlabel']) == False) & (
                        x['miss_f'] == False)
                else '[' + str(x['labelB']) + ']' + x['labelA'] if (np.isnan(x['f_maxlabel']) == False) & (
                        np.isnan(x['f_minlabel']) == False) & (x['miss_f'] == True)
                else '', axis=1)

            labellist_num = label[['variable_name', 'f_group', 'label']]
        else:
            labellist_num = pd.DataFrame()
        if len(varchar) >= 1:
            char_data_report = group_info[group_info['variable_name'].isin(varchar)]
            char_data_report['value'] = char_data_report['value'].astype('str')
            char_data_report['value'] = char_data_report['value'] + ' ; '
            labellist_char = char_data_report.groupby(['variable_name', 'f_group'])['value'].sum().reset_index().rename(
                {'value': 'label'}, axis=1)
        else:
            labellist_char = pd.DataFrame()
        label = pd.concat([labellist_char, labellist_num])
        label["f_group"] = pd.to_numeric(label["f_group"])
        reportA = pd.merge(base, label, how='left', on=['variable_name', 'f_group'])
        reportA.sort_values(by='iv', ascending=False)
        reportA['f_Bad_rate'] = reportA.apply(lambda x: "%.2f%%" % (x['f_Bad_rate'] * 100), axis=1)
        reportA['miss_rate'] = reportA.apply(lambda x: "%.2f%%" % (x['miss_rate'] * 100), axis=1)

        f_group_report = reportA

        # s_group
        group_info['miss_rate'] = group_info['miss_count'] / (group_info['miss_count'] + group_info['count'])
        group_info['total_count'] = (group_info['miss_count'] + group_info['count'])


        if len(varnum) < 1:
            base = group_info[['variable_name', 's_Bad_rate', 's_N_bad', 's_N_obs', 'f_group', 'value',
                               'miss_s']].drop_duplicates()

        else:
            base = group_info[['variable_name', 's_group', 's_Bad_rate', 's_N_bad', 's_N_obs', 'f_group', 'value',
                               'miss_s']].drop_duplicates()

        if len(varnum) >= 1:
            num_data_report = group_info[group_info['variable_name'].isin(varnum)]
            label_num = num_data_report.groupby(['variable_name', 's_group', 'miss_s']).agg(
                {'s_min': 'min', 's_max': 'max'}).reset_index().rename({'s_min': 's_minlabel', 's_max': 's_maxlabel'},
                                                                       axis=1)
            label_num['s_minlabel'] = round(label_num['s_minlabel'], 4)
            label_num['s_maxlabel'] = round(label_num['s_maxlabel'], 4)
            label_num['labelA'] = label_num.apply(
                lambda x: '%s < %s <= %s ' % (x['s_minlabel'], x['variable_name'], x['s_maxlabel']), axis=1)
            if len(num_data_report[num_data_report['value'].isnull() == False])>0:
                label_num_miss = num_data_report[num_data_report['value'].isnull() == False]
                label_num_miss['value'] = label_num_miss['value'].astype('str')
                label_num_miss['value'] = label_num_miss['value'] + ' ; '
                label_num_miss = label_num_miss.groupby(['variable_name', 's_group'])['value'].sum().reset_index().rename(
                    {'value': 'labelB'}, axis=1)
                label = pd.merge(label_num, label_num_miss, how='left', on=['variable_name', 's_group'])
            else:
                label = label_num
            label['label'] = label.apply(
                lambda x: x['labelB'] if (np.isnan(x['s_maxlabel'])) & (np.isnan(x['s_minlabel'])) & (
                            x['miss_s'] == True)
                else x['labelA'] if (np.isnan(x['s_maxlabel']) == False) & (np.isnan(x['s_minlabel']) == False) & (
                        x['miss_s'] == False)
                else '[' + x['labelB'] + ']' + x['labelA'] if (np.isnan(x['s_maxlabel']) == False) & (
                        np.isnan(x['s_minlabel']) == False) & (x['miss_s'] == True)
                else '', axis=1)
            labellist_num = label[['variable_name', 's_group', 'label']]
            labellist_num["s_group"] = pd.to_numeric(labellist_num["s_group"])
            reportA = pd.merge(base, labellist_num, how='right', on=['variable_name', 's_group'])
        else:
            reportA = pd.DataFrame()
        if len(varchar) >= 1:
            char_data_report = group_info[group_info['variable_name'].isin(varchar)]
            char_data_report['value'] = char_data_report['value'].astype('str')
            labellist_char = char_data_report.groupby(['variable_name'])['value'].sum().reset_index()
            reportB = char_data_report[
                ['variable_name', 'f_group', 's_Bad_rate', 's_N_bad', 's_N_obs', 'value', 'miss_s']]
            reportB['label'] = reportB['value']
        else:
            reportB = pd.DataFrame()

        ss = pd.concat([reportA, reportB])
        ss['s_Bad_rate'] = ss.apply(lambda x: "%.2f%%" % (x['s_Bad_rate'] * 100), axis=1)
        s_group_report = pd.merge(ss, f_group_report.drop(columns=['label']), how='left',
                                  on=['variable_name', 'f_group'])
        return s_group_report, f_group_report