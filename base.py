import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math



# the function is use for fine group binning with equivalent quantities sample. The output of this function is the bound of each    group(missing value will ignored)
# Parameter explain:
# Inputdata: the raw data you want fine bining 
# Col: the variable you want binning
# s_bin_num: the number of bins you wanted. 
class group_func():
    def __init__(self,):
        a=1
    def num_finebin_group(self, inputdata, col, s_bin_num, specialcode_list):
        if specialcode_list != []:
            inputdata = inputdata[inputdata.isin({col: specialcode_list}) == False]
        sort_df = inputdata[col][inputdata[col].isnull() == False]
        if len(sort_df.unique()) <= 1:
            old_list = set([float('-inf'), float('inf')])
        elif len(sort_df.unique()) <= s_bin_num:
            old_list = set(list(sort_df.unique()))
            old_list.remove(max(old_list))
            old_list.remove(min(old_list))
            old_list.add(float('-inf'))
            old_list.add(float('inf'))
        else:
            old_list = set()
            num = sort_df.size
            sort_df = sort_df.sort_values(ascending=True).reset_index().drop(columns='index')
            for i in range(1, s_bin_num):
                loca = int((i / s_bin_num) * num)
                value = sort_df.iloc[loca].values[0]
                old_list.add(value)
            old_list=old_list-set([max(old_list)])
            old_list.add(float('-inf'))
            old_list.add(float('inf'))
        new_list = list(old_list)
        new_list.sort()
        new_s_group = []
        for i in range(len(new_list) - 1):
            temp = {'s_group': i, 's_min': new_list[i], 's_max': new_list[i + 1]}
            new_s_group.append(temp)
        s_group_map = pd.DataFrame(new_s_group)
        return s_group_map

    # The function is use for produce the finebinning data set whit provide binning info(which the output of “finebin_group”
    # Parameter explain:
    # Inputdata: the raw data you want fine bining 
    # Col: the variable you want binning
    # Group_data: the data set of binning from finebin_group 

    def binning(self, group_data, inputdata, col, inputmax, inputmin, inputgroup, specialcode_list):
        s_group_data = pd.DataFrame()
        group_data = group_data.reset_index()
        if specialcode_list != []:
            # for sp_value in specialcode_list:
            #     inputdata = inputdata[inputdata[col]!=sp_value]
            inputdata = inputdata[inputdata.isin({col: specialcode_list}) == False]
        inputdata = inputdata.loc[inputdata[col].isnull() == False]
        for lins in range(len(group_data)):
            temp = inputdata.copy()
            temp[inputgroup] = group_data.loc[lins, inputgroup]
            temp[inputmin] = group_data.loc[lins, inputmin]
            temp[inputmax] = group_data.loc[lins, inputmax]
            temp_data = temp[((temp[col] <= temp[inputmax]) & (temp[col] > temp[inputmin]))]
            s_group_data = pd.concat([s_group_data, temp_data])

        # s_group_data=s_group_data.drop(columns=[inputmin,inputmax])
        # s_group_data.loc[s_group_data[col].isnull(),inputgroup]=-1

        return s_group_data

    def char_finebin_group(self, inputdata, col, target, min_num, min_pct):
        map_df = inputdata.groupby(col)[target].agg({'mean', 'count'}).reset_index()
        total = map_df['count'].sum()
        largermap = map_df[(map_df['count'] > min_num) | (map_df['count'] > total * min_pct)].sort_values(by='mean',
                                                                                                          ascending=True).reset_index().reset_index().rename(
            {'level_0': 's_group'}, axis=1).drop(columns='index')
        samallmap = map_df[(map_df['count'] <= min_num) & (map_df['count'] <= total * min_pct)]
        samallmap['s_group'] = samallmap.apply(lambda x: -2, axis=1)
        # samallmap.loc[:,'s_group']=-2 
        charmap = pd.concat([samallmap, largermap], sort=True)
        charmap['s_min'] = charmap['s_group']
        charmap['s_max'] = charmap['s_group']
        s_group_map = charmap.drop(columns=['mean', 'count'])
        del map_df, largermap, samallmap, charmap
        return s_group_map

    def char_finebinning(self, group_data, inputdata, col):
        s_group_data = pd.merge(inputdata, group_data, how='left', on=col)
        s_group_data.loc[s_group_data[col].isnull(), 's_group'] = -1
        return s_group_data

    # The function is use for rough binning with decision tree method on fine grouped sample. The output is a data about which fine group belong to rough group.
    # Parameter explain:
    # Inputdata: the raw data you want rough bing which shou include the target and fine group
    # s_group_col: the fine group columns
    # col: the original variable 
    # target：target for group and decision tree
    # criterion: the evaluation of decision tree: ‘gini’ entropy
    # splitter: default  'best'  
    # max_depth: maximum depth of decision tree 
    # min_samples_leaf:  minimum num of each leaf
    # max_leaf_nodes: maximum num of group
    # above parameter from sklearn decision tree , for more details can check the guide of  sklearn 

    # 粗分组
    def roughbin_group(self, inputdata, s_group_col, col, target, specialcode_list, criterion, splitter, max_depth,
                       min_samples_leaf, max_leaf_nodes):
        if specialcode_list != []:
            # for sp_value in specialcode_list:
            #     inputdata = inputdata[inputdata[col]!=sp_value]
            inputdata = inputdata[inputdata.isin({col: specialcode_list}) == False]
        inputdata = inputdata[inputdata[col].isnull() == False]
        if len(inputdata)>2:
            X_train = inputdata[[s_group_col]]
            y_train = inputdata[[target]]
            dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
            dtc.fit(X_train, y_train)
            n_nodes = dtc.tree_.node_count
            children_left = dtc.tree_.children_left
            children_right = dtc.tree_.children_right
            feature = dtc.tree_.feature
            threshold = dtc.tree_.threshold
            max_v = []
            min_v = []
            min_v.append(float('-inf'))
            max_v.append(float('inf'))
            df = pd.DataFrame()
            if n_nodes > 1:
                for i in range(1, n_nodes):
                    for p in range(n_nodes):
                        if children_left[p] == i:
                            max_vv = threshold[p]
                            min_vv = min_v[p]
                            min_v.append(min_vv)
                            max_v.append(max_vv)

                    for m in range(n_nodes):
                        if children_right[m] == i:
                            min_vv = threshold[m]
                            max_vv = max_v[m]
                            min_v.append(min_vv)
                            max_v.append(max_vv)
                    if children_left[i] == -1 and children_right[i] == -1:
                        if max_vv == np.nan:
                            max_vv = float('inf')
                        if min_vv == np.nan:
                            max_vv = float('-inf')
                        a = pd.DataFrame({'node_id': i, 'f_max': max_vv, 'f_min': min_vv, 'name': feature[i]}, index=[0])
                        df = df.append(a)
            if n_nodes == 1:
                df = pd.DataFrame({'node_id': 1, 'f_max': float('inf'), 'f_min': float('-inf'), 'name': col}, index=[0])
        else:
            df = pd.DataFrame({'node_id': 1, 'f_max': float('inf'), 'f_min': float('-inf'), 'name': col}, index=[0])
        rough_group = df.sort_values(by='f_max').reset_index().reset_index()[['level_0', 'f_max', 'f_min']].rename(
            {'level_0': 'f_group'}, axis=1)
        return rough_group

    def numericvar(self, inputdata, col, s_bin_num, target, criterion, splitter, max_depth, min_samples_leaf,
                   max_leaf_nodes, specialcode_list):

        s_group_map = self.num_finebin_group(inputdata=inputdata, col=col, s_bin_num=s_bin_num,
                                             specialcode_list=specialcode_list)
        s_group_data = self.binning(group_data=s_group_map, inputdata=inputdata, col=col, inputmax='s_max',
                                    inputmin='s_min', inputgroup='s_group', specialcode_list=specialcode_list)
        f_group_map_pre = self.roughbin_group(inputdata=s_group_data, s_group_col='s_group',
                                              specialcode_list=specialcode_list, col=col, target=target,
                                              criterion=criterion, splitter=splitter, max_depth=max_depth,
                                              min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
        f_group_data_pre = self.binning(group_data=f_group_map_pre, inputdata=s_group_map, col='s_group',
                                        inputmax='f_max', inputmin='f_min', inputgroup='f_group', specialcode_list=[])
        f_group_map = f_group_data_pre.groupby('f_group').agg({'s_min': 'min', 's_max': 'max'}).reset_index().rename(
            {'s_min': 'f_min', 's_max': 'f_max'}, axis=1)
        f_group_data = self.binning(group_data=f_group_map, inputdata=s_group_data, col=col, inputmax='f_max',
                                    inputmin='f_min', inputgroup='f_group', specialcode_list=specialcode_list)
        f_group_data['value'] = np.nan

        group_info_base = pd.merge(f_group_data_pre[['s_group', 's_min', 's_max', 'f_group']], f_group_map, how='left',
                                   on=['f_group'])
        group_info_base['miss_s'] = False
        group_info_base['miss_f'] = False
        group_info_base['value'] = np.nan
        if len(inputdata[inputdata[col].isnull()]) > 0:
            miss_data = inputdata[inputdata[col].isnull()]
            miss_data.loc[:, 's_group'] = -1
            miss_data.loc[:, 'f_group'] = -1
            miss_data['value'] = 'miss'
            tempg = pd.DataFrame({'s_group': -1, 'f_group': -1, 'value': 'miss', 'miss_s': True, 'miss_f': True},
                                 index=[0])
            f_group_data = f_group_data.append(miss_data)
            group_info_base = group_info_base.append(tempg)
        if specialcode_list != []:
            i = -2
            for special_value in specialcode_list:
                temp = inputdata[inputdata[col] == special_value].copy()
                temp['s_group'] = i
                temp['f_group'] = i
                temp['value'] = special_value
                temps = pd.DataFrame(
                    {'s_group': i, 'f_group': i, 'value': special_value, 'miss_s': True, 'miss_f': True}, index=[0])
                f_group_data = f_group_data.append(temp)
                group_info_base = group_info_base.append(temps)
                i = i - 1

        f_group_data['miss'] = f_group_data['s_group'] < 0
        tt = f_group_data.groupby(['s_group', 'f_group']).agg(
            {target: ['mean', 'count', 'sum'], 'miss': 'max', 's_max': 'max', 's_min': 'min',
             'value': 'max'}).reset_index()
        tt1 = f_group_data.groupby(['f_group']).agg(
            {target: ['mean', 'count', 'sum'], 'f_max': 'max', 'f_min': 'min', 'miss': 'max'}).reset_index()
        s_data = pd.DataFrame()
        s_data['s_group'] = tt['s_group']

        s_data['s_Bad_rate'] = tt[target]['mean']
        s_data['s_N_obs'] = tt[target]['count']
        s_data['s_N_bad'] = tt[target]['sum']

        s_data['variable_name'] = '%s' % col
        f_data = pd.DataFrame()
        f_data['f_group'] = tt1['f_group']

        f_data['f_Bad_rate'] = tt1[target]['mean']
        f_data['f_N_obs'] = tt1[target]['count']
        f_data['f_N_bad'] = tt1[target]['sum']

        f_data['variable_name'] = '%s' % col

        total_bad = f_data['f_N_bad'].sum()
        total_good = f_data['f_N_obs'].sum() - f_data['f_N_bad'].sum()
        f_data['woe'] = f_data.apply(lambda x: math.log(
            (max(1, x['f_N_bad']) / total_bad) / (max(1, (x['f_N_obs'] - x['f_N_bad'])) / total_good)), axis=1)

        f_data['iv_g'] = ((f_data['f_N_bad'] / total_bad) - ((f_data['f_N_obs'] - f_data['f_N_bad']) / total_good)) * \
                         f_data['woe']
        iv = f_data['iv_g'].sum()
        f_data['iv'] = iv

        dd = inputdata[col].describe()
        ds = pd.DataFrame(dd).T.reset_index().rename({'index': 'variable_name'}, axis=1)
        ds['miss_count'] = inputdata[col].isnull().sum()

        group_info = pd.merge(group_info_base, f_data, how='left', on=['f_group'])
        group_info = pd.merge(group_info, s_data, how='left', on=['variable_name', 's_group'])
        group_info = pd.merge(group_info, ds, how='left', on=['variable_name'])

        outputdata = pd.merge(f_group_data[[col, 'f_group']], f_data[['f_group', 'woe']], how='left', on=['f_group'])
        outputdata['woe_%s' % col] = outputdata['woe']
        outputdata['f_group_%s' % col] = outputdata['f_group']

        return group_info, outputdata.drop(columns=['woe', 'f_group'])

    def charvar(self, inputdata, col, min_num, min_pct, target, criterion, splitter, max_depth, min_samples_leaf,
                max_leaf_nodes):
        miss_part = inputdata[inputdata[col].isnull()].copy()
        inputdata = inputdata[inputdata[col].isnull() == False].copy()
        s_group_map = self.char_finebin_group(inputdata=inputdata, col=col, target=target, min_num=min_num,
                                              min_pct=min_pct)
        s_group_data = self.char_finebinning(group_data=s_group_map, inputdata=inputdata, col=col)
        f_group_map_pre = self.roughbin_group(inputdata=s_group_data, s_group_col='s_group', col=col, target=target,
                                              criterion=criterion, splitter=splitter, max_depth=max_depth,
                                              min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                              specialcode_list=[])
        f_group_map = self.binning(group_data=f_group_map_pre, inputdata=s_group_map, col='s_group', inputmax='f_max',
                                   inputmin='f_min', inputgroup='f_group', specialcode_list=[])
        f_group_data = pd.merge(inputdata, f_group_map, how='left', on=col)
        #         f_group_data=self.binning(group_data=f_group_map_pre , inputdata=s_group_data , col='s_group' , inputmax='f_max' , inputmin='f_min' ,inputgroup='f_group',specialcode_list=[] )
        f_group_data['miss'] = False
        f_group_data['value'] = f_group_data[col]
        miss_part['f_group'] = -1
        miss_part['miss'] = True
        miss_part['value'] = 'miss'
        f_group_data = f_group_data.append(miss_part)
        tt1 = f_group_data.groupby(['f_group']).agg(
            {target: ['mean', 'count', 'sum'], 'miss': 'max'}).reset_index()
        tt = f_group_data.groupby(['f_group', 'value']).agg(
            {target: ['mean', 'count', 'sum'], 'miss': 'max'}).reset_index()
        s_data = pd.DataFrame()
        # s_data[col]=tt[col] 

        s_data['s_Bad_rate'] = tt[target]['mean']
        s_data['s_N_obs'] = tt[target]['count']
        s_data['s_N_bad'] = tt[target]['sum']
        s_data['miss_s'] = tt['miss']['max']
        s_data['f_group'] = tt['f_group']
        s_data['value'] = tt['value']
        s_data['variable_name'] = '%s' % col
        f_data = pd.DataFrame()
        f_data['f_group'] = tt1['f_group']

        f_data['f_Bad_rate'] = tt1[target]['mean']
        f_data['f_N_obs'] = tt1[target]['count']
        f_data['f_N_bad'] = tt1[target]['sum']
        f_data['miss_f'] = tt1['miss']['max']
        f_data['variable_name'] = '%s' % col
        total_bad = f_data['f_N_bad'].sum()
        total_good = f_data['f_N_obs'].sum() - f_data['f_N_bad'].sum()
        f_data['woe'] = f_data.apply(lambda x: math.log(
            (max(1, x['f_N_bad']) / total_bad) / (max(1, (x['f_N_obs'] - x['f_N_bad'])) / total_good)), axis=1)
        f_data['iv_g'] = ((f_data['f_N_bad'] / total_bad) - ((f_data['f_N_obs'] - f_data['f_N_bad']) / total_good)) * \
                         f_data['woe']
        iv = f_data['iv_g'].sum()
        f_data['iv'] = iv
        dd = f_group_data[col].describe()
        ds = pd.DataFrame(dd).T.reset_index().rename({'index': 'variable_name'}, axis=1)
        ds['miss_count'] = f_group_data[col].isnull().sum()

        group_info = pd.merge(s_data, f_data, how='left', on=['variable_name', 'f_group'])
        group_info = pd.merge(group_info, ds, how='left', on=['variable_name'])
        #      group_info['f_group']=group_info['f_group'].astype('int8')
        #       group_info['woe']=group_info['woe'].astype('float16')
        outputdata = pd.merge(f_group_data[[col, 'f_group']], f_data[['f_group', 'woe']], how='left', on=['f_group'])
        #         outputdata['f_group']=outputdata['f_group'].astype('int8')
        #        outputdata['woe']=outputdata['woe'].astype('float16')
        outputdata['woe_%s' % col] = outputdata['woe']
        outputdata['f_group_%s' % col] = outputdata['f_group']
        return group_info, outputdata.drop(columns=['woe', 'f_group'])

    def numericexist(self, inputdata, col, group_info_old, target, modify=False, add_value=0, data_only=False):

        woe_data = group_info_old[['f_group', 'woe']]
        specialcode_list = list(group_info_old[group_info_old['miss_s'] == True]['value'])
        group_info_num = group_info_old[group_info_old['miss_s'] == False]
        s_group_map = group_info_num[['s_group', 's_max', 's_min']].drop_duplicates()
        if modify == True:
            old_list = set(list(s_group_map['s_max']) + list(s_group_map['s_min']))
            old_list.add(add_value)
            new_list = list(old_list)
            new_list.sort()
            new_s_group = []
            for i in range(len(new_list) - 1):
                temp = {'s_group': i, 's_min': new_list[i], 's_max': new_list[i + 1]}
                new_s_group.append(temp)
            s_group_map = pd.DataFrame(new_s_group)
        f_group_map = group_info_num.groupby('f_group').agg({'s_max': 'max', 's_min': 'min'}).reset_index().rename(
            {'s_max': 'f_max', 's_min': 'f_min'}, axis=1)

        group_info_base = pd.DataFrame()
        for lins in range(len(f_group_map)):
            temp = s_group_map.copy()
            temp['f_group'] = f_group_map.loc[lins, 'f_group']
            temp['f_min'] = f_group_map.loc[lins, 'f_min']
            temp['f_max'] = f_group_map.loc[lins, 'f_max']
            temp_data = temp[
                ((temp['s_min'] >= temp['f_min']) & (temp['s_max'] <= temp['f_max']))
            ]
            group_info_base = pd.concat([group_info_base, temp_data])
        group_info_base['miss_s'] = False
        group_info_base['miss_f'] = False
        group_info_base['value'] = np.nan

        # check unique and consistent
        # if f_group_map['f_group'].max() - f_group_map['f_group'].min() != len(
        #         list(f_group_map['f_group'].unique())) - 1:
        #     print('final_group num error')
        # for i in range(f_group_map['f_group'].min(), f_group_map['f_group'].max() + 1):
        #     if i == f_group_map['f_group'].min():
        #         if f_group_map[f_group_map['f_group'] == i]['f_min'].iloc[0] != -np.inf:
        #             print('f_group errora f_group= %s' % i)
        #     if (i > f_group_map['f_group'].min()) & (i < f_group_map['f_group'].max()):
        #         if (f_group_map[f_group_map['f_group'] == i]['f_min'].iloc[0] !=
        #             f_group_map[f_group_map['f_group'] == i - 1]['f_max'].iloc[0]) | (
        #                 f_group_map[f_group_map['f_group'] == i]['f_max'].iloc[0] !=
        #                 f_group_map[f_group_map['f_group'] == i + 1]['f_min'].iloc[0]):
        #             print('f_group errorb f_group= %s' % i)
        #     if i == f_group_map['f_group'].max():
        #         if f_group_map[f_group_map['f_group'] == i]['f_max'].iloc[0] != np.inf:
        #             print('f_group errorc f_group= %s' % i)
        s_group_data = self.binning(group_data=s_group_map, inputdata=inputdata, col=col, inputmax='s_max',
                                    inputmin='s_min', inputgroup='s_group', specialcode_list=specialcode_list)
        if len(s_group_data)>0:
            f_group_data = self.binning(group_data=f_group_map, inputdata=s_group_data, col=col, inputmax='f_max',
                                        inputmin='f_min', inputgroup='f_group', specialcode_list=specialcode_list)
        else:
            f_group_data=pd.DataFrame()
        f_group_data['miss'] = False
        f_group_data['value'] = np.nan
        if specialcode_list != []:
            for values in specialcode_list:
                if values == 'miss':
                    temp = inputdata[inputdata[col].isnull()].copy()
                else:
                    temp = inputdata[inputdata[col] == values].copy()
                s_group = group_info_old[group_info_old['value'] == values]['s_group'].unique()[0]
                temp['s_group'] = s_group
                f_group = group_info_old[group_info_old['value'] == values]['f_group'].unique()[0]
                temp['f_group'] = f_group
                temp['miss'] = True
                temp['value'] = values
                f_group_data = f_group_data.append(temp)
                temps = pd.DataFrame(
                    {'s_group': s_group, 'f_group': f_group, 'value': values, 'miss_s': True, 'miss_f': True},
                    index=[0])
                group_info_base = group_info_base.append(temps)
        if (len(group_info_old[group_info_old['value'] == 'miss']) == 0) & (len(inputdata[inputdata[col].isnull()]) > 0):
            temp = inputdata[inputdata[col].isnull()].copy()
            temp['s_group'] = -99
            temp['f_group'] = group_info_old[group_info_old['woe'] == group_info_old['woe'].max()]['f_group'].values[0]
            temp['miss'] = True
            temp['value'] = 'miss'
            f_group_data = f_group_data.append(temp)
            tempg = pd.DataFrame({'s_group': -99, 'f_group': group_info_old[group_info_old['woe'] == group_info_old['woe'].max()]['f_group'].values[0], 'value': 'miss', 'miss_s': True, 'miss_f': True},index=[0])
            group_info_base = group_info_base.append(tempg)

        if data_only == True:
            outputdata = pd.merge(f_group_data, woe_data, how='left', on='f_group')

            outputdata['woe_%s' % col] = outputdata['woe']
            outputdata['f_group_%s' % col] = outputdata['f_group']
            outputdata = outputdata[[col, 'woe_%s' % col, 'f_group_%s' % col]]
            return outputdata
        tt = f_group_data.groupby(['s_group', 'f_group']).agg(
            {target: ['mean', 'count', 'sum'], 's_max': 'max', 's_min': 'min'}).reset_index()
        tt1 = f_group_data.groupby(['f_group']).agg(
            {target: ['mean', 'count', 'sum'], 'f_max': 'max', 'f_min': 'min'}).reset_index()
        s_data = pd.DataFrame()
        s_data['s_group'] = tt['s_group']
        s_data['s_Bad_rate'] = tt[target]['mean']
        s_data['s_N_obs'] = tt[target]['count']
        s_data['s_N_bad'] = tt[target]['sum']

        s_data['variable_name'] = '%s' % col

        f_data = pd.DataFrame()
        f_data['f_group'] = tt1['f_group']
        f_data['f_Bad_rate'] = tt1[target]['mean']
        f_data['f_N_obs'] = tt1[target]['count']
        f_data['f_N_bad'] = tt1[target]['sum']
        f_data['variable_name'] = '%s' % col

        total_bad = f_data['f_N_bad'].sum()
        total_good = f_data['f_N_obs'].sum() - f_data['f_N_bad'].sum()
        f_data['woe'] = f_data.apply(lambda x: math.log(
            (max(1, x['f_N_bad']) / total_bad) / (max(1, (x['f_N_obs'] - x['f_N_bad'])) / total_good)), axis=1)
        f_data['woe'] = round(f_data['woe'], 8)
        f_data['iv_g'] = ((f_data['f_N_bad'] / total_bad) - ((f_data['f_N_obs'] - f_data['f_N_bad']) / total_good)) * \
                         f_data['woe']
        iv = f_data['iv_g'].sum()
        f_data['iv'] = iv

        dd = inputdata[col].describe()
        ds = pd.DataFrame(dd).T.reset_index().rename({'index': 'variable_name'}, axis=1)
        ds['miss_count'] = inputdata[col].isnull().sum()

        group_info = pd.merge(group_info_base, f_data, how='left', on=['f_group'])
        group_info = pd.merge(group_info, s_data, how='left', on=['variable_name', 's_group'])
        group_info = pd.merge(group_info, ds, how='left', on=['variable_name'])

        outputdata = pd.merge(f_group_data[[col, 'f_group']], f_data[['f_group', 'woe']], how='left', on=['f_group'])
        outputdata['woe_%s' % col] = outputdata['woe']
        outputdata['f_group_%s' % col] = outputdata['f_group']

        return group_info, outputdata.drop(columns=['woe', 'f_group'])

    def charvarexist(self, inputdata, col, target, group_info_old, data_only=False):
        woe_data = group_info_old[['f_group', 'woe']]
        specialcode_list = list(group_info_old[group_info_old['miss_s'] == True]['value'])
        inputdata_nomiss = inputdata[inputdata[col].isnull() == False]
        group_info_num = group_info_old[group_info_old['miss_s'] == False]
        f_group_map = group_info_num[['f_group', 'value']].drop_duplicates()
        f_group_map = f_group_map.rename({'value': col}, axis=1)
        f_group_data = pd.merge(inputdata_nomiss, f_group_map, how='left', on=col)
        f_group_data['value'] = f_group_data[col]
        f_group_data['miss'] = False

        if specialcode_list != []:
            for values in specialcode_list:
                if values == 'miss':
                    temp = inputdata[inputdata[col].isnull()].copy()
                else:
                    temp = inputdata[inputdata[col] == values].copy()
                temp['f_group'] = group_info_old[group_info_old['value'] == values]['f_group'].unique()[0]
                temp['miss'] = True
                temp['value'] = values
                f_group_data = f_group_data.append(temp)
        #原本没有空值出现了新的空值
        if (len(group_info_old[group_info_old['value'] == 'miss']) == 0) & (len(inputdata[inputdata[col].isnull()]) > 0):
            temp = inputdata[inputdata[col].isnull()].copy()
            temp['f_group'] = group_info_old[group_info_old['woe'] == group_info_old['woe'].max()]['f_group'].values[0]
            # temp['f_group'] = -99
            temp['miss'] = True
            temp['value'] = 'miss'
            f_group_data = f_group_data.append(temp)

        # 原本没有出现的值了新的值
        if len(f_group_data[f_group_data['f_group'].isnull()]) > 0:
            temp = f_group_data[f_group_data['f_group'].isnull()].copy()
            temp['f_group'] = group_info_old[group_info_old['woe'] == group_info_old['woe'].max()]['f_group'].values[0]
            # temp['f_group'] = -100
            temp['miss'] = False
            temp['value'] = temp[col]
            f_group_data = f_group_data[f_group_data['f_group'].isnull() == False].copy()
            f_group_data = f_group_data.append(temp)

        if data_only == True:
            outputdata = pd.merge(f_group_data, woe_data, how='left', on='f_group')
            outputdata['woe_%s' % col] = outputdata['woe']
            outputdata['f_group_%s' % col] = outputdata['f_group']
            outputdata = outputdata[[col, 'woe_%s' % col, 'f_group_%s' % col]]
            return outputdata
        #         f_group_data[col]=f_group_data[col].fillna('miss')
        tt1 = f_group_data.groupby(['f_group']).agg({target: ['mean', 'count', 'sum'], 'miss': 'max'}).reset_index()
        tt = f_group_data.groupby(['f_group', 'value']).agg({target: ['mean', 'count', 'sum'], 'miss': 'max'}).reset_index()
        s_data = pd.DataFrame()


        s_data['s_Bad_rate'] = tt[target]['mean']
        s_data['s_N_obs'] = tt[target]['count']
        s_data['s_N_bad'] = tt[target]['sum']
        s_data['miss_s'] = tt['miss']['max']
        s_data['f_group'] = tt['f_group']
        s_data['value'] = tt['value']
        s_data['variable_name'] = '%s' % col
        f_data = pd.DataFrame()
        f_data['f_group'] = tt1['f_group']

        f_data['f_Bad_rate'] = tt1[target]['mean']
        f_data['f_N_obs'] = tt1[target]['count']
        f_data['f_N_bad'] = tt1[target]['sum']
        f_data['miss_f'] = tt1['miss']['max']
        f_data['variable_name'] = '%s' % col
        total_bad = f_data['f_N_bad'].sum()
        total_good = f_data['f_N_obs'].sum() - f_data['f_N_bad'].sum()
        f_data['woe'] = f_data.apply(lambda x: math.log(
            (max(1, x['f_N_bad']) / total_bad) / (max(1, (x['f_N_obs'] - x['f_N_bad'])) / total_good)), axis=1)
        f_data['iv_g'] = ((f_data['f_N_bad'] / total_bad) - ((f_data['f_N_obs'] - f_data['f_N_bad']) / total_good)) * \
                         f_data['woe']
        iv = f_data['iv_g'].sum()
        f_data['iv'] = iv
        dd = inputdata[col].describe()
        ds = pd.DataFrame(dd).T.reset_index().rename({'index': 'variable_name'}, axis=1)
        ds['miss_count'] = inputdata[col].isnull().sum()

        group_info = pd.merge(s_data, f_data, how='left', on=['variable_name', 'f_group'])
        group_info = pd.merge(group_info, ds, how='left', on=['variable_name'])
        #      group_info['f_group']=group_info['f_group'].astype('int8')
        #     group_info['woe']=group_info['woe'].astype('float16')
        outputdata = pd.merge(f_group_data[[col, 'f_group']], f_data[['f_group', 'woe']], how='left', on=['f_group'])
        #       outputdata['f_group']=outputdata['f_group'].astype('int8')
        #        outputdata['woe']=outputdata['woe'].astype('float16')
        outputdata['woe_%s' % col] = outputdata['woe']
        outputdata['f_group_%s' % col] = outputdata['f_group']
        return group_info, outputdata.drop(columns=['woe', 'f_group'])