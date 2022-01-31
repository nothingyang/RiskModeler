import pandas as pd
import numpy as np
#from base import group_func
from funcc import binning

data=pd.read_csv('D:\\SynologyDrive\\temp\\application_train.csv')
target='TARGET'
data['date']=np.random.choice(['2019-09','2019-10','2019-11','2019-12','2020-01','2020-02','2020-03','2020-04'], size=len(data), replace=True)
specialcode_list=[]

colnum=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
#data.select_dtypes(include=['float','int8','int16','int32','int64']).columns.values.tolist()[3:]
colchar=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR']
#data.select_dtypes(include=['object']).columns.values.tolist()

min_size=10
min_lift=1.4
min_bad_rate=0.2
s_group_map,s_group_data=binning.fit_bin_aprior(binning,data=data,varnum=colnum,target='TARGET', s_bin_num=40)
print(s_group_map)
s_group_map['iv']=0
s_group_map['woe']=0
s_group_map['f_Bad_rate']=0
s_group_map['f_N_obs']=0
s_group_map['f_N_bad']=0
s_group_map['f_group']=0

s_group_report,f_group_report = binning.report(binning,group_info=s_group_map,
                                                          varchar=colchar,
                                                          varnum=colnum)
merge_data=s_group_report[['variable_name','s_group','value','label']]
merge_data.loc[merge_data['value'].isnull()==False,'label']=merge_data.loc[merge_data['value'].isnull()==False,'variable_name']+'_'+merge_data.loc[merge_data['value'].isnull()==False,'value']
merge_data['variable_name']='s_group_'+merge_data['variable_name']
aprior_data=s_group_data[colchar+list(merge_data['variable_name'].unique())+[target]]

for var in merge_data['variable_name'].unique():
    aprior_data=pd.merge(merge_data[merge_data['variable_name']==var][['s_group','label']],aprior_data,left_on='s_group',right_on=var,how='right')
    aprior_data['ap_%s'%var]=aprior_data['label']
    aprior_data=aprior_data.drop(columns=[var,'label','s_group'])
for var in colchar:
    aprior_data['ap_%s'%var]=var+aprior_data[var]
    aprior_data.loc[aprior_data['ap_%s' % var].isnull(),'ap_%s' % var]='miss'
    aprior_data=aprior_data.drop(columns=[var])
aprior_data[target]=s_group_data[target]


from itertools import combinations
print (list(combinations(aprior_data.columns, 2)))

pair_list=list(aprior_data.columns)
pair_list.remove('TARGET')

pair_list=list(combinations(pair_list, 1))
#+list(combinations(pair_list, 2))+list(combinations(pair_list, 3))
print(pair_list)

re_df=pd.DataFrame()
for i in range(len(pair_list)):
    pair=pair_list[i]
    tt=aprior_data[list(pair)+[target]]
    re=tt.groupby(list(pair))[target].agg(['sum','mean','count']).reset_index()
    re.head()
    re['rule']=re.apply(lambda x: [x[t]  for t in list(pair)],axis=1)
    re['variable']=re.apply(lambda x: list(pair) , axis=1)
    re=re[['sum','mean','count','rule','variable']]
    re_df=re_df.append(re)



re_df['lift']=re_df['mean']/np.mean(aprior_data[target])
re_df=re_df.sort_values(by='lift',ascending=False)
final_rule=re_df[(re_df['count']>min_size)&(re_df['mean']>min_bad_rate)&(re_df['lift']>min_lift)]
final_rule=final_rule.reset_index(drop=True)
final_rule=final_rule.reset_index()
for i in range(len(final_rule)):
    rule_t=final_rule.iloc[i]
    merge_data=pd.DataFrame(data=np.array([rule_t['rule']+[1]]),columns=np.array(rule_t['variable']+['flag_rule_'+str(rule_t['index'])]))
    aprior_data=pd.merge(aprior_data,merge_data,how='left')
    aprior_data.loc[aprior_data['flag_rule_'+str(rule_t['index'])].isnull()==True,'flag_rule_'+str(rule_t['index'])]=0

final_rule['rule_name']=final_rule['index'].apply(lambda x: 'flag_rule_'+str(x) )

plot_df=aprior_data.copy()
plot_df['rm__order__']=plot_df.index



tt=aprior_data.groupby(['flag_rule_'+str(x) for x in range(len(final_rule)) ])[target].agg({'count','mean'}).reset_index()

tt['target']=tt.apply(lambda x: "-".join(set([int(x[t])*t for t in ['flag_rule_'+str(m) for m in range(len(final_rule))]])),axis=1)

tt['label']=tt.apply(lambda x: "&".join(set([int(x['flag_rule_%s' %m])*str(final_rule['rule'][m]) for m in range(len(final_rule))])),axis=1)


tt['start']=tt.apply(lambda x: set([int(x[t])*t for t in ['flag_rule_'+str(m) for m in range(len(final_rule))]]),axis=1)


links_list =[]

nodes_list =[]
for i in range(len(tt)):
    for s in tt.iloc[i]['start']:
        if s !='':
            links_list.append({'source':s,'target':tt.iloc[i]['target'],'value':tt.iloc[i]['count'],'badrate':tt.iloc[i]['mean']})
            nodes_list.append(s)
            nodes_list.append(tt.iloc[i]['target'])
            
            #nodes_list.append({'name':s})
            #nodes_list.append({'name':tt.iloc[i]['target']})
nodes_list=list(set(nodes_list))
nodes_list=[{'name':x } for x in nodes_list]           



import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
dd=pd.DataFrame(links_list)
labels=list(set(list(dd['source'])+list(dd['target'])))
dd['num']=dd['value']*dd['badrate']
fi=dd.groupby('source')['num','value'].sum().reset_index()
fi['rate']=fi['num']/fi['value']

badrate_df=pd.concat([fi[['source','rate']].rename({'source':'label','rate':'rate'},axis=1),
dd[['target','badrate']].rename({'target':'label','badrate':'rate'},axis=1)])

comment_df=pd.concat([final_rule[['rule_name','rule']].rename({'rule_name':'label','rule':'comment'},axis=1),
tt[['target','label']].rename({'target':'label','label':'comment'},axis=1)])



bad_rate_list=[list(badrate_df.loc[badrate_df['label']==m,'rate'])[0] for m in labels]

commend=[list(comment_df.loc[comment_df['label']==m,'comment'])[0] for m in labels]



col_code=[rgb_to_hex((256-int(x*255), 256-int(x*255), 256-int(x*255))) for x in bad_rate_list]
mergedd=pd.DataFrame(labels)
mergedd['order']=mergedd.index
mergedd=mergedd.rename({0:'var'},axis=1)

sankey_df=pd.merge(dd,mergedd,left_on='source',right_on='var',how='left')
sankey_df=sankey_df.rename({'order':'source_order'},axis=1)

sankey_df=pd.merge(sankey_df,mergedd,left_on='target',right_on='var',how='left')
sankey_df=sankey_df.rename({'order':'target_order'},axis=1)

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = [commend[m] for m in range(len(commend))],
      color = col_code, 
      customdata =[[commend[m],round(bad_rate_list[m],3)] for m in range(len(commend))],
      hovertemplate='Node: %{label} <br /> Rules:%{customdata[0]} <br /> badrate: %{customdata[1]}<extra></extra>',
    ),
    link = dict(
      source = list(sankey_df['source_order']), # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = list(sankey_df['target_order']),
      value=list(sankey_df['value']),
      color=[rgb_to_hex((256-int(x*255), 256-int(x*255), 256-int(x*255))) for x in list(sankey_df['badrate']) ]
  ))])
fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.show()



