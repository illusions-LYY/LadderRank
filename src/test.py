# -*- coding:utf-8 -*-

import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 这两个参数的默认设置都是False
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

data = pd.read_csv('final_df/data.csv')

F1_score = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

final_res = pd.DataFrame(dict())
final_res['做题量'] = data.groupby('sch_name').apply(lambda x: len(x))
prov = dict()
for i in data['sch_name'].unique():
    prov[i] = data[data['sch_name'] == i]['province'].tolist()[0]
final_res['省份'] = [prov[i] for i in final_res.index]
uid_count = [len(x) for x in data.groupby('sch_name').uid.unique()]
final_res['用户总数'] = uid_count
final_res['人均题量'] = final_res['做题量'].values / uid_count

sch_user_ts = data[['uid', 'sch_name']]
sch_user_ts = sch_user_ts.drop_duplicates()
# sch_user_ts['trueskill'] = sch_user_ts.uid.apply(lambda x: user_ts[x].mu)
# final_res['人均TrueSkill'] = sch_user_ts.groupby('sch_name').apply(lambda x: x['trueskill'].mean())
final_res['知识点数'] = data.groupby('sch_name').apply(lambda x: len(x.gid.unique()))

rs = data.groupby(['gid', 'sch_name']).apply(lambda x: sum(x.ts_pts))
rs = pd.DataFrame(rs).reset_index()
rs.columns = ['gid', 'sch_name', 'pt']
rs['rank_int'] = rs[['gid', 'pt']].groupby(['gid']).rank(ascending=0)
rs['F1_pt'] = rs['rank_int'].apply(lambda x: F1_score.get(x, 0))

final_res['总分F1'] = rs.groupby('sch_name').apply(lambda x: sum(x['F1_pt']))
final_res['平均顺位'] = rs.groupby('sch_name').apply(lambda x: x['rank_int'].mean())
final_res['得到第一的次数'] = rs.groupby('sch_name').apply(lambda x: sum(x['rank_int'] == 1))
final_res['得到前二的次数'] = rs.groupby('sch_name').apply(lambda x: sum(x['rank_int'] < 2.1))
final_res['得到前三的次数'] = rs.groupby('sch_name').apply(lambda x: sum(x['rank_int'] < 3.1))
final_res['6名开外次数'] = rs.groupby('sch_name').apply(lambda x: sum(x['rank_int'] > 6.9))
final_res = final_res.sort_values(by='总分F1', ascending=False)

local_bj = final_res[final_res['省份'] == '北京市']['总分F1'].rank(axis=0, method='min', ascending=False).astype(int).astype(
    str)
local_tj = final_res[final_res['省份'] == '天津市']['总分F1'].rank(axis=0, method='min', ascending=False).astype(int).astype(
    str)
local_hb = final_res[final_res['省份'] == '河北省']['总分F1'].rank(axis=0, method='min', ascending=False).astype(int).astype(
    str)
for i in range(len(local_bj)):
    local_bj[i] = '京' + local_bj[i]
for i in range(len(local_tj)):
    local_tj[i] = '津' + local_tj[i]
for i in range(len(local_hb)):
    local_hb[i] = '冀' + local_hb[i]
local = pd.concat([local_bj, local_tj, local_hb], axis=0)
final_res['本地区名次'] = [local[sch] for sch in final_res.index]

print(final_res)
