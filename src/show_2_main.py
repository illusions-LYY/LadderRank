# -*- coding:utf-8 -*-

import pandas as pd
import utils
import trueskill as ts
import time

s = time.time()
param_f = open('final_df/r3/params.txt')
params = list(map(float, param_f.readline().split(',')))
big_zero = float(param_f.readline())
print(params, big_zero)

data = pd.read_csv('final_df/r2/data_round2.csv')
g_ts = data[['gid', 'temp_g']][::-1]
u_ts = data[['uid', 'temp_u']][::-1]

demo_data = pd.read_csv('final_df/r5(20180816-20190831)/demo_data.csv')
demo_data['g_player'] = ts.Rating().mu
demo_data['u_player'] = ts.Rating().mu

for i in range(len(demo_data['gid'])):
    prob_name = demo_data['gid'][i]
    if prob_name in g_ts['gid'].tolist():
        demo_data['g_player'][i] = data[data['gid'] == prob_name]['temp_g'].tolist()[-1]

for i in range(len(demo_data['uid'])):
    user_name = demo_data['uid'][i]
    if user_name in u_ts['uid'].tolist():
        demo_data['u_player'][i] = data[data['uid'] == user_name]['temp_u'].tolist()[-1]
demo_data.to_csv('final_df/r5(20180816-20190831)/demo_data_after_proc.csv')

# 完成全部的权重、时间处理，现在开始并行迭代计算：
print(demo_data.shape)
demo_data, user_ts, goal_ts = utils.parallel_demo(demo_data, params, big_zero)

# 将弥补项考虑进来，以此更新ts_pts值：
print('ts is updating...')
demo_data = utils.parallel_ts(demo_data, user_ts, goal_ts)

print('DATA is saving...')
demo_data.to_csv('final_df/r5(20180816-20190831)/origin_data_for_demo.csv', index=False)
print('==========================================')
e = time.time()
print('总共用时:% .2f分钟' % round((e - s) / 60, 2))
