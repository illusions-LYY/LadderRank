# -*- coding:utf-8 -*-


import pandas as pd
from result import person_rank

# epr = pd.read_csv('/Users/shen-pc/Desktop/WORK/Ladderrank/final_df/r2/eliminate_primary_rank2.csv')
data = pd.read_csv('final_df/r5(20180816-20190831)/origin_data_for_demo.csv')

data['sch_sk'] = data['sch_name'] + data['province'] + data['city'] + data['area']
res1 = person_rank(data, '清华大学附属中学朝阳学校北京市北京市朝阳区')
res1.to_csv('final_df/r5(20180816-20190831)/person_rank_清华大学附属中学朝阳学校.csv')