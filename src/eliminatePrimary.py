# -*- coding:utf-8 -*-

import pandas as pd

sch_rank = pd.read_csv('final_df/r6(20180916~20180930)all/school_rank_DROP6.csv')

for i in sch_rank.iterrows():
    if '小学' in i[1]['校名']:
        sch_rank.drop(i[0], axis=0, inplace=True)
sch_rank.reset_index(drop=True, inplace=True)
sch_rank.to_csv('final_df/r6(20180916~20180930)all/eliminate_primary_rank2.csv')
