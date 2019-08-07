# -*- coding:utf-8 -*-


import pandas as pd

# 每个知识点下每个学校得分排名：
F1_score = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def school_rank(data, user_ts, drop=False):
    data.index = data.index.astype(str)
    final_res = pd.DataFrame(dict())
    data['sch_sk'] = data['sch_name'] + data['province'] + data['city'] + data['area']
    final_res['校名'] = data.groupby('sch_sk').apply(lambda x: x['sch_name'].tolist()[0])
    final_res['做题量'] = data.groupby('sch_sk').apply(lambda x: len(x))

    temp = data.groupby('sch_sk').apply(lambda x: x[['province', 'city', 'area']].values[0])
    final_res[['省份', '城市', '区划']] = temp.apply(pd.Series)

    uid_count = [len(x) for x in data.groupby('sch_sk').uid.unique()]
    final_res['用户总数'] = uid_count
    final_res['人均题量'] = final_res['做题量'].values / uid_count

    sch_user_ts = data[['uid', 'sch_sk']]
    sch_user_ts = sch_user_ts.drop_duplicates()
    sch_user_ts['trueskill'] = sch_user_ts.uid.apply(lambda x: user_ts[x].mu)
    final_res['人均TrueSkill'] = sch_user_ts.groupby('sch_sk').apply(lambda x: x['trueskill'].mean())
    final_res['知识点数'] = data.groupby('sch_sk').apply(lambda x: len(x.gid.unique()))

    rs = data.groupby(['gid', 'sch_sk']).apply(lambda x: sum(x.ts_pts))
    rs = pd.DataFrame(rs).reset_index()
    rs.columns = ['gid', 'sch_sk', 'pt']
    rs['rank_int'] = rs[['gid', 'pt']].groupby(['gid']).rank(ascending=0)
    # rs['rank_pct'] = rs[['gid', 'pt']].groupby('gid').rank(ascending=True, method='max', pct=True)
    # rs['Func_pt'] = rs['rank_pct'].apply(lambda x: pow((x ** 8), 1 / 3) * 25)
    rs['F1_pt'] = rs['rank_int'].apply(lambda x: F1_score.get(x, 0))

    # final_res['总分Func'] = rs.groupby('sch_sk').apply(lambda x: sum(x['Func_pt']))
    final_res['总分F1'] = rs.groupby('sch_sk').apply(lambda x: sum(x['F1_pt']))
    final_res['平均顺位'] = rs.groupby('sch_sk').apply(lambda x: x['rank_int'].mean())
    final_res['得到第一的次数'] = rs.groupby('sch_sk').apply(lambda x: sum(x['rank_int'] == 1))
    final_res['得到前二的次数'] = rs.groupby('sch_sk').apply(lambda x: sum(x['rank_int'] < 2.1))
    final_res['得到前三的次数'] = rs.groupby('sch_sk').apply(lambda x: sum(x['rank_int'] < 3.1))
    final_res['6名开外次数'] = rs.groupby('sch_sk').apply(lambda x: sum(x['rank_int'] > 6.9))
    final_res = final_res.sort_values(by='总分F1', ascending=False)

    final_res.index = final_res.index.astype(str)
    local_bj = final_res[final_res['省份'] == '北京市']['总分F1'].rank(axis=0, method='min', ascending=False).astype(
        int).astype(str)
    local_tj = final_res[final_res['省份'] == '天津市']['总分F1'].rank(axis=0, method='min', ascending=False).astype(
        int).astype(str)
    local_hb = final_res[final_res['省份'] == '河北省']['总分F1'].rank(axis=0, method='min', ascending=False).astype(
        int).astype(str)
    for i in range(len(local_bj)):
        local_bj[i] = '京' + local_bj[i]
    for i in range(len(local_tj)):
        local_tj[i] = '津' + local_tj[i]
    for i in range(len(local_hb)):
        local_hb[i] = '冀' + local_hb[i]
    local = pd.concat([local_bj, local_tj, local_hb], axis=0)
    final_res['本地区名次'] = [local[sch] for sch in final_res.index]

    if drop:
        final_res = final_res[final_res['用户总数'] >= 10]
    return final_res


def person_rank(data, sch_sk):
    wu = data[data['sch_sk'] == sch_sk]
    ts_wu = pd.DataFrame(columns=['scores', 'rank'])
    ts_wu['scores'] = wu.groupby('uid').apply(lambda x: sum(x['ts_pts']))
    ts_wu['rank'] = ts_wu.rank(method='min', ascending=True, pct=True)
    ts_wu = ts_wu.sort_values(by='scores', ascending=False)
    return ts_wu
