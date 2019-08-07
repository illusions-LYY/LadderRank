# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy import optimize
from collections import defaultdict
import math
import trueskill as ts
import multiprocessing as mps


def fitting(mid, rate, stop):
    mid_stop = mid[mid < stop]
    dur_stop_rate = rate[:len(mid_stop)]

    a, b, c = optimize.curve_fit(f_2, mid_stop, dur_stop_rate)[0]
    big_zero = max((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a), (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a))
    return (a, b, c), big_zero


def f_2(x, A, B, C):
    return A * x ** 2 + B * x + C


# 将输入的duration按段转换成权值小数，区间[-0.5,0.5]：
def durToWeight(x, params, big_zero):
    a, b, c = params[0], params[1], params[2]
    if x < big_zero:
        summit = (4 * a * c - b ** 2) / (4 * a)
        return (f_2(x, a, b, c) / summit) * (-0.5) if f_2(x, a, b, c) > 0 else 0
    else:
        x = min(x, 600000)
        return (x / 600000) * 0.5


# 根据user和goal的能力值，计算出权重，区间[-0.5,0.5]：
def win_proba(user, goal):
    delta_mu = user.mu - goal.mu
    sum_sigma = user.sigma ** 2 + goal.sigma ** 2
    size = 2
    denom = math.sqrt(size * (ts.BETA * ts.BETA) + sum_sigma)
    tse = ts.global_env()
    return tse.cdf(delta_mu / denom)


# 权重处理总函数：
def norm_duration(item, idx, u_player, g_player, data, params, big_zero):
    d = item['dur']
    if item['corr'] == 1:
        d = max(2000, d)
        d = min(d, 600000)
        data.loc[idx, 'dur_weight'] = np.nan
        data.loc[idx, 'win_proba_weight'] = np.nan
        return (2000 / d) ** 0.3
    else:
        weight1 = 0.5 * (durToWeight(d, params, big_zero) - 0.5)
        weight2 = -0.5 * win_proba(u_player, g_player)
        data.loc[idx, 'dur_weight'] = weight1
        data.loc[idx, 'win_proba_weight'] = weight2
        return weight1 + weight2


# 核心迭代计算函数：
def cal_TrueSkill(args):
    data, params, big_zero = args[0], args[1], args[2]
    user = defaultdict(ts.Rating)
    goal = defaultdict(ts.Rating)

    for i in data.iterrows():
        idx = i[0]
        item = i[1]
        # 对每一条迭代进来的记录，分别初始化uid和pid的能力值，然后用上式计算。
        u_player = user[item['uid']]
        g_player = goal[item['gid']]

        # 核心句，计算一次做题提交之后，trueskill的变化。根据user做题的对错情况，决定这个delta数值是正还是负。
        data.loc[idx, 'ts_pts'] = (g_player.mu / u_player.mu) * norm_duration(item, idx, u_player, g_player, data,
                                                                              params, big_zero)
        # 记录即时的goal与user值，是为了后面计算compensate
        data.loc[idx, 'temp_g'] = g_player.mu
        data.loc[idx, 'temp_u'] = u_player.mu

        if item['corr']:
            u_player, g_player = ts.rate_1vs1(u_player, g_player)
        else:
            g_player, u_player = ts.rate_1vs1(g_player, u_player)

        user[item['uid']] = u_player
        goal[item['gid']] = g_player
    return user, goal, data


# user/goal两个defaultdict分别存储某个user或某个goal的实时trueskill；data的‘ts_pts’列，存储该条记录对战过后，user应该发生的trueskill变化

def cal_TrueSkill_demo(args):
    data, params, big_zero = args[0], args[1], args[2]
    user = defaultdict(ts.Rating)
    goal = defaultdict(ts.Rating)
    for i in data.iterrows():
        rows = i[1]
        user[rows['uid']] = ts.Rating(rows['u_player'], 3.0)
        goal[rows['gid']] = ts.Rating(rows['g_player'], 3.0)

    for i in data.iterrows():
        idx = i[0]
        item = i[1]
        # 对每一条迭代进来的记录，分别初始化uid和pid的能力值，然后用上式计算。
        u_player = user[item['uid']]
        g_player = goal[item['gid']]

        # 核心句，计算一次做题提交之后，trueskill的变化。根据user做题的对错情况，决定这个delta数值是正还是负。
        data.loc[idx, 'ts_pts'] = (g_player.mu / u_player.mu) * norm_duration(item, idx, u_player, g_player, data,
                                                                              params, big_zero)
        # 记录即时的goal与user值，是为了后面计算compensate
        data.loc[idx, 'temp_g'] = g_player.mu
        data.loc[idx, 'temp_u'] = u_player.mu

        if item['corr']:
            u_player, g_player = ts.rate_1vs1(u_player, g_player)
        else:
            g_player, u_player = ts.rate_1vs1(g_player, u_player)

        user[item['uid']] = u_player
        goal[item['gid']] = g_player
    return user, goal, data


def split_data(data):
    cores = mps.cpu_count()
    split_num = np.linspace(0, len(data), cores + 1, dtype=int)
    data_seg = [data[split_num[j]:split_num[j + 1]] for j in range(len(split_num) - 1)]
    return data_seg


def parallel(data, params, big_zero):
    cores = mps.cpu_count()
    pool = mps.Pool(processes=cores)

    r = []
    data_seg = split_data(data)
    for i in data_seg:
        arg_li = [i, params, big_zero]
        r.append(pool.apply_async(cal_TrueSkill, (arg_li,)))

    pool.close()
    pool.join()

    res = [i.get() for i in r]
    data = pd.concat([i[2] for i in res])

    user_ts = defaultdict(ts.Rating)
    goal_ts = defaultdict(ts.Rating)

    for i in res:
        user_ts.update(i[0])
        goal_ts.update(i[1])
    return data, user_ts, goal_ts


def parallel_demo(data, params, big_zero):
    cores = mps.cpu_count()
    pool = mps.Pool(processes=cores)

    r = []
    data_seg = split_data(data)
    for i in data_seg:
        arg_li = [i, params, big_zero]
        r.append(pool.apply_async(cal_TrueSkill_demo, (arg_li,)))

    pool.close()
    pool.join()

    res = [i.get() for i in r]
    data = pd.concat([i[2] for i in res])

    user_ts = defaultdict(ts.Rating)
    goal_ts = defaultdict(ts.Rating)

    for i in res:
        user_ts.update(i[0])
        goal_ts.update(i[1])
    return data, user_ts, goal_ts


# 计算权衡因子b：
def cal_b(user_ts, goal_ts):
    u_ts_max = max(list(map(lambda x: x.mu, list(user_ts.values()))))
    u_ts_min = min(list(map(lambda x: x.mu, list(user_ts.values()))))
    g_ts_max = max(list(map(lambda x: x.mu, list(goal_ts.values()))))
    g_ts_min = min(list(map(lambda x: x.mu, list(goal_ts.values()))))
    b1 = g_ts_min * (2000 / 600000) ** 0.3 / (u_ts_max - g_ts_min)
    b2 = g_ts_max / (g_ts_max - u_ts_min)
    return min(b1, b2)


# 更新ts取值：
def ts_update(data, b):
    data['compensate'] = data.apply(lambda row: round((row['temp_g'] / row['temp_u'] - 1) * b, 4), axis=1)
    data['ts_pts'] = data['ts_pts'] + data['compensate']
    return data


# 并行计算更新ts：
def parallel_ts(data, user_ts, goal_ts):
    b = cal_b(user_ts, goal_ts)
    cores = mps.cpu_count()
    pool2 = mps.Pool(processes=cores)

    r2 = []
    data_seg2 = split_data(data)
    for i in data_seg2:
        r2.append(pool2.apply_async(ts_update, args=(i, b,)))

    pool2.close()
    pool2.join()

    res2 = [i.get() for i in r2]
    data = pd.concat([i for i in res2])
    return data
