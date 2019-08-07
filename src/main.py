# -*- coding:utf-8 -*-
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import result

if __name__ == '__main__':
    start = time.time()

    data = pd.read_csv('final_df/r6(20180916~20180930)all/origin_DATA.csv')
    # 首先是数据处理流程
    pick_dur = data[['gid', 'dur', 'corr']]
    # 删除duration为负的记录：
    data.drop(data[data.dur < 0].index, axis=0, inplace=True)
    dur_17500 = pick_dur[pick_dur['dur'] < 17500]
    dur_17500_rate = dur_17500.groupby(pd.qcut(dur_17500['dur'], q=min(300, int(len(dur_17500) / 500)))) \
        .apply(lambda x: sum(x['corr'] == 0) / len(x))
    mid_17500 = [i.mid for i in dur_17500_rate.index]
    mid_17500 = np.array(mid_17500)

    # 寻找最佳的拟合数据的范围：
    stop = 10000
    for i in range(3000, 10000, 1000):
        back = np.mean(dur_17500_rate[mid_17500 > i][:20])
        front = np.mean(dur_17500_rate[mid_17500 < i][-20:])
        if front > back:
            stop = i + 500
            break

    # 用二次函数拟合数据，得到拟合的二次函数参数和零点（区分点）：
    params, big_zero = utils.fitting(mid_17500, dur_17500_rate, stop)

    # 我们画个图来看看拟合的是否有问题：
    # 获取当前的坐标轴, gca = get current axis
    ax = plt.gca()
    # 设置右边框和上边框
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # 设置x坐标轴为下边框
    ax.xaxis.set_ticks_position('bottom')
    # 设置y坐标轴为左边框
    ax.yaxis.set_ticks_position('left')
    # 设置x轴, y周在(0, 0)的位置
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.plot(mid_17500, dur_17500_rate, 'r.', label='raw data')
    x_draw = np.linspace(0, big_zero + 1000, 500)
    y_draw = utils.f_2(x_draw, params[0], params[1], params[2])

    print(params, 'stop=', stop)
    file = open('final_df/r6(20180916~20180930)all/params.txt', 'w')
    file.write(str(params[0]) + ',' + str(params[1]) + ',' + str(params[2]) + '\n' + str(big_zero))
    file.close()

    print('========================')
    plt.plot(x_draw, y_draw, 'g-', label='fitting curve')
    plt.legend(loc='best')
    plt.show()

    # 完成全部的权重、时间处理，现在开始并行迭代计算：
    print(data.shape)
    data = utils.parallel(data, params, big_zero)[0]
    user_ts = utils.parallel(data, params, big_zero)[1]
    goal_ts = utils.parallel(data, params, big_zero)[2]

    # 将弥补项考虑进来，以此更新ts_pts值：
    print('ts is updating...')
    data = utils.parallel_ts(data, user_ts, goal_ts)

    print('DATA is saving...')
    data.to_csv('final_df/r6(20180916~20180930)all/data_round6.csv', index=False)
    print('==========================================')

    print('school rank1 is saving...')
    frame1 = result.school_rank(data, user_ts)
    frame1.to_csv('final_df/r6(20180916~20180930)all/school_rank_round6.csv')
    print('==========================================')

    print('school rank drop is saving...')
    frame1_drop = result.school_rank(data, user_ts, True)
    frame1_drop.to_csv('final_df/r6(20180916~20180930)all/school_rank_DROP6.csv')

    print('person rank is saving...')
    frame2 = result.person_rank(data, '清华大学附属中学朝阳学校北京市北京市朝阳区')
    frame2.to_csv('final_df/r6(20180916~20180930)all/person_rank_round6.csv')
    print('==========================================')
    end = time.time()
    print('总共用时:% .2f分钟' % round((end - start) / 60, 2))
