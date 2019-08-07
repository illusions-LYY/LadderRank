# -*- coding:utf-8 -*-

# 9月京津冀地区全部高中的数据

from impala.dbapi import connect
import pandas as pd

conn = connect(host='10.8.8.21', port=10015, auth_mechanism='PLAIN', database='tmp', user='shenfei', password='iPqKSV')
cursor = conn.cursor()
cursor.execute('''
SELECT t1.*, t2.goal_id, t3.*
FROM
(
    SELECT id, u_user, problem_id,duration, day, correct, from_unixtime(event_time/1000) AS time
    FROM events.frontend_event_orc
    WHERE day BETWEEN 20180816 AND 20180831
      AND event_key = 'clickLTTPSumbit'
      AND u_user != ''
      AND u_user IS NOT NULL
)t1
inner join
(
    SELECT u_user,school_sk
    FROM dw.dim_user
)t0 on t1.u_user = t0.u_user
INNER JOIN
(
    SELECT id, goal_id
    FROM course.problem
)t2 ON t1.problem_id = t2.id
INNER JOIN
(
    SELECT school_sk, name, province, city, area
    FROM dw.dim_school
    WHERE province IS NOT NULL
    AND name = '清华大学附属中学朝阳学校'
    AND province = '北京市'
)t3 ON t0.school_sk = t3.school_sk
ORDER BY t1.day,t1.time ASC''')

data = cursor.fetchall()
data = pd.DataFrame(data)
data.columns = ['id', 'uid', 'pid', 'dur', 'day', 'corr', 'time', 'gid', 'sch_sk', 'sch_name', 'province', 'city',
                'area']
data['ts_pts'] = 0
data['dur_weight'] = 0
data['win_proba_weight'] = 0

data.to_csv('final_df/r5(20180816-20190831)/demo_data.csv')
