# 校园排行榜

* 文件注释：
  * connector.py：hive连接、取数据转成DataFrame；
  * main.py：代码主体部分，包括拟合、时间的处理、过往做题记录的处理；
  * result.py：对已被处理过的做题记录进行二次加工，得到各种聚合的结果用于后序展示，如学校排名、用户积分等；
  * utils.py：在处理数据过程中用到的所有函数方法，均抽象在该文件中。
  * show.py：这两个文件用于展示使用方法
* 该项目最终进行了Workshop Demo的展示，并使用前端技术绘制了手机页面
