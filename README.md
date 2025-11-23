Code Step by step 代码顺序

ontology_to_graph.py -> graph_ini.py -> graph_q_learning.py -> reward_update.py -> visualize_eval.py

ontology_to_graph.py 将本体文件转换为图谱，.ttl文件可用Protege软件打开查看，同时OntoKG.png可直接查看语义关系

graph_ini.py 图谱初始化与赋权

graph_q_learning.py 图谱采用Q-learning进行强化学习与路径溯源

reward_update.py 基于证据的奖励更新与路径调整

visualize_eval.py （非必要）Q-learning算法评估

