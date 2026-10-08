# RL_gomoku
强化学习五子棋的代码包，包含了五子棋的环境，以及强化学习代码。

# 算法  
蒙特卡洛，并行计算，强化学习


# env 输入输出
    Gymnasium 五子棋环境

    observation:
        shape = (6, size, size)

        channel 0 = 原始 board（取值 -1 / 0 / 1）
        channel 1 = 黑子
        channel 2 = 白子
        channel 3 = 当前玩家是黑还是白（黑=1，白=0）
        channel 4 = 黑子的最近一步
        channel 5 = 白子的最近一步

    action:
        0 ~ size * size - 1

    step 接口:
        env.step(action, board=None)

        - action : int
        - board  : 可选。若提供，则用它作为当前棋盘状态，
                   形状必须为 (size, size)，取值 {-1, 0, 1}。
                   若为 None，则使用环境内部维护的棋盘。


# 模型
    模型输入   
        shape = (6, size, size)
    模型输出
        policy 输出 [B, 15*15]：每个落子位置的先验概率 logits；
        value 输出 [B]：当前局面的状态价值；范围 [-1, 1]；
        win_logit 输出 [B]：当前玩家最终获胜概率的 logit。




# 更新日志 

2026年10月8日 
1. 将UI都放到了play.py文件中，gomoku env 只包含最基础的五子棋环境。简化了代码
2. 环境输出为6个：
        observation:
        shape = (6, size, size)

        channel 0 = 原始 board（取值 -1 / 0 / 1）
        channel 1 = 黑子
        channel 2 = 白子
        channel 3 = 当前玩家是黑还是白（黑=1，白=0）
        channel 4 = 黑子的最近一步
        channel 5 = 白子的最近一步
3. 支持输入棋盘



