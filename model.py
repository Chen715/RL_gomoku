'''
模型都在这定义

'''
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
    

#定义模型，价值函数和策略函数共享卷积层


# ============================================================
# 2. 神经网络
# ============================================================


class ResidualBlock(nn.Module):
    """两层残差块，可选 KataGo 风格 Global Pooling Bias。

    五子棋的"连五"威胁经常横跨大半个棋盘，单纯靠卷积堆深度传递这种远距离信息效率不高
    （15x15 棋盘上，4 个残差块的感受野直径就已经覆盖全盘了，继续堆深度主要是加非线性
    表达能力，不是"看得更远"）。Global Pooling Bias 的做法是：对 conv1 之后的特征图做
    全局均值/最大值池化，过一个小 FC 投影成每通道一个 bias，广播加回 conv2 的输出，
    让每个残差块都能直接拿到"全局棋盘状态摘要"，而不必完全依赖深度传递。
    """

    def __init__(self, channels: int, use_global_pool_bias: bool = True):
        super().__init__()
        self.use_global_pool_bias = bool(use_global_pool_bias)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if self.use_global_pool_bias:
            # 输入是 [mean, max] 两种全局池化的拼接，输出是每通道一个 bias。
            self.gpool_fc = nn.Linear(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.conv2(h)
        if self.use_global_pool_bias:
            mean_pool = h.mean(dim=(2, 3))
            max_pool = h.amax(dim=(2, 3))
            pooled = torch.cat([mean_pool, max_pool], dim=1)
            bias = self.gpool_fc(pooled).unsqueeze(-1).unsqueeze(-1)
            out = out + bias
        out = self.bn2(out)
        return F.relu(out + identity, inplace=True)


class GomokuNet(nn.Module):
    """Policy + Value + Win-Probability 三头网络。

    三个输出分别用于：
        policy 输出 [B, 15*15]：每个落子位置的先验概率 logits；
        value 输出 [B]：当前局面的状态价值；范围 [-1, 1]；
        win_logit 输出 [B]：当前玩家最终获胜概率的 logit。


    win 概率头不是 PPO，而是 AlphaZero 主干上的辅助预测头。它让模型可以直接给出
    更容易解释的胜率估计，并以较小权重参与 MCTS 的叶节点价值计算。
    """
    
    
    """
     输入通道变为6 5（新增"最近一手"平面），见 Gomoku.get_state()。
        shape = (6, size, size)

        channel 0 = 原始 board（取值 -1 / 0 / 1）
        channel 1 = 黑子
        channel 2 = 白子
        channel 3 = 当前玩家是黑还是白（黑=1，白=0）
        channel 4 = 黑子的最近一步
        channel 5 = 白子的最近一步
    """
    IN_CHANNELS = 6


    def __init__(
        self,
        board_size: int = 15,
        channels: int = 128,
        blocks: int = 8,
        use_global_pool_bias: bool = True,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.stem = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *(ResidualBlock(channels, use_global_pool_bias=use_global_pool_bias) for _ in range(blocks))
        )

        # 策略头保持 AlphaGo Zero 论文原版设计：1x1 卷积压到 2 通道再接全连接。
        # （注：Lc0 用的是 32 通道，那是 Lc0 自己的设计选择，不是"标准 AlphaZero"；
        #  这里保留论文原版写法，如果想做 Lc0 风格的宽策略头，可以把 2 改成 32/64 做对比实验。）
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

        # 显式胜率头：输出 logit，训练时使用 BCEWithLogitsLoss。
        # 使用独立 head 而不是直接由 value=(2*p-1) 推导，给网络一个专门学习概率校准的通道。
        self.win_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        win_logit = self.win_head(x).squeeze(-1)
        return policy, value, win_logit



def main():
    board_size = 15
    batch_size = 2
    model = GomokuNet(board_size=board_size, channels=32, blocks=2)
    model.eval()

    # 模拟环境返回的观察张量: [batch, channels, height, width]
    inputs = torch.rand(batch_size, GomokuNet.IN_CHANNELS, board_size, board_size)

    with torch.no_grad():
        policy_logits, values, win_logits = model(inputs)

    print(f"输入 shape: {tuple(inputs.shape)}")
    print(f"策略输出 shape: {tuple(policy_logits.shape)}")
    print(f"当前局面的状态价值 shape: {tuple(values.shape),values}")
    print(f"当前玩家最终获胜概率的 logit: {tuple(win_logits.shape),win_logits}")

    
    print("策略 logits（第一个样本前 10 个动作）:", policy_logits[0, :10])
    print("价值（每个样本一个）:", values)
    print("胜率概率（每个样本一个）:", torch.sigmoid(win_logits))


if __name__ == "__main__":
    main()
