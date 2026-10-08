import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple


class Gomoku(gym.Env):
    """
    Gymnasium 五子棋环境（纯逻辑，不含任何渲染代码）。

    board:
        0  = 空
        1  = 黑棋
        -1 = 白棋

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

    渲染 / 鼠标交互由外部（例如 play.py 中的 BoardRenderer）负责。
    """

    def __init__(
        self,
        size: int = 15,
        render_mode=None,
        dtype=np.float16,
    ):
        super().__init__()

        self.size = int(size)
        self.action_size = self.size * self.size
        self.render_mode = render_mode

        # dtype 可配置：默认 float16 以节省内存；
        # 若下游需要更高精度，可传 dtype=np.float32
        self.dtype = np.dtype(dtype)

        # ====================================================
        # Gymnasium spaces
        # ====================================================

        self.action_space = spaces.Discrete(self.action_size)

        # channel 0 为原始 board，取值范围 {-1, 0, 1}
        # 因此 low = -1.0
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6, self.size, self.size),
            dtype=self.dtype,
        )

        # ====================================================
        # 游戏状态
        # ====================================================

        self.board = None
        self.current_player = None
        self.done = None
        self.winner = None
        self.move_count = None
        self.last_action = None
        self.black_last_action = None
        self.white_last_action = None

        self.reset()

    # ========================================================
    # Gymnasium API
    # ========================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.size, self.size), dtype=np.int8)

        # 黑棋先手
        self.current_player = 1

        self.done = False
        self.winner = 0
        self.move_count = 0
        self.last_action = -1
        self.black_last_action = -1
        self.white_last_action = -1

        observation = self.get_state()

        info = {
            "current_player": self.current_player,
            "winner": self.winner,
            "move_count": self.move_count,
        }

        return observation, info

    def step(self, action: int, board=None):
        """
        Gymnasium API：

            observation, reward, terminated, truncated, info

        新增 board 参数：
            board 为 None  → 使用环境内部棋盘（旧行为）
            board 不为 None → 用传入的 board 覆盖内部棋盘后再落子
        """

        if self.done:
            raise ValueError("棋局已经结束，不能继续 step()")

        # ====================================================
        # 如果外部传入 board，则用它作为当前棋盘
        # ====================================================

        if board is not None:
            board_arr = np.asarray(board)

            if board_arr.shape != (self.size, self.size):
                raise ValueError(
                    f"board 形状必须为 ({self.size}, {self.size})，"
                    f"实际为 {board_arr.shape}"
                )

            self.board = board_arr.astype(np.int8, copy=True)

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"action 超出范围: {action}")

        row, col = self.action_to_pos(action)

        player = self.current_player

        # ====================================================
        # 非法动作
        # ====================================================

        if self.board[row, col] != 0:
            self.done = True
            self.winner = -player

            # 保持原环境逻辑：
            # 终局后切换 current_player
            self.current_player = -player

            observation = self.get_state()

            info = {
                "illegal": True,
                "winner": self.winner,
            }

            return observation, -1.0, True, False, info

        # ====================================================
        # 正常落子
        # ====================================================

        self.board[row, col] = player
        self.move_count += 1
        self.last_action = action

        if player == 1:
            self.black_last_action = action
        else:
            self.white_last_action = action

        # ====================================================
        # 判断胜利
        # ====================================================

        if self.check_win(row, col, player):
            self.done = True
            self.winner = player

            self.current_player = -player

            observation = self.get_state()

            info = {
                "winner": player,
                "winning_action": action,
            }

            return observation, 1.0, True, False, info

        # ====================================================
        # 判断和棋 和棋给 0.5 奖励
        # ====================================================

        if self.move_count >= self.action_size:
            self.done = True
            self.winner = 0

            self.current_player = -player

            observation = self.get_state()

            info = {
                "winner": 0,
            }

            return observation, 0.5, True, False, info

        # ====================================================
        # 普通状态
        # ====================================================

        self.current_player = -player

        observation = self.get_state()

        info = {
            "current_player": self.current_player,
            "winner": 0,
            "move_count": self.move_count,
        }

        return observation, 0.0, False, False, info

    # ========================================================
    # State
    # ========================================================

    def get_state(self) -> np.ndarray:
        """
        返回固定颜色视角的状态。

        channel 0: 原始 board（-1 / 0 / 1）
        channel 1: 黑子
        channel 2: 白子
        channel 3: 当前玩家是黑还是白（黑=1，白=0）
        channel 4: 黑子的最近一步
        channel 5: 白子的最近一步
        """

        dt = self.dtype

        board_channel = self.board.astype(dt)

        black = (self.board == 1).astype(dt)
        white = (self.board == -1).astype(dt)

        side_value = 1.0 if self.current_player == 1 else 0.0
        side = np.full_like(black, side_value, dtype=dt)

        black_last = np.zeros_like(black, dtype=dt)
        white_last = np.zeros_like(black, dtype=dt)

        if self.black_last_action >= 0:
            row, col = self.action_to_pos(self.black_last_action)
            black_last[row, col] = 1.0

        if self.white_last_action >= 0:
            row, col = self.action_to_pos(self.white_last_action)
            white_last[row, col] = 1.0

        return np.stack(
            [board_channel, black, white, side, black_last, white_last],
            axis=0,
        ).astype(dt, copy=False)

    # ========================================================
    # Action
    # ========================================================

    def get_valid_actions(self) -> np.ndarray:
        """
        返回合法动作 mask。

        合法动作 = 1
        非法动作 = 0
        """

        return (self.board.reshape(-1) == 0).astype(np.float32)

    def action_to_pos(self, action: int) -> Tuple[int, int]:
        return divmod(int(action), self.size)

    def pos_to_action(self, row: int, col: int) -> int:
        return int(row) * self.size + int(col)

    # ========================================================
    # Win
    # ========================================================

    def check_win(self, row: int, col: int, player: int) -> bool:
        directions = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )

        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)

            if count >= 5:
                return True

        return False

    def _count_direction(
        self, row: int, col: int, dr: int, dc: int, player: int
    ) -> int:
        count = 0
        r = row + dr
        c = col + dc

        while (
            0 <= r < self.size
            and 0 <= c < self.size
            and self.board[r, c] == player
        ):
            count += 1
            r += dr
            c += dc

        return count

    # ========================================================
    # Clone / MCTS
    # ========================================================

    def clone(self) -> "Gomoku":
        """
        创建环境的轻量副本。

        后续 MCTS 可以直接使用。
        """

        env = object.__new__(Gomoku)

        env.size = self.size
        env.action_size = self.action_size
        env.render_mode = None
        env.dtype = self.dtype

        env.action_space = spaces.Discrete(self.action_size)

        env.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6, self.size, self.size),
            dtype=self.dtype,
        )

        env.board = self.board.copy()
        env.current_player = self.current_player
        env.done = self.done
        env.winner = self.winner
        env.move_count = self.move_count
        env.last_action = self.last_action
        env.black_last_action = self.black_last_action
        env.white_last_action = self.white_last_action

        return env

    @staticmethod
    def terminal_value(env: "Gomoku") -> float:
        if env.winner == 0:
            return 0.0

        return 1.0 if env.winner == env.current_player else -1.0


from gymnasium.envs.registration import register

# 注册环境，使得可以通过 gym.make("Gomoku-v0") 来创建环境
register(
    id="Gomoku-v0",
    entry_point="gomoku_env:Gomoku",
)