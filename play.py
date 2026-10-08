import time

import gymnasium as gym
import numpy as np
import pygame

from gomoku_env import Gomoku


# ============================================================
# 随机 AI
# ============================================================

def random_policy(env: Gomoku) -> int:
    """
    最简单的随机策略。

    后面可以直接替换成：

        action = mcts.search(env)

    或：

        action = network.predict(observation)
    """

    valid_actions = np.flatnonzero(
        env.get_valid_actions() > 0
    )

    if len(valid_actions) == 0:
        return 0

    return int(
        env.np_random.choice(valid_actions)
    )


# ============================================================
# 棋盘渲染器（原 gomoku_env.py 中的 Pygame UI）
# ============================================================

class BoardRenderer:
    """
    把 Gomoku 环境画到 Pygame 窗口上，并负责鼠标选点。

    原 gomoku_env.py 里的 render() / _init_pygame() /
    _grid_position() / get_mouse_action() / close() 全部搬到这里。
    """

    def __init__(
        self,
        env: Gomoku,
        cell_size: int = 40,
        margin: int = 50,
        caption: str = "Gomoku - Gymnasium",
    ):
        self.env = env
        self.size = env.size

        self.cell_size = cell_size
        self.margin = margin

        self.board_pixel_size = (self.size - 1) * self.cell_size
        self.window_width = self.board_pixel_size + self.margin * 2
        self.window_height = self.board_pixel_size + self.margin * 2 + 70

        self.render_fps = 30

        pygame.init()

        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    # --------------------------------------------------------

    def _grid_position(self, row: int, col: int):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        return x, y

    # --------------------------------------------------------

    def draw(self):
        env = self.env

        # ====================================================
        # 背景
        # ====================================================

        self.window.fill((238, 200, 140))

        # ====================================================
        # 状态文字
        # ====================================================

        if env.done:
            if env.winner == 1:
                title = "Black Wins!"
            elif env.winner == -1:
                title = "White Wins!"
            else:
                title = "Draw!"
        else:
            if env.current_player == 1:
                title = "Black's Turn"
            else:
                title = "White's Turn"

        text = self.font.render(title, True, (30, 30, 30))
        self.window.blit(
            text,
            (self.margin, self.board_pixel_size + self.margin + 15),
        )

        # ====================================================
        # 棋盘
        # ====================================================

        for i in range(self.size):
            x = self.margin + i * self.cell_size
            y = self.margin + i * self.cell_size

            # 横线
            pygame.draw.line(
                self.window,
                (40, 40, 40),
                (self.margin, y),
                (self.margin + self.board_pixel_size, y),
                1,
            )

            # 竖线
            pygame.draw.line(
                self.window,
                (40, 40, 40),
                (x, self.margin),
                (x, self.margin + self.board_pixel_size),
                1,
            )

        # ====================================================
        # 星位
        # ====================================================

        if self.size == 15:
            star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]

            for row, col in star_points:
                x, y = self._grid_position(row, col)
                pygame.draw.circle(self.window, (30, 30, 30), (x, y), 4)

        # ====================================================
        # 最近一步红框
        # ====================================================

        if env.last_action >= 0:
            row, col = env.action_to_pos(env.last_action)
            x, y = self._grid_position(row, col)
            pygame.draw.rect(
                self.window,
                (220, 50, 50),
                (x - 19, y - 19, 38, 38),
                2,
            )

        # ====================================================
        # 棋子
        # ====================================================

        for row in range(self.size):
            for col in range(self.size):
                value = env.board[row, col]

                if value == 0:
                    continue

                x, y = self._grid_position(row, col)

                if value == 1:
                    # 黑棋
                    pygame.draw.circle(self.window, (25, 25, 25), (x, y), 17)
                else:
                    # 白棋
                    pygame.draw.circle(self.window, (245, 245, 245), (x, y), 17)
                    pygame.draw.circle(self.window, (30, 30, 30), (x, y), 17, 1)

        pygame.display.flip()

        self.clock.tick(self.render_fps)

    # --------------------------------------------------------

    def get_mouse_action(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()

        col = round((mouse_x - self.margin) / self.cell_size)
        row = round((mouse_y - self.margin) / self.cell_size)

        if not (0 <= row < self.size and 0 <= col < self.size):
            return None

        x, y = self._grid_position(row, col)

        distance = (mouse_x - x) ** 2 + (mouse_y - y) ** 2

        if distance > 20 ** 2:
            return None

        return int(row) * self.size + int(col)

    # --------------------------------------------------------

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

            self.window = None
            self.clock = None


# ============================================================
# Pygame 事件
# ============================================================

def process_events(renderer: BoardRenderer):
    """
    处理 UI 事件。

    返回：

        None       没有特殊事件
        "quit"     退出
        "reset"    重新开始
        action     鼠标选择的动作
    """

    for event in pygame.event.get():

        # ----------------------------------------------------
        # 点击窗口关闭
        # ----------------------------------------------------

        if event.type == pygame.QUIT:
            return "quit"

        # ----------------------------------------------------
        # 键盘
        # ----------------------------------------------------

        if event.type == pygame.KEYDOWN:

            # ESC 退出
            if event.key == pygame.K_ESCAPE:
                return "quit"

            # R 重新开始
            if event.key == pygame.K_r:
                return "reset"

        # ----------------------------------------------------
        # 鼠标
        # ----------------------------------------------------

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
        ):
            action = renderer.get_mouse_action()

            if action is not None:
                return action

    return None


# ============================================================
# 人类 vs 随机 AI
# ============================================================

def play():
    """
    黑棋：人类
    白棋：随机 AI

    操作：

        鼠标左键：落子
        R：重新开始
        ESC：退出
    """

    # --------------------------------------------------------
    # 创建 Gymnasium 环境（纯逻辑，不渲染）
    # --------------------------------------------------------

    env = gym.make("Gomoku-v0", size=15)

    observation, info = env.reset(seed=42)

    # --------------------------------------------------------
    # 创建渲染器
    # --------------------------------------------------------

    renderer = BoardRenderer(env.unwrapped)

    running = True

    while running:

        # 每轮循环先重绘棋盘
        renderer.draw()

        # ====================================================
        # 处理 Pygame 事件
        # ====================================================

        event_result = process_events(renderer)

        # ----------------------------------------------------
        # 退出
        # ----------------------------------------------------

        if event_result == "quit":
            break

        # ----------------------------------------------------
        # 重新开始
        # ----------------------------------------------------

        if event_result == "reset":
            observation, info = env.reset()
            continue

        # ====================================================
        # 如果游戏已经结束
        # ====================================================

        if env.unwrapped.done:
            time.sleep(1.5)
            observation, info = env.reset()
            continue

        # ====================================================
        # 黑棋：玩家
        # ====================================================

        if env.unwrapped.current_player == 1:

            # 鼠标还没有点击
            if not isinstance(event_result, int):
                time.sleep(0.01)
                continue

            action = event_result

            # 检查是否已经有棋子
            row, col = env.unwrapped.action_to_pos(action)

            if env.unwrapped.board[row, col] != 0:
                continue

            # ------------------------------------------------
            # Gymnasium step
            # ------------------------------------------------

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

        # ====================================================
        # 白棋：AI
        # ====================================================

        else:

            action = random_policy(env.unwrapped)

            print(f"AI action = {action}")

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            # 让玩家能看到 AI 的落子
            time.sleep(0.2)

        # 落子后立刻重绘一次
        renderer.draw()

        # ====================================================
        # Gymnasium episode 结束
        # ====================================================

        if terminated or truncated:
            print("Episode finished:", info)

            time.sleep(1.5)

            observation, info = env.reset()

    # ========================================================
    # 关闭
    # ========================================================

    renderer.close()
    env.close()


# ============================================================
# Gymnasium 随机测试
# ============================================================

def random_test():
    """
    类似 Gymnasium 官方示例：

        env.reset()

        for:
            action = env.action_space.sample()
            env.step(action)

            if terminated or truncated:
                env.reset()
    """

    env = gym.make("Gomoku-v0", size=15)

    observation, info = env.reset(seed=42)

    renderer = BoardRenderer(env.unwrapped)

    for _ in range(1000):

        renderer.draw()

        # 处理退出事件，避免窗口卡死
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                renderer.close()
                env.close()
                return

        # ====================================================
        # 这里以后替换成 MCTS / 神经网络
        # ====================================================

        action = env.action_space.sample()

        # ====================================================
        # step
        # ====================================================

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # ====================================================
        # Episode 结束
        # ====================================================

        if terminated or truncated:
            print("Episode finished:", info)
            renderer.draw()
            time.sleep(1)
            observation, info = env.reset()

    renderer.close()
    env.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Gomoku Gymnasium")
    print("=" * 60)
    print()
    print("黑棋：玩家")
    print("白棋：随机 AI")
    print()
    print("鼠标左键：落子")
    print("R：重新开始")
    print("ESC：退出")
    print("=" * 60)

    play()