"""Tests for the Q-learning update rule.

TF-Agents (and TensorFlow) are heavy, so the environment module is replaced by a
small pure-NumPy stand-in that follows the same reward convention: rewards are
from player 1's perspective, and an illegal move ends the game with -5 while
leaving the board unchanged.
"""

import sys
import types
from itertools import cycle

import numpy as np
import pytest

REWARD_ILLEGAL_MOVE = np.asarray(-5, dtype=np.float32)


class FakeTimeStep:
    def __init__(self, last, reward, observation):
        self._last = last
        self.reward = np.asarray(reward, dtype=np.float32)
        self.observation = observation.copy()

    def is_last(self):
        return self._last


class FakeEnv:
    REWARD_WIN = np.asarray(1.0, dtype=np.float32)
    REWARD_LOSS = np.asarray(-1.0, dtype=np.float32)
    REWARD_DRAW_OR_NOT_FINAL = np.asarray(0.0, dtype=np.float32)

    def __init__(self, board=None):
        self.board = np.zeros((3, 3), dtype=np.int32) if board is None else board
        self._ts = FakeTimeStep(False, 0.0, self.board)

    def current_time_step(self):
        return self._ts

    def step(self, action):
        i, j = divmod(int(action["position"]), 3)
        if self.board[i, j] != 0:
            self._ts = FakeTimeStep(True, REWARD_ILLEGAL_MOVE, self.board)
            return self._ts
        self.board[i, j] = action["value"]
        lines = list(self.board) + list(self.board.T) + [self.board.diagonal(), np.fliplr(self.board).diagonal()]
        for line in lines:
            if line[0] != 0 and np.all(line == line[0]):
                reward = self.REWARD_WIN if line[0] == 1 else self.REWARD_LOSS
                self._ts = FakeTimeStep(True, reward, self.board)
                return self._ts
        self._ts = FakeTimeStep(not np.any(self.board == 0), 0.0, self.board)
        return self._ts


@pytest.fixture(scope="module")
def q_learning():
    env_module = types.ModuleType("tic_tac_toe_multi_agent_environment")
    env_module.REWARD_ILLEGAL_MOVE = REWARD_ILLEGAL_MOVE
    env_module.TicTacToeMultiAgentEnvironment = FakeEnv
    q_table_module = types.ModuleType("q_table")
    q_table_module.QTable = object
    sys.modules["tic_tac_toe_multi_agent_environment"] = env_module
    sys.modules["q_table"] = q_table_module
    import q_learning as module

    yield module
    for name in ("tic_tac_toe_multi_agent_environment", "q_table", "q_learning"):
        sys.modules.pop(name, None)


def run_one_move(q_learning, monkeypatch, board, action, replies=()):
    """Let player 1 play `action` from `board`, followed by scripted replies."""
    env = FakeEnv(board)
    q_a = np.zeros((3 ** 9, 9))
    q_b = np.zeros((3 ** 9, 9))
    # A large value everywhere makes any accidental bootstrap visible.
    q_a[:, :] = 100.0
    state = q_learning.obs_to_state(board)
    q_a[state, action] = 0.0
    monkeypatch.setattr(q_learning, "softmax_action", lambda q, s: action)
    reply_iter = iter(replies)
    monkeypatch.setattr(q_learning, "softmax_legal_action", lambda q, s: next(reply_iter))
    q_learning.learning_iteration(env, q_a, cycle([q_a, q_b]), cycle([1, 2]), 0)
    return q_a[state, action]


def test_winning_move_does_not_bootstrap(q_learning, monkeypatch):
    board = np.array([[1, 1, 0], [2, 2, 0], [0, 0, 0]], dtype=np.int32)
    updated = run_one_move(q_learning, monkeypatch, board, action=2)
    assert updated == pytest.approx(q_learning.LEARNING_RATE * 1.0)


def test_illegal_move_does_not_bootstrap_from_itself(q_learning, monkeypatch):
    board = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int32)
    updated = run_one_move(q_learning, monkeypatch, board, action=4)
    assert updated == pytest.approx(q_learning.LEARNING_RATE * -5.0)


def test_losing_reply_does_not_bootstrap(q_learning, monkeypatch):
    # Player 1 plays 8; player 2 replies 5 and completes the middle row.
    board = np.array([[1, 0, 1], [2, 2, 0], [0, 0, 0]], dtype=np.int32)
    updated = run_one_move(q_learning, monkeypatch, board, action=8, replies=[5])
    assert updated == pytest.approx(q_learning.LEARNING_RATE * -1.0)


def test_non_terminal_move_bootstraps(q_learning, monkeypatch):
    board = np.zeros((3, 3), dtype=np.int32)
    env = FakeEnv(board)
    q_a = np.zeros((3 ** 9, 9))
    after = np.zeros((3, 3), dtype=np.int32)
    after[0, 0], after[1, 1] = 1, 2
    q_a[q_learning.obs_to_state(after), :] = 10.0
    actions = iter([0, 1, 3])
    replies = iter([4, 2, 6])
    monkeypatch.setattr(q_learning, "softmax_action", lambda q, s: next(actions))
    monkeypatch.setattr(q_learning, "softmax_legal_action", lambda q, s: next(replies))
    q_learning.learning_iteration(env, q_a, cycle([q_a, np.zeros_like(q_a)]), cycle([1, 2]), 0)
    expected = q_learning.LEARNING_RATE * (0.0 + q_learning.GAMMA * 10.0)
    assert q_a[0, 0] == pytest.approx(expected)


def test_obs_to_state_round_trips_with_ternary(q_learning):
    board = np.array([[1, 0, 2], [0, 2, 0], [1, 0, 0]])
    state = q_learning.obs_to_state(board)
    digits = q_learning.ternary(int(state))[::-1]
    assert [int(d) for d in digits] == list(board.flatten())
