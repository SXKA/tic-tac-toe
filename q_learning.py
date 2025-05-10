import pickle
import random

from functools import lru_cache
from itertools import cycle
from typing import (
    List,
    Tuple
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from q_table import QTable
from tic_tac_toe_multi_agent_environment import (
    REWARD_ILLEGAL_MOVE,
    TicTacToeMultiAgentEnvironment
)

# Hyperparameters
NUM_ITERATIONS = 2000000
LEARNING_RATE = 1e-3
GAMMA = 0.99
TAU = 5e-3
# Log information
PLOT_INTERVAL = 5000


def main():
    ttt_environment = TicTacToeMultiAgentEnvironment()

    q_learning(ttt_environment)


def q_learning(env: TicTacToeMultiAgentEnvironment):
    num_actions = 9
    num_states = 3 ** 9
    q_table_1 = np.zeros((num_states, num_actions))
    q_table_2 = np.zeros((num_states, num_actions))

    sns.set()

    illegal = [0]
    draw = [0]
    win_1 = [0]
    win_2 = [0]

    try:
        for i in range(NUM_ITERATIONS):
            illegal_count = 0
            draw_count = 0
            win_count_1 = 0
            win_count_2 = 0

            if random.getrandbits(1):
                q_table_list = [q_table_1, q_table_2]
                q_tables = cycle(q_table_list)
            else:
                q_table_list = [q_table_2, q_table_1]
                q_tables = cycle(q_table_list)

            values = cycle([1, 2])

            env.reset()

            ts, q_table = learning_iteration(env, q_table_1, q_tables, values, i)
            if ts.reward == TicTacToeMultiAgentEnvironment.REWARD_DRAW_OR_NOT_FINAL:
                draw_count += 1
            elif ts.reward == REWARD_ILLEGAL_MOVE:
                illegal_count += 1
            elif q_table is q_table_1:
                win_count_1 += 1
            else:
                win_count_2 += 1

            q_tables = cycle(q_table_list)
            values = cycle([2, 1])
            q_table = next(q_tables)

            ts = env.reset()
            state = obs_to_state(ts.observation)
            # Select an action for a given state and acts in env based on selected action
            action = softmax_action(q_table, state)
            env.step({"position": action, "value": 1})

            ts, q_table = learning_iteration(env, q_table_1, q_tables, values, i)
            if ts.reward == TicTacToeMultiAgentEnvironment.REWARD_DRAW_OR_NOT_FINAL:
                draw_count += 1
            elif ts.reward == REWARD_ILLEGAL_MOVE:
                illegal_count += 1
            elif q_table is q_table_1:
                win_count_1 += 1
            else:
                win_count_2 += 1

            if not i:
                illegal[-1] += illegal_count
                draw[-1] += draw_count
                win_1[-1] += win_count_1
                win_2[-1] += win_count_2
            else:
                illegal.append(illegal[-1] + illegal_count)
                draw.append(draw[-1] + draw_count)
                win_1.append(win_1[-1] + win_count_1)
                win_2.append(win_2[-1] + win_count_2)

            if i % PLOT_INTERVAL == 0:
                plot_history(illegal, draw, win_1, win_2)

    except KeyboardInterrupt:
        print("Interrupting training, plotting history...")
        plot_history(illegal, draw, win_1, win_2)
        save_q_table(QTable(q_table_1))

    save_q_table(QTable(q_table_1))


def learning_iteration(env: TicTacToeMultiAgentEnvironment, test_q_table: np.ndarray, q_tables: cycle, values: cycle,
                       num_iteration: int) -> Tuple:
    q_table = None

    ts = env.current_time_step()
    state = obs_to_state(ts.observation)

    while not ts.is_last():
        q_table = next(q_tables)
        value = next(values)

        # Select an action for a given state and acts in env based on selected action
        action = softmax_action(q_table, state)
        ts = env.step({"position": action, "value": value})
        next_state = obs_to_state(ts.observation)

        if ts.is_last():
            if num_iteration < NUM_ITERATIONS >> 1 or q_table is test_q_table:
                # Q update
                reward = ts.reward.copy()
                if reward == TicTacToeMultiAgentEnvironment.REWARD_LOSS:
                    reward = TicTacToeMultiAgentEnvironment.REWARD_WIN

                y = reward + GAMMA * np.max(q_table[next_state, :])
                q_table[state, action] += LEARNING_RATE * (y - q_table[state, action])
            break

        next_q_table = next(q_tables)
        next_value = next(values)

        # Select an action for a given state and acts in env based on selected action
        next_action = softmax_legal_action(next_q_table, next_state)
        ts = env.step({"position": next_action, "value": next_value})
        next_state = obs_to_state(ts.observation)

        if num_iteration < NUM_ITERATIONS >> 1 or q_table is test_q_table:
            # Q update
            reward = ts.reward.copy()
            if value == 2:
                reward = -reward

            y = reward + GAMMA * np.max(q_table[next_state, :])
            q_table[state, action] += LEARNING_RATE * (y - q_table[state, action])

        # Move to the next state
        state = next_state

        q_table = next_q_table

    return ts, q_table


def obs_to_state(obs: np.ndarray) -> int:
    i, j = np.indices(obs.shape)

    return np.sum(3 ** (3 * i + j) * obs)


def softmax_action(q_table: np.ndarray, state: int) -> float:
    actions = np.arange(9)

    exps = np.exp(q_table[state] / TAU)
    p = exps / np.sum(exps)
    return np.random.choice(actions, replace=False, p=p)


def softmax_legal_action(q_table: np.ndarray, state: int) -> float:
    state_t = np.array([c for c in ternary(state)][::-1])
    actions = np.where(state_t == '0')[0]

    exps = np.exp(q_table[state, actions] / TAU)
    p = exps / np.sum(exps)
    return np.random.choice(actions, replace=False, p=p)


@lru_cache(3 ** 9)
def ternary(num: int) -> str:
    return np.base_repr(num, 3).zfill(9)


def plot_history(illegal, draw: List, win_1: List, win_2: List):
    sns.lineplot(illegal, label="Illegal Ending")
    sns.lineplot(draw, label="Draw")
    sns.lineplot(win_1, label="Agent 1")
    sns.lineplot(win_2, label="Agent 2")

    plt.xlabel("iterations")
    plt.ylabel("count")
    plt.savefig("q_learning_result.png")
    plt.show()


def save_q_table(q_table: QTable):
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)


if __name__ == "__main__":
    main()
