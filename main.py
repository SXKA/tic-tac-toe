import pickle

import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories.time_step import TimeStep

from tic_tac_toe_multi_agent_environment import (
    print_tic_tac_toe,
    REWARD_ILLEGAL_MOVE,
    TicTacToeMultiAgentEnvironment
)


def main():
    tf_ttt_env = TFPyEnvironment(TicTacToeMultiAgentEnvironment())
    ts = tf_ttt_env.reset()

    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    policy = q_table
    first = input("First? (Y/N): ").upper() == 'Y'

    while True:
        if first:
            ts = tf_ttt_env.step({"position": ask_move(ts), "value": 1})
            print_tic_tac_toe(ts.observation.numpy())

            if ts.is_last():
                if ts.reward == TicTacToeMultiAgentEnvironment.REWARD_DRAW_OR_NOT_FINAL:
                    print("Draw!")
                elif ts.reward == TicTacToeMultiAgentEnvironment.REWARD_WIN:
                    print("Human win!")
                else:
                    print("Illegal move!")
                break

            ts = policy_step(tf_ttt_env, policy, ts, 2)
            if ts.is_last():
                if ts.reward == REWARD_ILLEGAL_MOVE:
                    print("Illegal move!")
                elif ts.reward == TicTacToeMultiAgentEnvironment.REWARD_DRAW_OR_NOT_FINAL:
                    print("Draw!")
                else:
                    print("Computer win!")
                break
        else:
            ts = policy_step(tf_ttt_env, policy, ts, 1)
            if ts.is_last():
                if ts.reward == REWARD_ILLEGAL_MOVE:
                    print("Illegal move!")
                elif ts.reward == TicTacToeMultiAgentEnvironment.REWARD_DRAW_OR_NOT_FINAL:
                    print("Draw!")
                else:
                    print("Computer win!")
                break

            ts = tf_ttt_env.step({"position": ask_move(ts), "value": 2})
            print_tic_tac_toe(ts.observation.numpy())

            if ts.is_last():
                if ts.reward == REWARD_ILLEGAL_MOVE:
                    print("Illegal move!")
                elif ts.reward == TicTacToeMultiAgentEnvironment.REWARD_DRAW_OR_NOT_FINAL:
                    print("Draw!")
                else:
                    print("Human win!")
                break


def ask_move(ts: TimeStep) -> int:
    """Ask the human for a move until it is a free cell 0-8.

    A typo used to crash the game (ValueError) or end it as an illegal move.

    Args:
        ts (TimeStep): Current time step; its observation is the board.

    Returns:
        int: Board position 0-8.
    """
    board = ts.observation.numpy().reshape(-1)

    while True:
        text = input("Move pos (0-8): ").strip()

        if text.isdigit() and 0 <= int(text) <= 8 and board[int(text)] == 0:
            return int(text)

        print("Please enter the number of an empty cell (0-8).")


def policy_step(env: TFPyEnvironment, policy, ts: TimeStep, value: int) -> TimeStep:
    step_type = tf.reshape(ts.step_type, ())
    reward = tf.reshape(ts.reward, ())
    discount = tf.reshape(ts.discount, ())
    obs = tf.reshape(ts.observation, (3, 3))
    ts = TimeStep(step_type, reward, discount, obs)
    ps = policy.action(ts)
    ts = env.step({"position": ps.action, "value": value})
    print_tic_tac_toe(ts.observation.numpy())

    return ts


if __name__ == "__main__":
    main()
