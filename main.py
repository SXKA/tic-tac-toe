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
            ts = tf_ttt_env.step({"position": int(input("Move pos: ")), "value": 1})
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

            ts = tf_ttt_env.step({"position": int(input("Move pos: ")), "value": 2})
            print_tic_tac_toe(ts.observation.numpy())

            if ts.is_last():
                print("Human win!")
                break


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
