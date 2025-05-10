import numpy as np

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep


class QTable:
    def __init__(self, q_table: np.array):
        self.q_table = q_table

    def action(self, ts: TimeStep) -> PolicyStep:
        from q_learning import obs_to_state

        step_type, reward, discount, obs = ts
        state = obs_to_state(obs.numpy())
        action = np.argmax(self.q_table[state])

        return PolicyStep(action, state + 3 ** action, self.q_table[state, action])
