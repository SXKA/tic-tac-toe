from typing import Dict

import numpy as np

from tf_agents.environments.examples.tic_tac_toe_environment import TicTacToeEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import (
    StepType,
    TimeStep
)

REWARD_ILLEGAL_MOVE = np.asarray(-5, dtype=np.float32)


def print_tic_tac_toe(state):
    table_str = '''
    {} | {} | {}
    - + - + -
    {} | {} | {}
    - + - + -
    {} | {} | {}
    '''.format(*tuple(state.flatten()))
    table_str = table_str.replace('0', ' ')
    table_str = table_str.replace('1', 'X')
    table_str = table_str.replace('2', 'O')
    print(table_str)


class TicTacToeMultiAgentEnvironment(TicTacToeEnvironment):
    def action_spec(self) -> Dict:
        position_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=8)
        value_spec = BoundedArraySpec((), np.int32, minimum=1, maximum=2)
        return {
            "position": position_spec,
            "value": value_spec
        }

    def _step(self, action: np.ndarray) -> TimeStep:
        if self._current_time_step.is_last():
            return self._reset()

        index_flat = np.array(range(9)) == action["position"]
        index = index_flat.reshape(self._states.shape) == True
        if self._states[index] != 0:
            return TimeStep(StepType.LAST,
                            REWARD_ILLEGAL_MOVE,
                            self._discount,
                            self._states)

        self._states[index] = action["value"]

        is_final, reward = self._check_states(self._states)

        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(step_type, reward, self._discount, self._states)
