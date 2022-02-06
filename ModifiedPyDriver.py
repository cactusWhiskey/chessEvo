from typing import Tuple
import numpy as np
from tf_agents.drivers import py_driver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types


class ModifiedPyDriver(py_driver.PyDriver):
    def run(
            self,
            time_step: ts.TimeStep,
            policy_state: types.NestedArray = ()
    ) -> Tuple[ts.TimeStep, types.NestedArray]:
        """Run policy in environment given initial time_step and policy_state.
        Args:
          time_step: The initial time_step.
          policy_state: The initial policy_state.
        Returns:
          A tuple (final time_step, final policy_state).
        """
        num_steps = 0
        num_episodes = 0
        while num_steps < self._max_steps and num_episodes < self._max_episodes:
            # For now we reset the policy_state for non batched envs.
            if not self.env.batched and time_step.is_first() and num_episodes > 0:
                policy_state = self._policy.get_initial_state(self.env.batch_size or 1)

            action_step = self.policy.action(time_step, policy_state)
            next_time_step = self.env.step(action_step.action)

            # When using observer (for the purpose of training), only the previous
            # policy_state is useful. Therefore substitube it in the PolicyStep and
            # consume it w/ the observer.
            action_step_with_previous_state = action_step._replace(state=policy_state)
            traj = trajectory.from_transition(
                time_step, action_step_with_previous_state, next_time_step)

            if time_step.step_type != 2:
                for observer in self._transition_observers:
                    observer((time_step, action_step_with_previous_state, next_time_step))
                for observer in self.observers:
                    observer(traj)

            num_episodes += np.sum(traj.is_boundary())
            num_steps += np.sum(~traj.is_boundary())

            time_step = next_time_step
            policy_state = action_step.state

        return time_step, policy_state
