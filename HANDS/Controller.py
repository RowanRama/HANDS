from abc import abstractmethod
import numpy as np

class BaseController():
    
    @abstractmethod
    def get_tensions(self, current_tip_position, goal_tip_position, current_time) -> np.ndarray:
        """
        Calculate the tensions for the soft manipulator to reach the target pose.

        Parameters:
        ee_pose (np.ndarray): The current end-effector pose.
        target_pose (np.ndarray): The target pose to reach.

        Returns:
        np.ndarray: The calculated tensions for the soft manipulator.
        """
        pass

class PIDController(BaseController):
    """
    A PID controller for controlling the soft manipulator.
    """
    def __init__(self, Kp = 15.0, Ki = 20.0, Kd = 0.0, max_tension = 10.0):
        """
        Initialize the PID controller.
        
        Parameters:
        - Kp: Proportional gain
        - Ki: Integral gain
        - Kd: Derivative gain
        - max_tension: To limit the output value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_tension = max_tension

        # Internal variables
        self._prev_error = np.zeros(3)
        self._integral = np.zeros(3)
        # self._last_time = 0.0

    def get_tensions(self, current_tip_position, goal_tip_position, dt) -> np.ndarray:
        """
        Compute the control tensions for a PID controller based on the current and goal tip positions.
        This method calculates the proportional, integral, and derivative terms of a PID controller
        to determine the control output. The output is then converted into tensions for a 4-tendon system.
        Args:
            current_tip_position (np.ndarray): A 1D array of 2 or 3 elements representing the current position of the tip.
            goal_tip_position (np.ndarray): A 1D array of 2 or 3 elements representing the desired goal position of the tip.
            dt (float): Time step since the last control update.
        Returns:
            np.ndarray: A 1D array of 4 elements representing the computed tensions for the system.
        Raises:
            ValueError: If `current_tip_position` or `goal_tip_position` is not a 1D array of 2 or 3 elements.
        Notes:
            - The method assumes that the PID gains (Kp, Ki, Kd) and internal states (_integral, _prev_error, _last_time)
              are already initialized.
            - The computed tensions are determined based on the sign of the control values for each axis.
        """
        # Check if the current_tip_position is a 1D array of 2 or 3 elements
        if current_tip_position.ndim != 1 or current_tip_position.size not in [2, 3]:
            raise ValueError(f"current_tip_position should be a 1D array of 2 or 3 elements. Got array of shape {current_tip_position.shape}.")
        if goal_tip_position.ndim != 1 or goal_tip_position.size not in [2, 3]:
            raise ValueError(f"goal_tip_position should be a 1D array of 2 or 3 elements. Got array of shape {goal_tip_position.shape}.")
        # Check if the current_tip_position is a 1D arrary of 2 elements
        error = current_tip_position - goal_tip_position
        #dt = current_time - self._last_time if self._last_time is not None else 0

        # print(f"Error: {error}, dt: {dt}")
        # Proportional term
        P = self.Kp * error

        # Integral term
        self._integral += error * dt
        I = self.Ki * self._integral

        # Derivative term
        D = 0
        if dt > 0:
            D = self.Kd * (error - self._prev_error) / dt

        # Compute the output
        control_value = P + I + D
        tensions = np.zeros(4)
        tensions[0], tensions[1] = (control_value[0], 0) if control_value[0] > 0 else (0, -control_value[0])
        tensions[2], tensions[3] = (control_value[1], 0) if control_value[1] > 0 else (0, -control_value[1])
        

        # Update internal state
        self._prev_error = error
        # self._last_time = current_time

        return tensions
    
class RLController(BaseController):
    """
    A Reinforcement Learning controller for controlling the soft manipulator.
    """
    def __init__(self, model, max_tension = 10.0):
        """
        Initialize the RL controller.
        
        Parameters:
        - model: The RL model to use for controlling the soft manipulator.
        - max_tension: To limit the output value
        """
        self.model = model
        self.max_tension = max_tension

    def get_tensions(self, current_tip_position, goal_tip_position, current_time) -> np.ndarray:
        """
        Compute the control tensions using a Reinforcement Learning model.
        
        Parameters:
        - current_tip_position: The current position of the tip.
        - goal_tip_position: The target position of the tip.
        - current_time: The current time in seconds.

        Returns:
        - tensions: The calculated tensions for the soft manipulator.
        """
        # Convert the current and goal positions to a suitable format for the model
        state = np.concatenate((current_tip_position, goal_tip_position))

        # Get the action from the RL model
        action = self.model.predict(state)

        # Convert the action to tensions
        tensions = np.clip(action, -self.max_tension, self.max_tension)

        return tensions