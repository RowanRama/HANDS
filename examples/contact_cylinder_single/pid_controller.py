import numpy as np

class PID_low_controller:
    def __init__(self, Kp = 3.0, Ki = 30.0, Kd = 0.0, max_tension = 10.0):
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
        self._prev_error = np.zeros(2)
        self._integral = np.zeros(2)
        self._last_time = None
        pass
    
    
    def compute(self, current_tip_position, goal_tip_position, current_time):
        """
        Compute the control tensions for a PID controller based on the current and goal tip positions.
        This method calculates the proportional, integral, and derivative terms of a PID controller
        to determine the control output. The output is then converted into tensions for a 4-tendon system.
        Args:
            current_tip_position (np.ndarray): A 1D array of 2 elements representing the current position of the tip.
            goal_tip_position (np.ndarray): A 1D array of 2 elements representing the desired goal position of the tip.
            current_time (float): The current time in seconds.
        Returns:
            np.ndarray: A 1D array of 4 elements representing the computed tensions for the system.
        Raises:
            ValueError: If `current_tip_position` or `goal_tip_position` is not a 1D array of 2 elements.
        Notes:
            - The method assumes that the PID gains (Kp, Ki, Kd) and internal states (_integral, _prev_error, _last_time)
              are already initialized.
            - The computed tensions are determined based on the sign of the control values for each axis.
        """
        
        
        # Check if the current_tip_position is a 1D arrary of 2 elements
        if current_tip_position.ndim != 1 or current_tip_position.size != 2:
            raise ValueError("current_tip_position should be a 1D array of 2 elements.")
        if goal_tip_position.ndim != 1 or goal_tip_position.size != 2:
            raise ValueError("goal_tip_position should be a 1D array of 2 elements.")
        error = current_tip_position - goal_tip_position
        dt = current_time - self._last_time if self._last_time is not None else 0

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
        tensions[0], tensions[1] = (control_value[0], 0) if tx > 0 else (0, -control_value[0])
        tensions[2], tensions[3] = (control_value[1], 0) if ty > 0 else (0, -control_value[1])
        

        # Update internal state
        self._prev_error = error
        self._last_time = current_time

        return tensions
        
        