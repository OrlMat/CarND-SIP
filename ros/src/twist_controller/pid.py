
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, u_rising_slew_rate, u_falling_slew_rate, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = mn
        self.u_max = mx

        self.u_rising_slew_rate = u_rising_slew_rate
        self.u_falling_slew_rate = u_falling_slew_rate

        self.error_i = 0.
        self.error_prev = 0.
        self.u_prev = 0

    def reset(self):
        self.error_i = 0.0

    def step(self, error, delta_t):

        # Calculate derivative and integral parts
        error_i = self.error_i + error * delta_t
        error_d = (error - self.error_prev) / delta_t

        # Calculate control
        u = self.kp * error + self.ki * error_i + self.kd * error_d

        # Saturate control with control rate limiter
        u_rising_delta = self.u_rising_slew_rate * delta_t
        u_falling_delta = self.u_falling_slew_rate * delta_t
        u_max = min(self.u_prev + u_rising_delta, self.u_max)
        u_min = max(self.u_prev - u_falling_delta, self.u_min)
        u_sat = max(u_min, min(u, u_max))

        # Anti Windup
        u_oversaturated = u_sat - u
        error_i += u_oversaturated

        # Preserve last error data
        self.error_prev = error
        self.error_i = error_i
        self.u_prev = u_sat

        # Return control
        return u_sat
