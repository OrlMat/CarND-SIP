
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.error_i = 0.
        self.error_prev = 0.

    def reset(self):
        self.error_i = 0.0

    def step(self, error, delta_t):

        # Calculate derivative and integral parts
        error_i = self.error_i + error * delta_t
        error_d = (error - self.error_prev) / delta_t

        # Calculate control
        u = self.kp * error + self.ki * error_i + self.kd * error_d
        u_sat = max(self.min, min(u, self.max))

        # Anti Windup
        u_oversaturated = u_sat - u
        error_i += u_oversaturated

        # Preserve last error data
        self.error_prev = error
        self.error_i = error_i

        # Return control
        return u_sat
