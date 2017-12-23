from yaw_controller import YawController 
from pid import PID 
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# PID gain coeffs
kp = 0.3
ki = 0.1
kd = 0.02
u_rising_rate = 0.75
u_falling_rate = 3.0

# LowPassFilter paramters
tau = 3
ts = 1


class Controller(object):
    def __init__(self, vehicle_params):
        self.yaw_controller = YawController(
            wheel_base=vehicle_params.wheel_base,
            steer_ratio=vehicle_params.steer_ratio,
            max_lat_accel=vehicle_params.max_lat_accel,
            min_speed=vehicle_params.min_speed,
            max_steer_angle=vehicle_params.max_steer_angle)
        
        self.vehicle_params = vehicle_params
        self.pid = PID(kp, ki, kd, u_rising_rate, u_falling_rate,
                       mn=vehicle_params.decel_limit, mx=vehicle_params.accel_limit)
        self.s_lpf = LowPassFilter(tau, ts)
        self.t_lpf = LowPassFilter(tau, ts)

    def control(self, twist_cmd, current_velocity, sample_time):

        # Rewrite and calculate basic car state
        linear_velocity = twist_cmd.twist.linear.x
        angular_velocity = twist_cmd.twist.angular.z
        current_linear_velocity = current_velocity.twist.linear.x
        velocity_error = linear_velocity - current_linear_velocity

        # Lateral control
        u_steering = self.lat_ctrl(linear_velocity, angular_velocity, current_linear_velocity)

        # Longitudinal control
        u_throttle, u_break = self.lon_ctrl(velocity_error, sample_time)

        return u_throttle, u_break, u_steering

    def lon_ctrl(self, velocity_error, sample_time):

        # PID step
        accel = self.pid.step(velocity_error, sample_time)

        # Steering filtering
        accel_filtered = self.t_lpf.filt(accel)

        # Control logic
        if accel > 0.0:
            # Throttle control
            u_throttle = accel_filtered
            u_brake = 0.0

        else:
            # Breaking control
            u_throttle = 0.0
            decel = -accel

            # Check for too small deceleration to use breaks
            if decel < self.vehicle_params.brake_deadband:
                u_brake = 0.0
            else:
                u_brake = decel * self.vehicle_params.wheel_radius * (self.vehicle_params.vehicle_mass +
                                                                      self.vehicle_params.fuel_capacity * GAS_DENSITY)

        return u_throttle, u_brake

    def lat_ctrl(self, linear_vel, angular_vel, linear_vel_curr):

        # Derive control for steering
        get_steer = self.yaw_controller.get_steering(linear_vel,
                                                     angular_vel,
                                                     linear_vel_curr)
        # Filter control with use of low pass filter
        steering_filtered = self.s_lpf.filt(get_steer)

        return steering_filtered
