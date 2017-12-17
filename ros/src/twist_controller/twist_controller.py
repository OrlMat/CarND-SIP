from yaw_controller import YawController 
from pid import PID 
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# PID gain coeffs
kp = 5
ki = 0.5
kd = 0.5

# LowPassFilter paramters
tau = 3
ts = 1

class Controller(object):
    def __init__(self, vehicle_params):
        self.yaw_controller = YawController(
            wheel_base=vehicle_params.wheel_base,
            steer_ratio=vehicle_params.steer_ratio,
            max_lat_accel=vehicle_params.max_lat_accel,
            max_steer_angle=vehicle_params.max_steer_angle)
        
        self.vehicle_params = vehicle_params
        self.pid = PID(kp, ki, kd, mn=vehicle_params.decel_limit, mx=vehicle_params.accel_limit)
        self.s_lpf = LowPassFilter(tau, ts)
        self.t_lpf = LowPassFilter(tau, ts)

        vehicle_mass = self.vehicle_params.vehicle_mass
        fuel_mass = self.vehicle_params.fuel_capacity * GAS_DENSITY
        wheel_radius = self.vehicle_params.wheel_radius

    def control(self, twist_cmd, current_velocity):

        linear_velocity = twist_cmd.twist.linear.x
        angular_velocity = twist_cmd.twist.angular.z
        current_linear_velocity = current_velocity.twist.linear.x
        velocity_error = linear_velocity - current_linear_velocity

        get_steer = self.yaw_controller.get_steering(linear_velocity, 
                                                     angular_velocity,
                                                     current_linear_velocity)

        steer = self.s_lpf.filt(get_steer)

        get_accel = self.pid.step(velocity_error, sample_time)
        accel = self.t_lpf.filt(get_accel)

        if accel > 0.0:
            throttle = accel
            brake = 0.0
        else:
            throttle = 0.0
            decel = -accel
            
            if decel < self.vehicle_params.brake_deadband:
                decel = 0.0

            brake = decel * wheel_radius * (vehicle_mass + fuel_mass)
 
        return throttle, brake, steer
