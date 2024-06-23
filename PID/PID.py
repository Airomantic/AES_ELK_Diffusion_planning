class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0


    def cal_output(self, error, dt):
        derivative = error - self.previous_error
        u = self.Kp * error + self.Ki * self.integral * dt + self.Kd * derivative / dt
        self.integral += error
        self.previous_error = error
        return u