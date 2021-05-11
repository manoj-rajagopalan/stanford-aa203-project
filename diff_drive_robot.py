import numpy as np
import scipy.integrate

from PyQt5 import QtGui, QtCore

class DifferentialDriveRobot:
    def __init__(self, radius, wheel_radius, wheel_thickness):
        self.radius = radius
        self.wheel_radius = wheel_radius
        self.wheel_thickness = wheel_thickness

        x = self.radius
        y = self.radius + self.wheel_thickness
        theta = 0
        self.state = np.array([x, y, theta])
    # /__init__()

    def dynamics(self, x_y_theta, t, omega_l, omega_r, wheel_radius, baseline):
        _, _, theta = x_y_theta
        v = wheel_radius * 0.5 * (omega_l + omega_r)
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = wheel_radius / baseline * (omega_r - omega_l)
        return np.array([x_dot, y_dot, theta_dot])
    # /dynamics()

    def applyControl(self, omega_l, omega_r, delta_t):
        '''
            omega_{l,r}: angular velocities of left and right wheels in rad/s
        '''
        self.state = scipy.integrate.odeint(self.dynamics,
                                            self.state,
                                            np.array([0, delta_t]),
                                            (omega_l, omega_r, self.wheel_radius, 2 * self.radius))[1]
    # /applyControls()

    # Draw this instance onto a qpainter
    def render(self, qpainter, window_height):
        x, y, theta = self.state
        # Position
        original_transform = qpainter.worldTransform()
        transform = QtGui.QTransform()
        transform.translate(x, window_height -1 -y)
        transform.rotate(-theta * 180 / np.pi) # degrees
        qpainter.setWorldTransform(transform, combine=False)

        # main frame
        brush = QtGui.QBrush()
        brush.setColor(QtCore.Qt.black)
        brush.setStyle(QtCore.Qt.SolidPattern)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0, 0), self.radius, self.radius)
        qpainter.drawRect(-self.wheel_radius, -self.radius -self.wheel_thickness, 2 * self.wheel_radius, self.wheel_thickness)
        qpainter.drawRect(-self.wheel_radius,  self.radius,                       2 * self.wheel_radius, self.wheel_thickness)

        # draw single dot to mark orientation
        brush.setColor(QtCore.Qt.yellow)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0.75 * self.radius, 0), 0.1 * self.radius, 0.1 * self.radius)
        qpainter.setWorldTransform(original_transform)
    # /render()