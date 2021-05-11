import numpy as np
import scipy.integrate as spint

from PyQt5 import QtGui, QtCore

class DifferentialDriveRobot:
    def __init__(self, radius, wheel_radius, wheel_thickness):
        self.radius = 50
        self.wheel_radius = 0.2 * self.radius
        self.wheel_thickness = wheel_thickness

        self.x = self.radius
        self.y = self.radius + self.wheel_thickness
        self.theta = 0

    def setPose(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    # /setPose()

    # Draw this instance onto a qpainter
    def render(self, qpainter, window_height):
        # Position
        original_transform = qpainter.worldTransform()
        transform = QtGui.QTransform()
        transform.translate(self.x, window_height -1 -self.y)
        transform.rotate(-self.theta * 180 / np.pi) # degrees
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