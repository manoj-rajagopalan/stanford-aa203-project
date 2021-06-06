from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as MplFigCanvas
import matplotlib.figure

from fsm_state import FsmState

class XyPlot(MplFigCanvas):
    def __init__(self, parent, title, width, height, dpi=100):
        fig = matplotlib.figure.Figure(figsize=(width,height), dpi=dpi)
        self.distance_axes = fig.subplots()
        self.distance_axes.set_xlabel('t')
        self.distance_axes.set_title(title)
        self.angle_axes = self.distance_axes.twinx()
        super().__init__(fig)
        self.setParent(parent)
        self.setUpdatesEnabled(True)
    # /__init__()
# /class XyPlot

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, width, height, robot):
        super(MainWindow, self).__init__()
        self.robot = robot
        self.setupUi(width, height)
        self.setupAnimation()
        self.resize(width+800, height)
        self.updateGeometry()
    # /__init__()

    def setupUi(self, width, height):
        hlayout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel()
        robot_canvas = QtGui.QPixmap(width, height)
        self.label.setPixmap(robot_canvas)
        hlayout.addWidget(self.label)
        plots_vlayout = QtWidgets.QVBoxLayout()
        self.state_plot = XyPlot(self, 'State', 300, self.label.height()//2)
        plots_vlayout.addWidget(self.state_plot)
        self.control_plot = XyPlot(self, 'Control', 300, self.label.height()//2)
        plots_vlayout.addWidget(self.control_plot)
        hlayout.addItem(plots_vlayout)
        
        shortcut_replay = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self)
        shortcut_replay.activated.connect(self.replay)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(hlayout)
        self.setCentralWidget(central_widget)
    # /setupUi()

    def setupAnimation(self):
        # control loop
        self.controller_timer = QtCore.QTimer()
        self.controller_timer.timeout.connect(self.robot.update)
        self.controller_timer.start(10) # ms => 100 Hz

        # animations
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self.render)
        self.render_timer.start(25) # ms => 40 fps
    # /setupAnimation()

    def replay(self):
        self.robot.drive()
    #/

    def render(self):
        qpainter = QtGui.QPainter(self.label.pixmap())

        # background
        bg_brush = QtGui.QBrush()
        bg_brush.setColor(QtCore.Qt.white)
        bg_brush.setStyle(QtCore.Qt.SolidPattern)
        qpainter.setBrush(bg_brush)
        qpainter.drawRect(0,0, self.label.width(), self.label.height())

        # loop
        if self.robot.fsm_state == FsmState.IDLE:
            self.state_plot.distance_axes.cla()
            self.state_plot.angle_axes.cla()
            self.control_plot.distance_axes.cla()
            self.control_plot.angle_axes.cla()
        # /if

        # screen coords (top left, downwards) -> math coords (bottom left, upwards)
        qpainter.translate(0, self.label.height()-1)
        qpainter.scale(1, -1)

        self.robot.render(qpainter)

        qpainter.end()

        self.robot.plotTrajectory(self.state_plot, self.control_plot)
        self.update()
    # /render()

# /class MainWindow
