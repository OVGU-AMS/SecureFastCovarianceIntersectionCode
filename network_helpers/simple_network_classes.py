"""

"""

import numpy as np
import network_helpers.base_network_classes as bnc

class NoisySinSensor(bnc.SensorBase):
    def __init__(self, name, xStep, noiseStdDev, plotter, toLog=True, toPlot=True):
        self.xStep = xStep
        self.noiseStdDev = noiseStdDev
        super().__init__(name, plotter, toLog, toPlot)
        return

    def generateData(self, t):
        p = np.sin(t*self.xStep) + np.random.normal(0, self.noiseStdDev)
        return p
    
    def getDataToSendToServer(self, t, p):
        return p
    
    def plotData(self, t, d, plotter):
        plotter.scatter(t, d, c='lightgrey', marker='x')
        return

class PointTimePlottingClient(bnc.ClientBase):
    def processServerData(self, time, serverData):
        out = serverData
        return out
    
    def plotData(self, t, d, plotter):
        plotter.scatter(t, d, c='g', marker='.')
        return