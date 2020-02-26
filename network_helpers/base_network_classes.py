"""

"""

class LoggingAgent():
    """Class which provides logging capability given a name."""
    def __init__(self, name, toLog=True):
        self._name = name
        self._toLog = toLog
        return

    def log(self, msg):
        if self._toLog:
            print(str(self._name) + ": " + str(msg))
        return

class PlottingAgent(LoggingAgent):
    """Class which provides plotting capability given a plotter."""
    def __init__(self, name, plotter, toLog=True, toPlot=True, toPlotSent=False):
        super().__init__(name, toLog)
        self._toPlot = toPlot
        self._toPlotSent = toPlotSent
        self._plotter = plotter
        return
    
    def plot(self, time, data, plotter):
        if self._toPlot:
            self.plotData(time, data, plotter)
    
    def plotSent(self, time, data, plotter):
        if self._toPlotSent:
            self.plotSentData(time, data, plotter)
    
    def plotData(self, time, data, plotter):
        raise NotImplementedError
    
    def plotSentData(self, time, data, plotter):
        raise NotImplementedError

class SensorBase(PlottingAgent):
    """The base class for a sensor. Logs and plots generated data, and processes and logs data to be sent to the server."""
    def generateDataAndDisplay(self, time, groundTruth):
        d = self.generateData(time, groundTruth)
        self.log('Observed data: ' + str(d))
        self.plot(time, d, self._plotter)
        return d
    
    def getDataToSendToServerAndDisplay(self, time, data):
        d = self.getDataToSendToServer(time, data)
        self.log('Sending data: ' + str(d))
        self.plotSent(time, d, self._plotter)
        return d
    
    def generateData(self, time, groundTruth):
        raise NotImplementedError

    def getDataToSendToServer(self, time, data):
        raise NotImplementedError


class ServerBase(LoggingAgent):
    """The base class for a server. Logs recieved data, and processes and logs data to be sent to the client."""
    def getDataToSendToClientFromSensorDataListAndDisplay(self, time, sensorDataList):
        self.log('Recieving data: ' + str(sensorDataList))
        d = self.getDataToSendToClientFromSensorDataList(time, sensorDataList)
        self.log('Sending data: ' + str(d))
        return d
    
    def getDataToSendToClientFromSensorDataList(self, time, sensorDataList):
        raise NotImplementedError


class ClientBase(PlottingAgent):
    """The base class for a client. Logs recieved data from the server, and processed and logs/plots data."""
    def processServerDataAndDisplay(self, time, serverData):
        self.log('Recieving data: ' + str(serverData))
        self.plotSent(time, serverData, self._plotter)
        d = self.processServerData(time, serverData)
        self.log('Computed data: ' + str(d))
        self.plot(time, d, self._plotter)
        return d
    
    def processServerData(self, time, serverData):
        raise NotImplementedError