"""

"""

import functools as ft
import numpy as np
import matplotlib.pyplot as plt

import network_helpers.base_network_classes as bnc
import other_helpers.plotting_helper as ph
import other_helpers.encryption_simulation_classes as enc

class MovingObjectSmartSensor(bnc.SensorBase):
    def __init__(self, name, 
                trueStateTransition, 
                trueStateErrorMean,
                trueStateErrorCov, 
                trueMeasurementTransition,
                trueMeasurementErrorMean,
                trueMeasurementErrorCov,
                trueInitState,
                stateTransition,
                stateErrorCov,
                measurementTransition,
                measurementErrorCov, 
                plotter, toLog=True, toPlot=True):
        super().__init__(name, plotter, toLog, toPlot)

        # Used for true trajectory and measurments
        self.trueStateTransition = trueStateTransition
        self.trueStateErrorMean = trueStateErrorMean
        self.trueStateErrorCov = trueStateErrorCov
        self.trueMeasurementTransition = trueMeasurementTransition
        self.trueMeasurementErrorMean = trueMeasurementErrorMean
        self.trueMeasurementErrorCov = trueMeasurementErrorCov
        # The true initial position
        self.truePrevState = trueInitState

        # Used for filter
        self.F = stateTransition
        self.Q = stateErrorCov
        self.H = measurementTransition
        self.R = measurementErrorCov
        self.x = None
        self.P = None
        self.inits = []
        return

    def generateData(self, t):
        # ===== Generate ground truth
        # Generate the next state acording to model. On first time step just used the supplied value.
        if t > 0:
            signalError = np.random.multivariate_normal(self.trueStateErrorMean, self.trueStateErrorCov)
            self.truePrevState = self.trueStateTransition@self.truePrevState + signalError
        
        # Measure the true state according to model
        measureError = np.random.multivariate_normal(self.trueMeasurementErrorMean, self.trueMeasurementErrorCov)
        measurement = self.trueMeasurementTransition@self.truePrevState + measureError

        # ===== Filter true measurement
        # Special case at the start, compute initial from first 2 measurements
        if t == 0:
            self.inits.append(measurement)
            return self.truePrevState, measurement, None, None
        if t == 1:
            self.inits.append(measurement)
            # Now that we have the first 2 measurements, we can create the believed initial state
            # Make the initial x, y position be the second measurement that was gotten
            # Make the initial x, y velocity be computed from the distance between the first and second measurements that were gotten
            self.x = np.array([self.inits[1][0], (self.inits[1][0] - self.inits[0][0])/self.F[0][1], self.inits[1][1], (self.inits[1][1] - self.inits[0][1])/self.F[0][1]])
            self.P = self.Q
            return self.truePrevState, measurement, self.x, self.P
        
        # Prediction
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)

        # Update
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)

        return self.truePrevState, measurement, self.x, self.P
    
    def getDataToSendToServer(self, t, p):
        # TODO ORE and PHE and send those
        #trace = np.trace(self.P)

        return self.x, self.P
    
    def plotData(self, t, d, plotter):
        true, measurement, state, error = d
        true2D = np.array([true[0], true[2]])
        plotter.scatter(true2D[0], true2D[1], c='lightgrey', marker='.')
        plotter.scatter(measurement[0], measurement[1], c='grey', marker='x')
        
        # State and error estimates only made after first time step
        if t > 0:
            state2D = np.array([state[0], state[2]])
            error2D = np.array([[error[0,0], error[2,0]],[error[0,2],error[2,2]]])
            plotter.scatter(state2D[0], state2D[1], c='red', marker='.')
            plotter.add_artist(ph.get_cov_ellipse(error2D, state2D, 2, fill=False, linestyle='-', edgecolor="green"))
        return

class KalmanFilterServer(bnc.ServerBase):
    def __init__(self, name, 
                stateTransition, 
                stateErrorCov, 
                measureTransition,
                measurementErrorCov, 
                plotter, toLog=True):
        super().__init__(name, toLog)

        self.F = stateTransition
        self.Q = stateErrorCov
        self.H = measureTransition
        self.R = measurementErrorCov

        self.inits = []

        return
    
    def getDataToSendToClientFromSensorDataList(self, time, sensorDataList):
        

        return 1


class KalmanFilterClient(bnc.ClientBase):
    def processServerData(self, time, serverData):
        out = serverData
        return out
    
    def plotData(self, t, d, plotter):
        return
        if t > 1:
            plotter.scatter(d[0], d[2], c='g', marker='.')


# A single sensor kalman filter
def setupSim():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    q = 0.01 # Noise strength
    t = 0.5 # Time step
    xMeasureErrorStdDev = 0.5
    yMeasureErrorStdDev = 0.5
    stateTransition = np.array([[1, t, 0, 0],[0, 1, 0, 0],[0, 0, 1, t],[0, 0, 0, 1]])
    stateErrorMean = np.array([0,0,0,0])
    stateErrorCov = q*np.array([[t**3/3,t**2/2,0,0],[t**2/2,t,0,0],[0,0,t**3/3,t**2/2],[0,0,t**2/2,t]])
    measurementTransition = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
    measurementErrorMean = np.array([0,0])
    measurementErrorCov = np.array([[xMeasureErrorStdDev**2, 0], [0, yMeasureErrorStdDev**2]])

    initState = np.array([0,1,0,1])
    
    sensor = MovingObjectSmartSensor('sensor', 
                                    stateTransition, 
                                    stateErrorMean, 
                                    stateErrorCov, 
                                    measurementTransition, 
                                    measurementErrorMean, 
                                    measurementErrorCov,
                                    initState, 
                                    stateTransition, 
                                    stateErrorCov, 
                                    measurementTransition,  
                                    measurementErrorCov, 
                                    ax, True, True)
    sensors = [sensor]
    server = KalmanFilterServer('server', 
                                    stateTransition, 
                                    stateErrorCov, 
                                    measurementTransition, 
                                    measurementErrorCov,
                                    ax, True)
    client = KalmanFilterClient('client', ax, True, True)
    return sensors, server, client