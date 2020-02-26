"""

"""

import functools as ft
import numpy as np
import matplotlib.pyplot as plt

import network_helpers.base_network_classes as bnc
import other_helpers.plotting_helper as ph
import other_helpers.encryption_simulation_classes as enc
import covar_int_computation as fci

class MovingObjectSmartSensor(bnc.SensorBase):
    def __init__(self, name, 
                trueMeasurementTransition,
                trueMeasurementErrorMean,
                trueMeasurementErrorCov,
                stateTransition,
                stateErrorCov,
                measurementTransition,
                measurementErrorCov, 
                omegaStep, 
                plotter, toLog=True, toPlot=True):
        super().__init__(name, plotter, toLog, toPlot)

        # Used for true trajectory and measurments
        self.trueMeasurementTransition = trueMeasurementTransition
        self.trueMeasurementErrorMean = trueMeasurementErrorMean
        self.trueMeasurementErrorCov = trueMeasurementErrorCov
        # Current true initial position variable
        self.truePrevState = None

        # Used for filter
        self.F = stateTransition
        self.Q = stateErrorCov
        self.H = measurementTransition
        self.R = measurementErrorCov
        self.x = None
        self.P = None
        self.inits = []

        # Used for fusion approximation
        self.omegaStep = omegaStep
        return

    def generateData(self, t, groundTruth):
        # ===== Save the ground truth value
        self.truePrevState = groundTruth
        
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
        colour_to_plot = 'C'+str(self._name)

        # Only the first sensor should plot the ground truth
        if self._name == 0:
            true2D = np.array([true[0], true[2]])
            plotter.scatter(true2D[0], true2D[1], c='lightgrey', marker='.')
        
        # Both should plot their measurement
        plotter.scatter(measurement[0], measurement[1], c=colour_to_plot, marker='x')

        # State and error estimates only made after first time step
        if t > 0:
            state2D = np.array([state[0], state[2]])
            error2D = np.array([[error[0,0], error[2,0]],[error[0,2],error[2,2]]])
            plotter.scatter(state2D[0], state2D[1], c=colour_to_plot, marker='.')
            plotter.add_artist(ph.get_cov_ellipse(error2D, state2D, 2, fill=False, linestyle='-', edgecolor=colour_to_plot))
        return

class FCIFusionServer(bnc.ServerBase):
    def __init__(self, name, 
                omegaStep, 
                plotter, toLog=True):
        super().__init__(name, toLog)
        self.omegaStep = omegaStep
        return
    
    def getDataToSendToClientFromSensorDataList(self, time, sensorDataList):
        if time == 0:
            return None, None
        
        traces = []
        for _,P in sensorDataList:
            traces.append(np.trace(P))

        omegas = fci.omega_exact(traces)

        Pinv_fused = sum([np.linalg.inv(sensorDataList[i][1])*omegas[i] for i in range(len(sensorDataList))])
        Pinvx_fused = sum([np.linalg.inv(sensorDataList[i][1])@sensorDataList[i][0]*omegas[i] for i in range(len(sensorDataList))])
        P_fused = np.linalg.inv(Pinv_fused)
        x_fused = P_fused@Pinvx_fused
        return x_fused, P_fused


class FusionQueryClient(bnc.ClientBase):
    def processServerData(self, time, serverData):
        x, P = serverData
        return x, P
    
    def plotData(self, t, d, plotter):
        state, error = d
        if t > 1:
            state2D = np.array([state[0], state[2]])
            error2D = np.array([[error[0,0], error[2,0]],[error[0,2],error[2,2]]])
            plotter.scatter(state2D[0], state2D[1], c='green', marker='.')
            plotter.add_artist(ph.get_cov_ellipse(error2D, state2D, 2, fill=False, linestyle='-', edgecolor="green"))
        return

# An iterator for the ground truth
class GroundTruth:
    def __init__(self,
                 trueStateTransition, 
                 trueStateErrorMean, 
                 trueStateErrorCov, 
                 trueInitState):
        # Used for true trajectory and measurments
        self.trueStateTransition = trueStateTransition
        self.trueStateErrorMean = trueStateErrorMean
        self.trueStateErrorCov = trueStateErrorCov
        # The true initial position
        self.truePrevState = trueInitState
        self.time = 0
        return
    def __iter__(self):
        return self

    def __next__(self):
        if self.time > 0:
            signalError = np.random.multivariate_normal(self.trueStateErrorMean, self.trueStateErrorCov)
            self.truePrevState = self.trueStateTransition@self.truePrevState + signalError
        self.time+=1
        return self.truePrevState


# A multi sensor fusion server network
def setupSim():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Process parameters
    q = 0.02 # Noise strength
    t = 0.5 # Time step
    stateTransition = np.array([[1, t, 0, 0],[0, 1, 0, 0],[0, 0, 1, t],[0, 0, 0, 1]])
    stateErrorMean = np.array([0,0,0,0])
    stateErrorCov = q*np.array([[t**3/3,t**2/2,0,0],[t**2/2,t,0,0],[0,0,t**3/3,t**2/2],[0,0,t**2/2,t]])
    initState = np.array([0,1,0,1])

    # Ground truth iterator
    groundTruth = GroundTruth(stateTransition, 
                              stateErrorMean, 
                              stateErrorCov,
                              initState)
    
    # Discretisation for fusion approximation
    omegaStep = 0.1

    # Define sensors
    num_sensors = 2
    sensors = []
    for i in range(num_sensors):
        # Have different measurement parameters for different sensors
        xMeasureErrorStdDev = 0.5 + 0.5*np.random.rand()
        yMeasureErrorStdDev = 0.5 + 0.5*np.random.rand()
        measurementTransition = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        measurementErrorMean = np.array([0,0])
        measurementErrorCov = np.array([[xMeasureErrorStdDev**2, 0], [0, yMeasureErrorStdDev**2]])
        
        sensor = MovingObjectSmartSensor(i,
                                        measurementTransition, 
                                        measurementErrorMean, 
                                        measurementErrorCov, 
                                        stateTransition, 
                                        stateErrorCov, 
                                        measurementTransition, 
                                        measurementErrorCov, 
                                        omegaStep, 
                                        ax, True, True)
        sensors.append(sensor)
    
    # Define fusion server
    server = FCIFusionServer('server', 
                             omegaStep, 
                             ax, True)
    
    # Define query client
    client = FusionQueryClient('client', ax, True, True)

    return iter(groundTruth), sensors, server, client