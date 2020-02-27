"""

"""
import pickle as pkl
import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import network_helpers.base_network_classes as bnc
import other_helpers.plotting_helper as ph
import other_helpers.encryption_simulation_classes as enc
import covar_int_computation as fci

# Horrid global shortcut to make post plotting doable quickly
# TODO remove this shit and implement properly
saved_sim_output = {'ground_truth': [],
                    'measurements': {},
                    'sensor_estimates': {},
                    'fusion_estimates': [],
                    'secure_fusion_estimates': []}

"""
 .d8888b.
d88P  Y88b
Y88b.
 "Y888b.    .d88b.  88888b.  .d8888b   .d88b.  888d888
    "Y88b. d8P  Y8b 888 "88b 88K      d88""88b 888P"
      "888 88888888 888  888 "Y8888b. 888  888 888
Y88b  d88P Y8b.     888  888      X88 Y88..88P 888
 "Y8888P"   "Y8888  888  888  88888P'  "Y88P"  888



"""
class MovingObjectSmartSensor(bnc.SensorBase):
    curId = 0
    def __init__(self, name, 
                trueMeasurementTransition,
                trueMeasurementErrorMean,
                trueMeasurementErrorCov,
                stateTransition,
                stateErrorCov,
                measurementTransition,
                measurementErrorCov, 
                initState,
                initErrorCov,
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
        self.x = initState
        self.P = initErrorCov
        self.inits = []

        # Used for fusion approximation
        self.omegaStep = omegaStep
        self.id = MovingObjectSmartSensor.curId
        MovingObjectSmartSensor.curId+=1

        # Used for plotting colour
        offset = 1 # Simple way to affect the chosen colours a bit
        self.colourId = list(mcolors.TABLEAU_COLORS.values())[self.id+offset%len(mcolors.TABLEAU_COLORS)]
        
        saved_sim_output['sensor_estimates'][name] = []
        saved_sim_output['measurements'][name] = []

        return

    def generateData(self, t, groundTruth):
        # ===== Save the ground truth value
        self.truePrevState = groundTruth
        
        # Measure the true state according to model
        measureError = np.random.multivariate_normal(self.trueMeasurementErrorMean, self.trueMeasurementErrorCov)
        measurement = self.trueMeasurementTransition@self.truePrevState + measureError
        saved_sim_output['measurements'][self._name].append(measurement)

        # ===== Filter true measurement
        # Special case at the start, compute initial from first 2 measurements
        # if t == 0:
        #     self.inits.append(measurement)
        #     return self.truePrevState, measurement, None, None
        # if t == 1:
        #     self.inits.append(measurement)
        #     # Now that we have the first 2 measurements, we can create the believed initial state
        #     # Make the initial x, y position be the second measurement that was gotten
        #     # Make the initial x, y velocity be computed from the distance between the first and second measurements that were gotten
        #     self.x = np.array([self.inits[1][0], (self.inits[1][0] - self.inits[0][0])/self.F[0][1], self.inits[1][1], (self.inits[1][1] - self.inits[0][1])/self.F[0][1]])
        #     self.P = self.Q
        #     return self.truePrevState, measurement, self.x, self.P
        
        # TODO would be faster for encryption is the following with done in the information filter form, with the appropriate variables saved
        # Prediction
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)

        # Update
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)

        saved_sim_output['sensor_estimates'][self._name].append((self.x, self.P))
        return self.truePrevState, measurement, self.x, self.P
    
    def getDataToSendToServer(self, t, p):
        # Nothing to process on the first time step
        if t==0:
            return None, None, None, None, None

        trace = np.trace(self.P)
        oreTraces = []
        # TODO don't really have to send first discretisation (om=0) since it's always the same
        for om in np.arange(0, 1+self.omegaStep, self.omegaStep):
            oreTraces.append(enc.ORE_Number(om*trace, 'L' if self.id%2==0 else 'R'))
        
        Pinv = np.linalg.inv(self.P)
        Pinvx = Pinv@self.x

        phePinv = np.array([[enc.Add_PHE_Number(val) for val in r] for r in Pinv])
        phePinvx = np.array([enc.Add_PHE_Number(val) for val in Pinvx])
        
        # Send both plaintext and encryptions for simulation comparison
        return self.x, self.P, phePinvx, phePinv, oreTraces
    
    def plotData(self, t, d, plotter):
        true, measurement, state, error = d

        # Only the first sensor should plot the ground truth
        if self._name == 0:
            true2D = np.array([true[0], true[2]])
            plotter.scatter(true2D[0], true2D[1], c='lightgrey', marker='.')
        
        # Both should plot their measurement
        plotter.scatter(measurement[0], measurement[1], c=self.colourId, marker='x')

        # State and error estimates only made after first time step
        if t > 0:
            state2D = np.array([state[0], state[2]])
            error2D = np.array([[error[0,0], error[2,0]],[error[0,2],error[2,2]]])
            plotter.scatter(state2D[0], state2D[1], c=self.colourId, marker='.')
            plotter.add_artist(ph.get_cov_ellipse(error2D, state2D, 2, fill=False, linestyle='-', edgecolor=self.colourId))
        return

"""
 .d8888b.
d88P  Y88b
Y88b.
 "Y888b.    .d88b.  888d888 888  888  .d88b.  888d888
    "Y88b. d8P  Y8b 888P"   888  888 d8P  Y8b 888P"
      "888 88888888 888     Y88  88P 88888888 888
Y88b  d88P Y8b.     888      Y8bd8P  Y8b.     888
 "Y8888P"   "Y8888  888       Y88P    "Y8888  888



"""
class FCIFusionServer(bnc.ServerBase):
    def __init__(self, name, 
                omegaStep, 
                plotter, toLog=True):
        super().__init__(name, toLog)
        self.omegaStep = omegaStep
        return
    
    def getDataToSendToClientFromSensorDataList(self, time, sensorDataList):
        # No estimate on first time step
        if time == 0:
            return None, None, None, None
        
        # Compute the exact fast covariance intersection
        traces = []
        for _,P,_,_,_ in sensorDataList:
            traces.append(np.trace(P))
        omegas = fci.omega_exact(traces)
        Pinv_fused = sum([np.linalg.inv(sensorDataList[i][1])*omegas[i] for i in range(len(sensorDataList))])
        Pinvx_fused = sum([np.linalg.inv(sensorDataList[i][1])@sensorDataList[i][0]*omegas[i] for i in range(len(sensorDataList))])
        P_fused = np.linalg.inv(Pinv_fused)
        x_fused = P_fused@Pinvx_fused

        # Compute the approximate secure fast covariance intersection
        sensorLists = []
        for _,_,_,_,oreList in sensorDataList:
            sensorLists.append(oreList)
        approxOmegas = fci.omega_estimates(sensorLists, self.omegaStep)
        Pinv_fused_approx = sum([sensorDataList[i][3]*approxOmegas[i] for i in range(len(sensorDataList))])
        Pinvx_fused_approx = sum([sensorDataList[i][2]*approxOmegas[i] for i in range(len(sensorDataList))])

        print('Exact solution: ', ['%1.4f'%i for i in omegas])
        print('Approx solution:', ['%1.4f'%i for i in approxOmegas])

        return x_fused, P_fused, Pinvx_fused_approx, Pinv_fused_approx

"""
 .d8888b.  888 d8b                   888
d88P  Y88b 888 Y8P                   888
888    888 888                       888
888        888 888  .d88b.  88888b.  888888
888        888 888 d8P  Y8b 888 "88b 888
888    888 888 888 88888888 888  888 888
Y88b  d88P 888 888 Y8b.     888  888 Y88b.
 "Y8888P"  888 888  "Y8888  888  888  "Y888



"""
class FusionQueryClient(bnc.ClientBase):
    def processServerData(self, time, serverData):
        if time==0:
            return None, None, None, None
        x, P, Pinvx, Pinv = serverData
        approxP = np.linalg.inv(np.array([[val.get_number() for val in r] for r in Pinv]))
        approxX = approxP@np.array([val.get_number() for val in Pinvx])

        saved_sim_output['fusion_estimates'].append((x, P))
        saved_sim_output['secure_fusion_estimates'].append((approxX, approxP))
        return x, P, approxX, approxP
    
    def plotData(self, t, d, plotter):
        state, error, approxState, approxError = d
        trueColour = mcolors.CSS4_COLORS['cyan']
        approxColour = mcolors.CSS4_COLORS['darkcyan']
        if t > 1:
            # Normal fast covariance intersection plot
            state2D = np.array([state[0], state[2]])
            error2D = np.array([[error[0,0], error[2,0]],[error[0,2],error[2,2]]])
            plotter.scatter(state2D[0], state2D[1], c=trueColour, marker='.')
            plotter.add_artist(ph.get_cov_ellipse(error2D, state2D, 2, fill=False, linestyle='-', edgecolor=trueColour))

            # Secure fast covariance plot
            approxState2D = np.array([approxState[0], approxState[2]])
            approxError2D = np.array([[approxError[0,0], approxError[2,0]],[approxError[0,2],approxError[2,2]]])
            plotter.scatter(approxState2D[0], approxState2D[1], c=approxColour, marker='.')
            plotter.add_artist(ph.get_cov_ellipse(approxError2D, approxState2D, 2, fill=False, linestyle='-', edgecolor=approxColour))
        return

"""
 .d8888b.     88888888888
d88P  Y88b        888
888    888        888
888               888
888  88888        888
888    888        888
Y88b  d88P d8b    888  d8b
 "Y8888P88 Y8P    888  Y8P



"""
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
        saved_sim_output['ground_truth'].append(self.truePrevState)
        return self.truePrevState


"""
8888888888 d8b
888        Y8P
888
8888888    888 88888b.
888        888 888 "88b
888        888 888  888
888        888 888  888 d8b
888        888 888  888 Y8P



"""
class Finaliser:
    def __init__(self):
        return
    
    def end_sim(self):
        pkl.dump(saved_sim_output, open( "simout.p", "wb" ))
        return




"""
 .d8888b.  d8b                         .d8888b.           888
d88P  Y88b Y8P                        d88P  Y88b          888
Y88b.                                 Y88b.               888
 "Y888b.   888 88888b.d88b.            "Y888b.    .d88b.  888888 888  888 88888b.
    "Y88b. 888 888 "888 "88b              "Y88b. d8P  Y8b 888    888  888 888 "88b
      "888 888 888  888  888                "888 88888888 888    888  888 888  888
Y88b  d88P 888 888  888  888 d8b      Y88b  d88P Y8b.     Y88b.  Y88b 888 888 d88P
 "Y8888P"  888 888  888  888 Y8P       "Y8888P"   "Y8888   "Y888  "Y88888 88888P"
                                                                          888
                                                                          888
                                                                          888
"""
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
    initState = np.array([0,0.5,0,0.5])

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

        sensorInitState = initState+np.random.rand(4)
        sensorInitErrorCov = stateTransition*(1+0.2*np.random.rand(4,4))
        
        sensor = MovingObjectSmartSensor(i,
                                        measurementTransition, 
                                        measurementErrorMean, 
                                        measurementErrorCov, 
                                        stateTransition, 
                                        stateErrorCov, 
                                        measurementTransition, 
                                        measurementErrorCov, 
                                        sensorInitState,
                                        sensorInitErrorCov,
                                        omegaStep, 
                                        ax, False, True)
        sensors.append(sensor)
    
    # Define fusion server
    server = FCIFusionServer('server', 
                             omegaStep, 
                             ax, False)
    
    # Define query client
    client = FusionQueryClient('client', ax, False, True)

    return iter(groundTruth), sensors, server, client, Finaliser()