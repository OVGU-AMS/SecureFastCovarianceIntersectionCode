"""

"""

# Global imports
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# Local import of network to simulation
import FCI_fusion_network as ntwrk

# The number of time 'ticks' to run the simulation for, and whether to display logs and plots of the simulation
NUM_TIME_TICKS = 50
# Whether or not we wish to display plotting - note requires the setupSim function to have been setup accordingly
TO_PLOT = True

def runSim():
    """Based on the imported network file, create the correct sensor(s), server, and client, and then run the simulation."""

    groundTruth, sensors, server, client, finaliser = ntwrk.setupSim()

    # The simulation itself
    time = 0
    while time < NUM_TIME_TICKS:
        print ('Sim time = ' + str(time))
        trueState = next(groundTruth)
        # Readings are gotten by generating a reading from each of the sensors in the list created
        readings = [s.generateDataAndDisplay(time, trueState) for s in sensors]
        # Info to send to server (may include encrypting readings) is computed by each sensor for its generated reading
        infoToSendToServer = [s.getDataToSendToServerAndDisplay(time, p) for s,p in zip(sensors, readings)]
        # Info to send to client (may be processed sensor readings) is computed by the server
        infoToSendToClient = server.getDataToSendToClientFromSensorDataListAndDisplay(time, infoToSendToServer)
        # The client is then given the info computed for it by the server
        client.processServerDataAndDisplay(time, infoToSendToClient)

        # If plotting, this will display the point incrementally as they are computed
        if TO_PLOT:
            plt.pause(0.05)

        # Increase the time step and repeat the process
        time+=1
    
    finaliser.end_sim()

    # If plotting, keep the window open when simulation finishes
    if TO_PLOT:
        plt.show()

# Run the sim on file start
runSim()