

class flightSoftware:

    _controller = None
    _estimator = None
    _referenceComputer = None

    def __init__(self, estimator, controller, referenceComp):
        self._controller = controller
        self._estimator = estimator
        self._referenceComputer = referenceComp
        return

    def getEstimator(self):
        return self._estimator

    def getController(self):
        return self._controller

    def getReferenceComputer(self):
        return self._referenceComputer

    def initFlightSoftware(self, t0):
        """
        Call this method to get the flight sofwtare started.
        :param t0: [double] Initial time.
        :return:
        """
        self._estimator.start(t0)
        self._referenceComputer.start(t0)
        self._controller.start(t0)
        return


    def runTask(self, t):
        """
        This method actually simulates a very simply task manager. The "run" methods called
        have their one time schedule.
        :param t: [double] Current time.
        :return:
        """
        self._estimator.runEstimation(t)
        self._referenceComputer.runReferenceComputing(t)
        self._controller.runControl(t)
        return

    def createControlForceHistory(self, number_of_elements):
        self._controller.createControlForceHistory(number_of_elements)
        return

    def getControlForceHistory(self):
        return self._controller.getControlForceHistory()

    def saveControlForces(self, element):
        """
        Once the state history is created, this method saves a state at a given time in a certain position of the state history.
        :param element:
        :return:
        """
        self._controller.saveControlForces(element)
        return
