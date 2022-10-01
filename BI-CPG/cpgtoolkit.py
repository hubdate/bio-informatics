import builtins, typing
import numpy



__all__ = ["ObjectConstructor_CentralPatternGenerator"]

class ObjectConstructor_CentralPatternGenerator(builtins.object):
    def __init__(self
                ,nn__neuronsNumber: int
                ,nscf__neuronsStepCycleFrequency: numpy.ndarray
                ,ncar__neuronsCriticalActuationAmplitudeRange: numpy.ndarray
                ,nao__neuronsAmplitudeOffset: numpy.ndarray
                ,*args: tuple
                ,ncw__neuronsCouplingWeights: typing.Optional[numpy.ndarray] =None
                ,npb__neuronsPhaseBiases: typing.Optional[numpy.ndarray] =None
                ,aac__amplitudeAccelerationCoefficient: typing.Optional[float] =20.
                ,oac__offsetAccelerationCoefficient: typing.Optional[float] =20.) -> None:
        """
        (method) : Initialize and configure the neural circuit.                         


        Parameters
        ----------
        nn__neuronsNumber: int
            The neural circuit size (number of neurons).

        nscf__neuronsStepCycleFrequency: numpy.ndarray
            An array of each neuron step cycle frequency (must be in Hertz). 
            Setting the frequency of each neuron to the same value, leads to a synchronous
            neural circuit, which means that each neuron will evolve in time at the same
            speed as the others.

        ncar__neuronsCriticalActuationAmplitudeRange: numpy.ndarray
            An array of each neuron actuation amplitude range (must be in radians).

            Explanation:
                If the systeme is a Servomotor with a π(rad) travel limit, the amplitude 
                actuation range might just be half the travel limit, π/2(rad). Since the 
                neuron act as an oscillator, the output values ∈ [-π/2(rad), π/2(rad)].

        nao__neuronsAmplitudeOffset: numpy.ndarray
            An array of each neuron amplitude offset (must be in radians).

            Explanation:
                If the systeme is a Servomotor with a π(rad) travel limit, and a π/2(rad) 
                amplitude actuation range. Since the neuron act as an oscillator, the 
                output values ∈ [-π/2(rad), π/2(rad)].

                In this example, as the Servo motor can't handle less than 0(rad) angles,
                the amplitude offset is build to avoid such outputs. Considering a 
                π/2(rad) amplitude offset, the neuron output values ∈ [0, π(rad)].

        *args: tuple
            Additional arguments should be passed as keyword arguments. 
            THEY ARE NOT HANDLE IN THIS METHOD.
        
        ncw__neuronsCouplingWeights: typing.Optional[numpy.ndarray], {default: None}
            An array representing the neurons coupling weights among themselves
            (neural circuit's synapses). 
            
            If not specified, each neuron will be coupled to the others 
            (coupling weight of 1) except itself (couplig wieght of 0).

        npb__neuronsPhaseBiases: typing.Optional[numpy.ndarray], {default: None}
            An array representing the neurons phases biases among themselves.

            If not specified, each neuron will have +2π/nneurons(rad) phase bias based on 
            the previous neuron phase bias.  

        aac__amplitudeAccelerationCoefficient: typing.Optional[float], {default: 20.}
            The amplitude acceleration coefficient, must always be greater than 0.

        oac__offsetAccelerationCoefficient: typing.Optional[float], {default: 20.}
            The offset acceleration coefficient, must always be greater than 0.


        Returns
        -------
        out: None
        """

        self.nn__neuronsNumber      =nn__neuronsNumber

        self.ncw__neuronsCouplingWeights    = ncw__neuronsCouplingWeights if isinstance(ncw__neuronsCouplingWeights, numpy.ndarray) else self.functions.st__fillWrappedDiagonal(numpy.ones([nn__neuronsNumber, nn__neuronsNumber]), 0)
        self.npb__neuronsPhaseBiases        = npb__neuronsPhaseBiases if isinstance(npb__neuronsPhaseBiases, numpy.ndarray) else numpy.array([     numpy.subtract(numpy.arange(0., 2*numpy.pi, 2*numpy.pi/self.nn__neuronsNumber), numpy.full(numpy.arange(0., 2*numpy.pi, 2*numpy.pi/self.nn__neuronsNumber).shape, neuron *(2*numpy.pi/self.nn__neuronsNumber)))     for neuron in range(self.nn__neuronsNumber)] )

        self.st_nscf__neuronsStepCycleFrequency                 = nscf__neuronsStepCycleFrequency
        self.st_nav__neuronsAngularVelocity                     = 2*numpy.pi *(nscf__neuronsStepCycleFrequency)
        self.st_ncaar__neuronsCriticalActuationAmplitudeRange   = ncar__neuronsCriticalActuationAmplitudeRange
        self.st_nao__neuronsAmplitudeOffset                     = nao__neuronsAmplitudeOffset

        self.nav__neuronsAngularVelocity     = numpy.zeros([self.nn__neuronsNumber, ])
        self.na__neuronsAmplitudes          = numpy.zeros([self.nn__neuronsNumber, ])
        self.noa__neuronsAmplitudeOffset    = numpy.zeros([self.nn__neuronsNumber, ])

        self.navd__neuronsAngularVelocity_DOT   = 0.
        self.nad__neuronsAmplitudes_DOT         = numpy.zeros([self.nn__neuronsNumber, ])
        self.naod__neuronsAmplitudeOffset_DOT   = numpy.zeros([self.nn__neuronsNumber, ])

        self.nad2__neuronsAmplitudes_DOT2           = 0.
        self.naod2__neuronsAmplitudeOffset_DOT2     = 0.

        self.aac__amplitudeAccelerationCoefficient  = aac__amplitudeAccelerationCoefficient
        self.oac__offsetAccelerationCoefficient     = oac__offsetAccelerationCoefficient

    
    def next(self, dt__deltaTime: float, *args: tuple) -> numpy.ndarray:
        """
        (method) : Compute the next the step of the neural circuit.

        
        Parameters
        ----------
        dt__deltaTime: float
            Delta time, wich represent the amount of time between calculations. Setting 
            it to a lower value can increase robustness and smoother of the solution, at
            the expense of a higher number of calculations steps.

        *args: tuple
            Additional arguments should be passed as keyword arguments. 
            THEY ARE NOT HANDLE IN THIS METHOD.


        Returns
        -------
        out:  numpy.ndarray
            An array of each neurons output value.
        """
        for neuron in range(self.nn__neuronsNumber):
            self.navd__neuronsAngularVelocity_DOT = self.st_nav__neuronsAngularVelocity[neuron]
            for ns__neuronSynapse in range(self.nn__neuronsNumber):
                self.navd__neuronsAngularVelocity_DOT += self.ncw__neuronsCouplingWeights[neuron][ns__neuronSynapse] *self.st_ncaar__neuronsCriticalActuationAmplitudeRange[ns__neuronSynapse] *numpy.sin( (self.nav__neuronsAngularVelocity[ns__neuronSynapse] -self.nav__neuronsAngularVelocity[neuron] -self.npb__neuronsPhaseBiases[neuron][ns__neuronSynapse]) )
            
            self.nav__neuronsAngularVelocity[neuron] = self.nav__neuronsAngularVelocity[neuron] +(self.navd__neuronsAngularVelocity_DOT *dt__deltaTime)


            self.nad2__neuronsAmplitudes_DOT2               =self.aac__amplitudeAccelerationCoefficient *( (self.aac__amplitudeAccelerationCoefficient/4) *(self.st_ncaar__neuronsCriticalActuationAmplitudeRange[neuron] -self.na__neuronsAmplitudes[neuron]) -self.nad__neuronsAmplitudes_DOT[neuron])
            self.nad__neuronsAmplitudes_DOT[neuron]         =self.nad__neuronsAmplitudes_DOT[neuron] +(self.nad2__neuronsAmplitudes_DOT2 *dt__deltaTime)
            self.na__neuronsAmplitudes[neuron]              =self.na__neuronsAmplitudes[neuron] +(self.nad__neuronsAmplitudes_DOT[neuron] *dt__deltaTime)

            self.naod2__neuronsAmplitudeOffset_DOT2         =self.oac__offsetAccelerationCoefficient *( (self.oac__offsetAccelerationCoefficient/4) *(self.st_nao__neuronsAmplitudeOffset[neuron] - self.noa__neuronsAmplitudeOffset[neuron]) -self.naod__neuronsAmplitudeOffset_DOT[neuron])
            self.naod__neuronsAmplitudeOffset_DOT[neuron]   =self.naod__neuronsAmplitudeOffset_DOT[neuron] +(self.naod2__neuronsAmplitudeOffset_DOT2 *dt__deltaTime)
            self.noa__neuronsAmplitudeOffset[neuron]        =self.noa__neuronsAmplitudeOffset[neuron] +(self.naod__neuronsAmplitudeOffset_DOT[neuron] *dt__deltaTime)

        return numpy.array([(self.noa__neuronsAmplitudeOffset[neuron] + self.na__neuronsAmplitudes[neuron] * numpy.sin(self.nav__neuronsAngularVelocity[neuron])) for neuron in range(self.nn__neuronsNumber)] )


    class functions(builtins.object):
        @staticmethod
        def st__fillWrappedDiagonal(array: numpy.ndarray, value: float) -> numpy.ndarray:
            """
            (method): Fill the main diagonal of the given array. For tall matrices, the diagonal
            "wrapped" when it reach the last column. This function does not modifies the input
            array in-place, it does return a value.
            Parameters
            ----------
            array: numpy.ndarray
                Array whose diagonal is to be filled.
            value: int 
                Value to write on the diagonal.
            Returns
            -------
            out: numpy.ndarray
                The result array.
            """
            for i in range(array.shape[1]):
                array[numpy.arange(array.shape[0]) % array.shape[1] == i, i] = value

            return array


if __name__ == "__main__":
    cpg = ObjectConstructor_CentralPatternGenerator(
         nn__neuronsNumber=2
        ,nscf__neuronsStepCycleFrequency=numpy.array((0.5, 0.5))
        ,ncar__neuronsCriticalActuationAmplitudeRange=numpy.array((0.4, 0.4))
        ,nao__neuronsAmplitudeOffset=numpy.array((0.+numpy.pi/2, 0.+numpy.pi/2))

    )






    import time
    current_time = time.time()
    dt = 1./500

    while True:
        if (time.time() - current_time > dt):
            tmp_dt = time.time() - current_time
            current_time = time.time()

