import builtins, typing
import numpy



__all__ = ["ObjectConstructor_CentralPatternGenerator"]

class ObjectConstructor_2SHOFA(builtins.object):
    def __init__(self
                ,nscf__neuronStepCycleFrequency: float
                ,ncaar__neuronCriticalActuationAmplitudeRange: float
                ,ncs__neuronConvergenceSpeed: float
                ,nao__neuronAmplitudeOffset: float) -> None:
        
        self.nscf__neuronStepCycleFrequency                 = nscf__neuronStepCycleFrequency
        self.ncaar__neuronCriticalActuationAmplitudeRange   = ncaar__neuronCriticalActuationAmplitudeRange**2
        self.ncs__neuronConvergenceSpeed                    = ncs__neuronConvergenceSpeed
        self.nao__neuronAmplitudeOffset                     = nao__neuronAmplitudeOffset
        

        self.eno__excitatoryNeuronOutput        = 1.
        self.ino__inhibitoryNeuronOutput        = 0.

        self.enod__excitatoryNeuronOutput_DOT   = 0.
        self.inod__inhibitoryNeuronOutput_DOT   = 0.


    def next(self, dt: float, enef__excitatoryNeuronExternalFeedback: float, inef__inhibitoryNeuronExternalFeedback: float):
        """
        # The Runge-Kutta method of order 1 (RK1)
        self.enod__excitatoryNeuronOutput_DOT = -self.nscf__neuronStepCycleFrequency * self.ino__inhibitoryNeuronOutput + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - (self.eno__excitatoryNeuronOutput**2 + self.ino__inhibitoryNeuronOutput**2) )* self.eno__excitatoryNeuronOutput + enef__excitatoryNeuronExternalFeedback
        self.inod__inhibitoryNeuronOutput_DOT =  self.nscf__neuronStepCycleFrequency * self.eno__excitatoryNeuronOutput + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - (self.eno__excitatoryNeuronOutput**2 + self.ino__inhibitoryNeuronOutput**2) )* self.ino__inhibitoryNeuronOutput + inef__inhibitoryNeuronExternalFeedback

        self.eno__excitatoryNeuronOutput = self.eno__excitatoryNeuronOutput + self.enod__excitatoryNeuronOutput_DOT *dt
        self.ino__inhibitoryNeuronOutput = self.ino__inhibitoryNeuronOutput + self.inod__inhibitoryNeuronOutput_DOT *dt

    
        return (
             self.eno__excitatoryNeuronOutput + self.nao__neuronAmplitudeOffset
            ,self.ino__inhibitoryNeuronOutput + self.nao__neuronAmplitudeOffset
        )
        """
        self.enod__excitatoryNeuronOutput_DOT_RK4_K1 = -self.nscf__neuronStepCycleFrequency * self.ino__inhibitoryNeuronOutput + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - (self.eno__excitatoryNeuronOutput**2 + self.ino__inhibitoryNeuronOutput**2) )* self.eno__excitatoryNeuronOutput + enef__excitatoryNeuronExternalFeedback
        self.inod__inhibitoryNeuronOutput_DOT_RK4_K1 =  self.nscf__neuronStepCycleFrequency * self.eno__excitatoryNeuronOutput + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - (self.eno__excitatoryNeuronOutput**2 + self.ino__inhibitoryNeuronOutput**2) )* self.ino__inhibitoryNeuronOutput + inef__inhibitoryNeuronExternalFeedback
        

        self.enod__excitatoryNeuronOutput_DOT_RK4_K2 = -self.nscf__neuronStepCycleFrequency * (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K1) + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - ((self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K1)**2 + (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K1)**2) )* (self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K1) + enef__excitatoryNeuronExternalFeedback
        self.inod__inhibitoryNeuronOutput_DOT_RK4_K2 =  self.nscf__neuronStepCycleFrequency * (self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K1) + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - ((self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K1)**2 + (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K1)**2) )* (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K1) + inef__inhibitoryNeuronExternalFeedback


        #self.enod__excitatoryNeuronOutput_DOT_RK4_K3 = -self.nscf__neuronStepCycleFrequency * (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K2) + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - ((self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K2)**2 + (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K2)**2) )* (self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K2) + enef__excitatoryNeuronExternalFeedback
        #self.inod__inhibitoryNeuronOutput_DOT_RK4_K3 =  self.nscf__neuronStepCycleFrequency * (self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K2) + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - ((self.eno__excitatoryNeuronOutput + (dt/2)*self.enod__excitatoryNeuronOutput_DOT_RK4_K2)**2 + (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K2)**2) )* (self.ino__inhibitoryNeuronOutput + (dt/2)*self.inod__inhibitoryNeuronOutput_DOT_RK4_K2) + inef__inhibitoryNeuronExternalFeedback


        #self.enod__excitatoryNeuronOutput_DOT_RK4_K4 = -self.nscf__neuronStepCycleFrequency * (self.ino__inhibitoryNeuronOutput + dt*self.inod__inhibitoryNeuronOutput_DOT_RK4_K3) + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - ((self.eno__excitatoryNeuronOutput + dt*self.enod__excitatoryNeuronOutput_DOT_RK4_K3)**2 + (self.ino__inhibitoryNeuronOutput + dt*self.inod__inhibitoryNeuronOutput_DOT_RK4_K3)**2) )* (self.eno__excitatoryNeuronOutput + dt*self.enod__excitatoryNeuronOutput_DOT_RK4_K3) + enef__excitatoryNeuronExternalFeedback
        #self.inod__inhibitoryNeuronOutput_DOT_RK4_K4 =  self.nscf__neuronStepCycleFrequency * (self.eno__excitatoryNeuronOutput + dt*self.enod__excitatoryNeuronOutput_DOT_RK4_K3) + self.ncs__neuronConvergenceSpeed* (self.ncaar__neuronCriticalActuationAmplitudeRange - ((self.eno__excitatoryNeuronOutput + dt*self.enod__excitatoryNeuronOutput_DOT_RK4_K3)**2 + (self.ino__inhibitoryNeuronOutput + dt*self.inod__inhibitoryNeuronOutput_DOT_RK4_K3)**2) )* (self.ino__inhibitoryNeuronOutput + dt*self.inod__inhibitoryNeuronOutput_DOT_RK4_K3) + inef__inhibitoryNeuronExternalFeedback


        self.eno__excitatoryNeuronOutput = self.eno__excitatoryNeuronOutput + dt* (self.enod__excitatoryNeuronOutput_DOT_RK4_K2)
        self.ino__inhibitoryNeuronOutput = self.ino__inhibitoryNeuronOutput + dt* (self.inod__inhibitoryNeuronOutput_DOT_RK4_K2)

        return (
             self.eno__excitatoryNeuronOutput
            ,self.ino__inhibitoryNeuronOutput
        )


if __name__ == "__main__":
    _2shofa = ObjectConstructor_2SHOFA(0.5 *(2*numpy.pi), 1., 2. , 0.)
    dt = 1./500
    """
    import time
    current_time = time.time()
    

    while True:
        if (time.time() - current_time > dt):
            tmp_dt = time.time() - current_time
            print(_2shofa.next(tmp_dt, 0., 0.))
            current_time = time.time()
    """

    import matplotlib.pyplot

    matplotlib.pyplot.plot(numpy.arange(0., (1./0.5) +dt, dt), [_2shofa.next(dt, 0., 0.) for t in numpy.arange(0., (1./0.5) +dt, dt)])
    matplotlib.pyplot.show()