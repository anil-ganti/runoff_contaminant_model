from numpy import *

class Channel:
    '''
    1D channel object.

    Properties:
        L - length of channel
        x_ - discretized local spatial coordinate
        runoff_params__ - 2D array of the run-off parameters for each point
        contaminant_params__ - 2D array of contaminant parameters for each point
    '''

class Simple_Channel(Channel):
  '''
  Channel object which contains run-off and contaminant parameters.
  '''
  def __init__(self, L, x_, runoff_params__, contaminant_params__):
    self.L = L # total channel length
    self.x_local_ = x_
    self.runoff_params__ = runoff_params__
    self.contaminant_params__ = contaminant_params__

  @property
  def K(self):
    '''
    Returns length of the channel domain
    '''
    return len(self.x_local_)

  def initialize_domain(self, sim, L):
    self.x_global_ = sim.dx*arange(int(L / sim.dx))
