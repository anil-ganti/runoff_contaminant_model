from numpy import *
from .common import *

class Simulation:
  def __init__(self, cfg, watershed):
    self.dx = cfg['dx']
    self.dt = cfg['dt']

    self.T = cfg['T']

    # Gaussian pulse as unit hydrograph function
    #self.fn_UHF = lambda t_: gaussian(t_, *sim.UHF_params_)
    #self.fn_IRF = lambda x_,t_: IRF(x_, t_, *sim.IRF_params_)

    self.L = sum([channel.L for channel in watershed.channel_])

    self.N = int(self.T / self.dt)
    self.K = int(self.L / self.dx)

    watershed.initialize(self)

  ### Solve coupled PDEs ###
  ##########################

  def solve_runoff(self, watershed, fn_UHF, fn_IRF, fn_RUN):
    '''
    Solve for Q(x,t) over the whole domain.
    Loop through each point in the domain
      propagate that point's flood wave downstream
      add that point's contribution to the total solution, Q(x,t)
    '''
    t_ = watershed.t_ # time domain is universal
    u_ = fn_UHF(t_) # unit hydrograph function is same everywhere
    for k,channel in enumerate(watershed.channel_):
      print("Solving channel %d" % k)
      x_ = channel.x_global_ # spatial domain in coordinate system of local channel
      i__ = fn_IRF(x_,t_)
      Q_idx_ = watershed.get_downstream_indices(channel)
      L = len(channel.x_global_)
      for i,w in enumerate(channel.x_local_):
        #print("Solving point %d of size %d" % (i,len(channel.x_local_)-i))
        r_ = fn_RUN(t_, *channel.runoff_params__[i]) # Compute run-off function
        h_ = convolve(r_, u_)[:len(t_)]
        for j,x in enumerate(channel.x_global_):
          q_ = convolve(h_, i__[:,j])[:len(t_)]
          watershed.Q__[:,Q_idx_[j]] = q_ + watershed.Q__[:,Q_idx_[j]]

class Watershed:
  '''
  Top-level object which defines the entire spatial domain. Consists of a tree of channels.
  Spatial domain in this object is flattened into a single column vector.
  Properties:
      Simulation properties:
          channel_ - list of 1D channels which comprise the watershed
          routing__ - Defines the routing topology through an adjacency matrix
          UHF_params_ - UHF function parameters
          IRF_params_ - IRF function parameters

      Variable fields:
          Q__ - 2D array of total discharge. dim 0 is time, dim 1 is space
          c__ - 2D array of concentration. dim 0 is time, dim 1 is space

      Function handles:
          fn_UHF - unit hydrograph function handle
          fn_IRF - fundamental solution to diffusive wave equation
          fn_runoff - run-off function handle, has parameters for every pixel
          fn_mass - contaminant release function handle, has parameters for every pixel
  '''

class Simple_Watershed(Watershed):
  def __init__(self, channel_, confluence_):
    self.channel_ = channel_
    self.confluence_ = confluence_

  @property
  def C(self):
    return len(self.channel_)

  @property
  def N(self):
    return self.Q__.shape[0]

  @property
  def K(self):
    return self.Q__.shape[1]

  @property
  def graph(self):
    a__ = zeros([self.K,self.K])
    ptr = 0
    # Connect individual channels
    for c in range(self.C):
      L_ch = self.channel_[c].L
      a__[ptr:ptr+L_ch,ptr:ptr+L_ch] = diag(ones(L_ch-1),k=1)
      ptr+= L_ch

    # Connect channels together
    for c1idx in range(self.C):
      c1 = self.channel_[c1idx]
      if sum(self.adjacency__[c1idx,:]) > 0:
        c2idx = argmax(self.adjacency__[c1idx,:])
        c2 = self.channel_[c2idx]
        trib_idx = self.get_channel_indices(c1)[-1]
        join_location = self.routing__[c1idx,c2idx]
        main_idx = self.get_channel_indices(c2)[int(join_location)]
        a__[trib_idx,main_idx] = 1
    return a__

  ### Initialization ###
  ######################
  def initialize(self,sim):
    '''
    Given a simulation, initialize the spatial, temporal domain, etc.
    '''

    self.dt = sim.dt
    self.dx = sim.dx

    self.initialize_routing_matrix()
    self.initialize_domain(sim)
    for i,channel in enumerate(self.channel_):
      L_ch = sum(self.routing__[i,:]) # calculate total downstream distance
      channel.initialize_domain(sim, L_ch)

    self.initialize_fields(sim)

  def initialize_routing_matrix(self):
    routing__ = zeros([self.C,self.C])
    adjacency__ = zeros([self.C,self.C])
    # set the main diagonal to the length of each channel
    for c in range(self.C):
      routing__[c,c] = self.channel_[c].L

    # set the entries for each confluence value
    for confluence in self.confluence_:
      # a confluence is just a tuple. (trib, main, upstream_position)
      trib_idx = confluence[0]
      main_idx = confluence[1]
      upstream_point = confluence[2]
      routing__[trib_idx,main_idx] = upstream_point
      adjacency__[trib_idx,main_idx] = 1

    # complete the tree by filling in the further downstream values
    for c1 in range(self.C):
      for c2 in range(self.C):
        if c1 == c2:
          continue
        if routing__[c1,c2] > 0:
          # look for any channels that c2 is connected to
          for c3 in range(self.C):
            if c3 == c1 or c3 == c2:
              continue
            if routing__[c2,c3] > 0:
              routing__[c1,c3] = routing__[c2,c3]

    self.routing__ = routing__
    self.adjacency__ = adjacency__

  def initialize_domain(self, sim):
    self.t_ = sim.dt*arange(sim.N) #

  def initialize_fields(self, sim):
    '''
    Initialize discharge and concentration fields to zero
    '''
    self.Q__ = zeros([sim.N,sim.K])
    self.c__ = zeros([sim.N,sim.K])

  ### Index manipulation ###
  ##########################

  def get_index(self, x):
    return int(x / self.dx)

  def get_downstream_channel_sequence(self, channel):
    '''
    Return a sequence of channels that are downstream of this channel
    '''
    seq_ = []
    ch_idx = self.channel_.index(channel)

    next_ch_idx = ch_idx
    while sum(self.adjacency__[next_ch_idx]) > 0:
      next_ch_idx = argmax(self.adjacency__[next_ch_idx,:])
      seq_.append(next_ch_idx)
    return asarray(seq_)

  def get_downstream_indices(self, channel, x=0):
    '''
    Returns the indices of the channel and all downstream points
    '''
    c1_idx = self.channel_.index(channel)
    idx_ = list(self.get_channel_indices(channel))
    for c2_idx in self.get_downstream_channel_sequence(channel):
      route_val = self.routing__[c1_idx, c2_idx]
      x = 0 if route_val == 1 else route_val # How far upstream does this trib come in?
      idx_ += list(self.get_channel_indices(self.channel_[c2_idx], x=x))
    return asarray(idx_)

  def get_channel_indices(self, channel, x=0):
    '''
    Return all global indices for a given channel at a point x upstream from it's outlet.
    Parameters:
        x - upstream point. defaults to zero which returns all channel indices
    '''
    ch_idx = self.channel_.index(channel)
    ptr = 0
    for i in range(ch_idx):
      ptr += self.channel_[i].K
    return ptr + arange(channel.K)[-self.get_index(x):]
