from numpy import *
from numpy.linalg import inv
from copy import deepcopy

from .common import *

class Simulation:
  '''
  Wrapper simulation object used to store the simulation hyper-parameters and initialize the whole thing.
  This object is used to store properties we want to keep out of the objects which represent physical things.
  The split between what lives here and what lives in Watershed is not perfect or precise, just done by "feel".
  '''
  def __init__(self, cfg, watershed):
    self.dx = cfg['dx'] # spatial resolution
    self.dt = cfg['dt'] # temporal resolution

    self.T = cfg['T'] # Time domain over which to solve

    self.Q0 = cfg['Q0'] # Add baseflow to avoid div by zero. TODO: this should be cumulative!!!
    self.v0 = cfg['v0'] # Placeholder contaminant velocity. TODO: This should be linked to run-off field
    self.D_cont = cfg['D_cont'] # Contaminant diffusion coefficient

    self.L = sum([channel.L for channel in watershed.channel_]) # summed channel lengths

    self.N = int(self.T / self.dt) # Number of time points
    self.K = int(self.L / self.dx) # Number of spatial points

    watershed.initialize(self)

  @property
  def alpha(self):
    '''
    Finite differencing stability parameter. See write-up on finite differencing.
    '''
    return self.D_cont * self.dt / (self.dx**2)

  @property
  def beta(self):
    '''
    Implicit scheme parameter. See write-up on finite differencing.
    '''
    return self.dt / (2*self.dx)

  def solve_runoff(self, watershed, fn_UHF, fn_IRF, fn_INPUT):
    '''
    Solve for Q(x,t) over the whole domain.
    Loop through each point in the domain
      propagate that point's flood wave downstream
      add that point's contribution to the total solution, Q(x,t)
    '''
    t_ = watershed.t_ # time domain is universal
    p_ = fn_INPUT(t_) # input, ie. precipitation is same everywhere
    for k,channel in enumerate(watershed.channel_):
      print("Solving channel %d" % k)
      x_ = channel.x_global_ # spatial domain in coordinate system of local channel
      i__ = fn_IRF(x_,t_)
      Q_idx_ = watershed.get_downstream_indices(channel)
      L = len(channel.x_global_)
      for i,w in enumerate(channel.x_local_):
        #print("Solving point %d of size %d" % (i,len(channel.x_local_)-i))
        u_ = fn_UHF(t_, *channel.runoff_params__[i]) # Compute run-off function
        h_ = convolve(p_, u_)[:len(t_)]
        for j,x in enumerate(channel.x_global_):
          q_ = convolve(h_, i__[:,j])[:len(t_)]
          watershed.Q__[:,Q_idx_[j]] = q_ + watershed.Q__[:,Q_idx_[j]]
    watershed.Q__ += self.Q0 # add baseflow

  def calc_B(self, watershed, L):
    '''
    Compute the simultaneous equation matrix B. See finite differencing write-up.
    '''
    B__ = zeros([L,L])
    v = self.v0
    fill_diagonal(B__, 1+2*self.alpha)
    for i in range(L-1):
      B__[i,i+1] = -(self.beta*v + self.alpha)
      B__[i+1,i] = -(self.beta*v + self.alpha)
    #B__[0,0] = 1
    #B__[-1,-1] = 1
    B__[0,1] = 0
    B__[-1,-2] = 0
    return B__

  def solve_contaminant(self, watershed, fn_MASS):
    '''
    Solve for c(x,t) over the whole domain.
    First must compute s(x,t) for each point in the domain.
    Each spatial point in the domain has associated mass transfer parameters.
    Mass m(x,t) must be converted to a source concentration s(x,t).
    We use functional form of m(x,t) to compute s(x,t) based on the current flow.
    The total field, c(x,t) can then be computed as the sum of contributions of each s(x,t)
    '''
    t_ = watershed.t_
    for k,channel in enumerate(watershed.channel_):
      print("solving channel %d" % k)
      idx_ = watershed.get_channel_indices(channel)
      c_idx_ = watershed.get_downstream_indices(channel)
      x_ = channel.x_global_
      Lws = len(x_) # length of the total domain
      Lch = len(idx_)
      B__ = self.calc_B(watershed, Lws)
      Binv__ = inv(B__)
      c_ = zeros([Lws,1])
      c_np1_ = zeros([Lws,1])
      for n,t in enumerate(t_[:-1]):
        m_ = asarray(list(map(
          lambda p_: fn_MASS(t, *p_),channel.contaminant_params__)))
        Q_k_ = watershed.Q__[n,c_idx_]
        s_ = zeros(Lws)
        s_[:Lch] = divide(m_, Q_k_[:Lch])

        ### Find flow into the current pixel to determine concentration ###
        Q_km1_ = zeros(Lws)
        Q_km1_[1:] = Q_k_[:1]
        c_km1_ = zeros(Lws)
        c_km1_[1:] = c_[:1]

        ### Compute source concentration field based on current flow ###
        s_ = s_ + multiply(c_km1_,(divide(Q_km1_,Q_k_)-1))
        s_ = s_.reshape(Lws,1)

        ### Add source to current concentration, propagate forward ###
        c_ = c_ + s_
        c_np1_ = mdot(Binv__,c_)

        # add to global solution
        watershed.c__[n+1,c_idx_] = watershed.c__[n+1,c_idx_] + c_np1_.flatten()
        c_ = deepcopy(c_np1_)

class Watershed:
  '''
  Top-level object which defines the entire spatial domain. Consists of a tree of channels.
  Spatial domain in this object is flattened into a single column vector.

  Properties:
      Simulation properties:
          channel_ - list of 1D channels which comprise the watershed
          routing__ - Defines the routing topology through an adjacency matrix
          UHF_params_ - Unit hydrograph function parameters
          IRF_params_ - Impulse response function parameters

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
    '''
    Returns the number of channels in the watershed
    '''
    return len(self.channel_)

  @property
  def N(self):
    '''
    Returns the number of time indices
    '''
    return self.Q__.shape[0]

  @property
  def K(self):
    '''
    Returns the number of spatial indices
    '''
    return self.Q__.shape[1]

  @property
  def graph(self):
    '''
    Compute the watershed graph. This provide a binary mask indicating where the spatial domains of two channels intersect.
    '''
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
    '''
    Initialize the graphs which represent connections between channels.
    This will be used to join the spatial domains of each channel based on routing topology.
    '''
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
    '''
    Initialize the time domain variable t_.
    '''
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
    '''
    Get spatial index of a point relative to a channel outlet.
    '''
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

  ### Watershed visualization ###
  def channel_xy_points(self, ch_idx, xf, yf, th):
    '''
    "Render" the channel as a series of points in cartesian coordinates.
    '''
    x_local_ = -1*self.channel_[ch_idx].x_local_
    x_ = flip(xf + cos(radians(th))*x_local_)
    y_ = flip(yf + sin(radians(th))*x_local_)
    return vstack((x_,y_)).transpose() # L x 2

  def watershed_xy_points(self, _th):
    '''
    "Render" the whole watershed as a series of points in cartesian coordinates.
    TODO: Should replace this code with a standard library for visualizing trees
    Generate a visualization of the watershed as a collection of x,y points
    Parameters:
        _th: a dictionary whose keys are the channel, and value is the angle at which to join
    '''
    outlet_pos__ = empty([self.C,3])
    xy__ = []
    for i,ch in enumerate(self.channel_):
      seq_ = self.get_downstream_channel_sequence(ch)
      if len(seq_) == 0:
        xf_ = asarray([0,0,0]) # set main channel to flow into 0,0
      else:
        c2idx = seq_[0]
        L = self.routing__[i,c2idx] # how far upstream to join?
        c2pos_ = outlet_pos__[c2idx]
        xf_ = deepcopy(c2pos_)
        xf_[2] += _th[self.channel_.index(ch)]
        xf_[:2] -= asarray([L*cos(radians(c2pos_[2])),L*sin(radians(c2pos_[2]))])
      xy__.append(self.channel_xy_points(i,xf_[0],xf_[1],xf_[2]))
      outlet_pos__[i] = xf_
    xy__ = vstack(tuple(xy__))
    return xy__
