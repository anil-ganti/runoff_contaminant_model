from numpy import *

import runoff_contaminant_model.common as cmn
import runoff_contaminant_model.watershed as ws
import runoff_contaminant_model.channel as ch
from runoff_contaminant_model.common import *

cfg = {'dx':1, 'dt':1, 'T':10, 'Q0':1, 'v0':1,'D_cont':1}

L_ = [10,10,10,5]
uhf_params_ = asarray([1,1]) #gaussian location, width
uhf_params__ = kron(ones(10).reshape([10,1]),uhf_params_)
mass_params_ = asarray([0,0,0])
mass_params__ = kron(ones(10).reshape([10,1]),mass_params_)
mass_params__[0] = asarray([10,0,0.1])

#ch_ = list(map(
#  lambda L: ch.Simple_Channel(L,arange(L),uhf_params__,mass_params__[:int(L_[c]/cfg['dx'])]),L_))

ch_ = []

for c in range(len(L_)):
  ch_.append(
    ch.Simple_Channel(
      L_[c],cfg['dx']*arange(L_[c]),
      uhf_params__,
      mass_params__[:int(L_[c]/cfg['dx'])]))

confluence_ = [(1,0,7),(2,1,7),(3,2,3)]

watershed = ws.Simple_Watershed(ch_,confluence_)
sim = ws.Simulation(cfg,watershed)

def test_watershed_indexing():
  print(watershed.routing__[0,1])

  print("Routing matrix:")
  print(watershed.routing__)

  for c in range(watershed.C):
    ch_idx_ = watershed.get_channel_indices(watershed.channel_[c])
    print(ch_idx_)
    assert((ch_idx_ == sum(L_[:c]) + arange(L_[c])).all())

  for c in range(watershed.C):
    ds_idx_ = watershed.get_downstream_indices(watershed.channel_[c])
    print(ds_idx_)

  for c in range(watershed.C):
    ds_ch_ = watershed.get_downstream_channel_sequence(watershed.channel_[c])
    print(ds_ch_)

def test_watershed_graph():
  print("Testing graph")
  a__ = watershed.graph

def test_solve_runoff():
  print("Running solve_runoff")
  fn_UHF = gaussian
  fn_IRF = lambda x,t: fundamental_solution(x,t,celerity=1,diffusivity=1)
  def input(t_,step_loc):
    return (t_ < step_loc).astype(float)
  fn_IN = lambda t: input(t,10)
  sim.solve_runoff(watershed, fn_UHF, fn_IRF, fn_IN)
  assert(not isnan(watershed.Q__).any())

def test_solve_contaminant():
  print("Running solve_contaminant")
  fn_MASS = gaussian
  watershed.Q__ += 10
  sim.solve_contaminant(watershed, fn_MASS)
  assert(not isnan(watershed.c__).any())

def test_watershed_xy_points():
  _th = {1:10,2:10,3:5}
  xy__ = watershed.watershed_xy_points(_th)
