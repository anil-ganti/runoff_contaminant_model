from numpy import *

import runoff_contaminant_model.common as cmn
import runoff_contaminant_model.watershed as ws
import runoff_contaminant_model.channel as ch
from runoff_contaminant_model.common import gaussian, IRF

L_ = [10,10,10]
r_ = 2*ones(10).reshape(10,1)
ch_ = list(map(lambda L: ch.Simple_Channel(L,arange(L),r_,None),L_))
confluence_ = [(1,0,7),(2,1,7)]

cfg = {'dx':1, 'dt':1, 'T':10}
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

def test_solve_runoff():
  print("Running solve_runoff")
  fn_UHF = lambda x: gaussian(x, 1, 10, 1)
  fn_IRF = lambda x,t: IRF(x,t,1,1,10, 1)
  def runoff(t_,step_loc):
    return (t_ < step_loc).astype(float)
  fn_RUN = runoff
  sim.solve_runoff(watershed, fn_UHF, fn_IRF, fn_RUN)
