from numpy import *

def gaussian(x, amp, loc, width):
    return amp*exp(-(x-loc)**2/(2*width**2))

def IRF(x_,t_,cel,amp,loc,width):
  i__ = empty([len(t_),len(x_)])
  for i,t in enumerate(t_):
    i__[i,:] = gaussian(x_,amp,loc + cel*t,width)
  return i__
