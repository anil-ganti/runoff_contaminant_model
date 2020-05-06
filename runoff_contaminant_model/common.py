from numpy import *

def gaussian(x, amp=1, loc=0, width=1):
  return amp*exp(-(x-loc)**2/(2*width**2))

def normalized_gaussian(x,loc=0,width=1):
  amp = 1/(width*sqrt(2*pi))
  return gaussian(x,amp=amp,loc=loc,width=width)

def IRF(x_,t_,celerity=1, loc=1,width=1):
  i__ = empty([len(t_),len(x_)])
  for i,t in enumerate(t_):
    i__[i,:] = normalized_gaussian(x_,loc=loc + celerity*t,width=width)
  return i__

def fundamental_solution(x_,t_,diffusivity=1,celerity=1):
  out__ = empty([len(t_),len(x_)])
  for i,t in enumerate(t_):
    for j,x in enumerate(x_):
      amp = x / (2*t*sqrt(pi*t*diffusivity))
      loc = celerity*t
      width = sqrt(2*diffusivity*t)
      out__[i,j] = gaussian(x,amp=amp,loc=loc,width=width)
  out__[0,:] = 0
  return out__
