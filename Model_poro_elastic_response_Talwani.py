#!/usr/bin/env python
import numpy as np
import scipy
import scipy.special as sp
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime as UTC
import matplotlib.dates as dates
from obspy.signal.util import smooth

plt.rc('font', family='serif')
plt.rc('font', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('font', size=16)

#---------------- Script to model simple poro-elastic response -----------------#
#--> Adopted from Clements and Denolle, 2023 https://doi.org/10.1029/2022JB025553
    # https://github.com/tclements/Clements-Denolle-2022/blob/main/src/04-fit-thermo-hydro-models.jl
    # -and-
    # Talwani et al., 2007 https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JB004665

#The instantaneous undrained effect manifests itself in two ways:
# 1) Elastic response to the reservoir load
# 2) Instantaneous pore pressure change in the vicinity of the reservoir due to an undrained response

#----- Notes ------#
# error function --> erf(z) = 2/sqrt(np.pi)*intgral(0,Z) e**-t**2 *dt
                #-->  https://en.wikipedia.org/wiki/Error_function

#--- global variables
days = 90
# model constants
r = 600. # radius...in meters ? see Talwani et al., 2007 figure 2
δt = 86400 #number of incremnts (seconds)?
α = (1 + 0.27) / 3 / (1 - 0.27) #elastic constant related to B (Skempton's coefficient)
                                #and the undrained Poisson’s ratio.
ϕ = 0.15 #porosity
c = 100 # Hydraulic diffusivity (ranges from .01 to 100 -- fig 3 Talwani et al., 2007)
#The hydrologic property controlling pore pressure diffusion is hydraulic diffusivity c,
#which is directly related to intrinsic permeability k
#print(r,δt,α,ϕ)

#--- functions ---#
def elastic(precip,c,α,r,δt): #first term, undrained -- instantaneous elastic loading response
    erf_ij = sp.erf(r/np.sqrt(4*c*np.float64(np.arange(0,len(precip)))*δt))
    P = np.convolve(erf_ij,precip-np.mean(precip))*α
    return P[0:int(len(P)/2)+1] #[1:end÷2]

def drained(precip,c,r,δt): #second term, pore pressure induced by diffusion at a distance r
                            #due to an applied load at the surface
    erfc_ij = sp.erf(r/np.sqrt(4*c*np.float64(np.arange(0,len(precip)))*δt))
    P = np.convolve(erfc_ij,precip-np.mean(precip))
    return P[0:int(len(P)/2)+1]

def fullycoupled(precip,c,α,r,δt): # delayed response due to diffusion of pore pressure ?
    erf_ij =  sp.erf(r/np.sqrt(4*c*np.float64(np.arange(0,len(precip)))*δt))
    erfc_ij = sp.erfc(r/np.sqrt(4*c*np.float64(np.arange(0,len(precip)))*δt))
    P1 = np.convolve(erf_ij,precip-np.mean(precip))*α
    P2 = np.convolve(erfc_ij,precip-np.mean(precip))
    P = P1+P2
    return P[0:int(len(P)/2)+1]


#--- model
def smooth_dat(x, half_win, window='boxcar'):
    # TODO: docsting
    window_len = 2 * half_win + 1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]] #row-wise merging
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.hanning(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]

#--- import precipitation
f = open('precip_ANMO_2012_2022.txt', 'r') #--- ANMO weather data
pdates_ANMO = []
pvals_ANMO = []

for line in f: #
    line = line.rstrip()
    line = line.lstrip()
    line = line.split(',')
    pdates_ANMO.append(UTC(line[0].strip('"')))
    pvals_ANMO.append(float(line[1].strip('"')))
f.close()
pdates_ANMO = dates.date2num(pdates_ANMO)
# print(np.shape(pdates_ANMO))
# print(pvals_ANMO)

#--------# Main #-------#

#--- smooth precipitation data
pvals_smooth_ANMO_30 = smooth_dat(pvals_ANMO,30)
pvals_smooth_ANMO_60 = smooth_dat(pvals_ANMO,60)
pvals_smooth_ANMO_90 = smooth_dat(pvals_ANMO,90)

elastic_ASL = elastic(pvals_smooth_ANMO_90,c,α,r,δt)
fc_ASL = fullycoupled(pvals_smooth_ANMO_90,c,α,r,δt)
drained_ASL = drained(pvals_smooth_ANMO_90,c,r,δt)

#--- plot
fig1, ax = plt.subplots(1,1,figsize=(15,5))

plt.plot_date(pdates_ANMO,elastic_ASL,'k-',label='elastic',zorder=1)
plt.plot_date(pdates_ANMO,fc_ASL,'b-',label='fully coupled',zorder=1)
plt.plot_date(pdates_ANMO,drained_ASL,'g-',label='drained',zorder=1)
plt.legend(loc='upper left', ncol = 3)
plt.xlabel('Date')
plt.savefig('ANMO_annual_precip_phase_pub_supp_fig.png',format='png',dpi=300)
plt.show()
