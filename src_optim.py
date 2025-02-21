import sys # system commands
import string as string # string functions4
import math
import numpy as np # numerical tools
from scipy import *
from pylab import *
import os
import itertools
import math as maths
from scipy import integrate
from scipy.stats import distributions,pearsonr,chisquare,norm
from scipy.optimize import curve_fit, minimize,fmin, fmin_powell, root
from scipy import interpolate
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
import random
from sklearn import linear_model, datasets
import uncertainties.unumpy as unumpy 
from uncertainties import ufloat
from tqdm import tqdm
from shapely import geometry
# import matplotlib
# matplotlib.use('Agg')
plt.ion()
rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
############ PATHS ##########################################
c_light = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
H0=70 #km/s/Mpc
zlens = 0.396
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2
pcconv = 30856775814914 # km per 1 pc
syear = 365*24*60*60 # seconds in 1 year
rad_per_deg = np.pi/180 # radians per degree
hubparam2 = (H0**2)*(matter*((1+zlens)**3) + darkenergy)
critdens = ((3*hubparam2)/(8*np.pi*G))*(1/(10**12)) # solar mass per pc^3
p='JOHN GO FUCK YOURSELF'
def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

Ddpc = 1110940809.0062053 # in pc
arcsec_pc = Ddpc*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec
zd = 0.396

#### RAW DATA
def RADEC_to_ARCSEC(RA,Dec,zpra,zpdec):
	# Takes you from RA,Dec in observed degrees to arcsec with respect to given zero point zpra,zpdec

	# First convert RA,Dec to real RA,Dec in deg
	RA = (RA-zpra)*np.cos(Dec*rad_per_deg)
	Dec = Dec - zpdec

	# Next convert to arcsec
	x_arc,y_arc = RA*3600,Dec*3600

	return x_arc,y_arc

# print('dir00 = Normal')
# print('dir11 = Normal + S1/S2 and F1/F2')
# print('dir12 = Normal + S1/F1 and S2/F2')
# print('dir21 = Normal + S1/S2 and F1/F2')
# print('dir22 = Normal + S1/F1 and S2/F2')
# choice = input('Which? (dir00,dir11,dir12,dir21,dir22) ')
# path = '/Users/derek/Desktop/UMN/Research/MACSJ0416/%s'%choice #CHANGE PATH NAME FOR DIFFERENT RUNS
zeropointRA,zeropointDec = 64.03730721518987,-24.070971485232068 # Some random zero point
RAgal = np.loadtxt('MACSJ0416_clustermembers.txt',usecols=1) # Image positions of cluster members in RA degrees
Decgal = np.loadtxt('MACSJ0416_clustermembers.txt',usecols=2) # Image positions of cluster members in Dec degrees
xgal,ygal = RADEC_to_ARCSEC(RAgal,Decgal,zeropointRA,zeropointDec)

# #### RAW DATA
# # 88 Source ID names
# if ((choice == 'dir00') or (choice == 'dir30') or (choice == 'dir40')):
# 	sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
# 					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
# 	ID = np.loadtxt('MACSJ0416.txt',usecols=0,dtype='str')
# 	RAdeg = np.loadtxt('MACSJ0416.txt',usecols=2) # Image positions in RA degrees
# 	Decdeg = np.loadtxt('MACSJ0416.txt',usecols=3) # Image positions in Dec degrees
# 	zs = np.loadtxt('MACSJ0416.txt',usecols=4)
# 	sourceplanes=np.unique(zs)
# 	xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec
# elif ((choice == 'dirsub15')):
# 	trans = 'sub15'
# 	sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,24,5.1,5.2,5.3,5.4,5.5,5.6,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,203,8,29,107,109,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,207,208,108,209,32,35,33,113,
# 					2.1,2.2,112,210.1,210.2,210.3,210.4,211]) # Sub 15
# 	ID = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=0,dtype='str')
# 	RAdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=2) # Image positions in RA degrees
# 	Decdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=3) # Image positions in Dec degrees
# 	zs = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=4)
# 	sourceplanes=np.unique(zs)
# 	xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec
# 	print(trans)
# else:
# 	if choice == 'dir11':
# 		print('130 (D21), 131 (S1/S2), 132 (F1/F2)')
# 		trans = 'trans1'
# 		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,130,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
# 					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
# 	if choice == 'dir12':
# 		print('130 (D21), 131 (S1/F1), 132 (S2/F2)')
# 		trans = 'trans2'
# 		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,130,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
# 					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
# 	if ((choice == 'dir21') or (choice=='dir31') or (choice == 'dir41')):
# 		print('131 (S1/S2), 132 (F1/F2)')
# 		trans = 'trans21'
# 		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
# 					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
# 	if ((choice == 'dir22') or (choice=='dir32')):
# 		print('131 (S1/F1), 132 (S2/F2)')
# 		trans = 'trans22'
# 		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
# 					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])	

# 	ID = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=0,dtype='str')
# 	RAdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=2) # Image positions in RA degrees
# 	Decdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=3) # Image positions in Dec degrees
# 	zs = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=4)
# 	sourceplanes=np.unique(zs)
# 	xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec

# count=0
# imx,imy,sourcez = [],[],[] # x images, y images, source redshifts
# for i in range(len(sourceID)):
# 	xim,yim,zsrc = [],[],[]
# 	if str(sourceID[i])[-2:] == '.0':
# 		for j in range(len(ID)):
# 			if str(sourceID[i])[0:len(str(sourceID[i]))-len(str(sourceID[i])[-2:])] == ID[j].translate({ord(i): None for i in 'abcdefghi'}):
# 				# print(str(sourceID[i])[0:len(str(sourceID[i]))-len(str(sourceID[i])[-2:])],ID[j].translate({ord(i): None for i in 'abcdefghi'}))
# 				xim.append(xarc[j])
# 				yim.append(yarc[j])
# 				zsrc.append(zs[j])
# 				count+=1
# 		imx.append(xim)
# 		imy.append(yim)
# 		sourcez.append(zsrc)
# 		continue
# 	else:
# 		for j in range(len(ID)): 
# 			if str(sourceID[i]) == ID[j].translate({ord(i): None for i in 'abcdefghi'}):
# 				xim.append(xarc[j])
# 				yim.append(yarc[j])
# 				zsrc.append(zs[j])
# 				count+=1
# 		imx.append(xim)
# 		imy.append(yim)
# 		sourcez.append(zsrc)
# # KEY: imx[sourceID][image]

choice = 'CANUCS'
path = '/Users/derek/Desktop/UMN/Research/MACSJ0416/%s'%choice #CHANGE PATH NAME FOR DIFFERENT RUNS 
zeropointRA,zeropointDec = 64.03730721518987,-24.070971485232068 # Some random zero point
fitspath = '/Users/derek/Desktop/UMN/Research/MACSJ0416/imagefits'
idriht = np.loadtxt('%s/images_rihtarsic24.dat'%fitspath,usecols=0,dtype=str)
zsriht = np.loadtxt('%s/images_rihtarsic24.dat'%fitspath,usecols=6)
IDCANUCS = np.array([idriht[i].split('.')[0] for i in range(len(idriht))])
idindex = np.unique(IDCANUCS,return_index=True)[1]
IDCANUCS = np.array([IDCANUCS[index] for index in sorted(idindex)])
# idchoice = input('Source ID: ')
# srcind=np.argwhere(IDCANUCS==idchoice)[0][0] # index in idindex
# zsrc = sourcez[srcind][0]
# print(zsrc)
sourceID = IDCANUCS

# This step now done in dataprocess v2
imx = np.load('%s/imx.npy'%path,allow_pickle=True)
imy = np.load('%s/imy.npy'%path,allow_pickle=True)
sourcez = np.load('%s/sourcez.npy'%path,allow_pickle=True)
# KEY: imx[sourceID][image]

# Mean Source Positions from backprojecting
xsmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=1)
ysmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=2)
Dspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=3)
Ddspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=4)

############## Find Images ##########################
fig2,ax2=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
def fims(x,y,tdsurf):
	# Function takes in a Time delay surface and finds all the image positions
	# Basic idea is to compute the gradients along x and y and find where they are 0, then find the intersection points
	# x,y are the meshgrid that forms the map for tdsurf

	contourx=ax2.contour(x,y,np.gradient(tdsurf)[0],levels=[0],origin='lower',colors='pink')
	contoury=ax2.contour(x,y,np.gradient(tdsurf)[1],levels=[0],origin='lower')

	tgradxx=[] # x values for the gradx of tdsurf
	tgradxy=[] # y values for the gradx of tdsurf
	for item in contourx.collections:
   		for i in item.get_paths():
   		 	v = i.vertices
   		 	xcrit = v[:, 0]
   		 	ycrit = v[:, 1]
   		 	tgradxx.append(xcrit)
   		 	tgradxy.append(ycrit)

	tgradyx=[] # x values for the grady of tdsurf
	tgradyy=[] # y values for the grady of tdsurf
	for item in contoury.collections:
   		for i in item.get_paths():
   			v = i.vertices
   			xcrit = v[:, 0]
   			ycrit = v[:, 1]
   			tgradyx.append(xcrit)
   			tgradyy.append(ycrit)

    # The following loop first checks to see if the contours of x and y intersect
    # If they do, then we calculate the intersection points as the minimum distance between
    # the two contours. 
    # A threshold is set at 0.05 arcsec in separation. We will filter this to get 1 solution later
	xint=[]
	yint=[]
	for ix,contxx in enumerate(tgradxx):
		contxy = tgradxy[ix]
	
		min1x = min(contxx)
		max1x = max(contxx)
		min1y = min(contxy)
		max1y = max(contxy)

		for iy,contyy in enumerate(tgradyy):
			contyx = tgradyx[iy]

			min2y = min(contyy)
			max2y = max(contyy)
			min2x = min(contyx)
			max2x = max(contyx)

			if ((min1x > max2x or min2x > max1x) or 
				(min1y > max2y or min2y > max1y)):

				# This means the contours do not overlap!
				continue

			# Finds the smallest distance between contours at each point
			dist=[]
			for i in range(len(contxx)):
				dist.append(min(np.array([np.sqrt((contxx[i]-contyx[j])**2 + (contxy[i] - contyy[j])**2) for j in range(len(contyy))])))

			# Consider the intersection points as the distances less than 0.05 arcsec
			if min(dist) <= 20.0:
				# xint.append(contxx[np.argmin(dist)])
				# yint.append(contxy[np.argmin(dist)])
				intind = [i for i,item in enumerate(dist) if item<=0.25]
				if len(intind)!=0:
					xint.append(contxx[intind])
					yint.append(contxy[intind])
					
	# if (xint==[] or yint==[]):
	# 	return np.array([]),np.array([])
	xint=np.concatenate(xint)
	yint=np.concatenate(yint)

	# Now identify clusters of distances
	# Take the average of the cluster to be the intersection point
	xrec=[]
	yrec=[]
	ptsep = np.sqrt(np.diff(xint)**2 + np.diff(yint)**2)
	ptsep = np.insert(ptsep,len(ptsep) , 10) # Just to make sure the last distance ends the final cluster
	gate=1 # gate=1 means the cluster has ended
	thresh=0.5 # This is the separation between clusters, 0.2 seems to be fine. Change if necessary (i.e. too frequent finding of phantom images)
	for i,sep in enumerate(ptsep):
		if gate==1:
			if sep>thresh: # Cluster has 1 point, the intersection!
				xrec.append(xint[i])
				yrec.append(yint[i])
				gate=1
				continue
			if sep<=thresh: # New cluster starts, flip gate
				xclust=[]
				yclust=[]
				gate=0
		if gate==0: # We are in a cluster
			if sep<=thresh: # Still in a cluster
				xclust.append(xint[i])
				yclust.append(yint[i])
				continue
			if sep>thresh: # Cluster over!
				xclust.append(xint[i])
				yclust.append(yint[i])

				xrec.append(np.mean(xclust))
				yrec.append(np.mean(yclust))
				gate=1

	return np.array(xrec),np.array(yrec)

def fims2(x,y,tdsurf):
	# Function takes in a Time delay surface and finds all the image positions
	# Basic idea is to compute the gradients along x and y and find where they are 0, then find the intersection points
	# x,y are the meshgrid that forms the map for tdsurf

	contourx=ax2.contour(x,y,np.gradient(tdsurf)[0],levels=[0],origin='lower',colors='pink')
	contoury=ax2.contour(x,y,np.gradient(tdsurf)[1],levels=[0],origin='lower')

	tgradxx=[] # x values for the gradx of tdsurf
	tgradxy=[] # y values for the gradx of tdsurf
	verticesx=[]
	for item in contourx.collections:
		for i in item.get_paths():
			v = i.vertices
			verticesx.append(v)
			xcrit = v[:, 0]
			ycrit = v[:, 1]
			tgradxx.append(xcrit)
			tgradxy.append(ycrit)

	tgradyx=[] # x values for the grady of tdsurf
	tgradyy=[] # y values for the grady of tdsurf
	verticesy=[]
	for item in contoury.collections:
		for i in item.get_paths():
			v = i.vertices
			verticesy.append(v)
			xcrit = v[:, 0]
			ycrit = v[:, 1]
			tgradyx.append(xcrit)
			tgradyy.append(ycrit)

	xrec,yrec = [],[]
	for vx in verticesx:
		for i,vy in enumerate(verticesy):
			polyx = geometry.LineString(vx)
			polyy = geometry.LineString(vy)
			try:
				intersection = polyx.intersection(polyy)
				xrec.append(np.array([float(intersection.geoms[i].x) for i in range(len(intersection.geoms))]))
				yrec.append(np.array([float(intersection.geoms[i].y) for i in range(len(intersection.geoms))]))
			except AttributeError: # Lines don't intersect!
				continue

			
	return np.concatenate(np.array(xrec)),np.concatenate(np.array(yrec))

##### DEFINE METROPOLIS HASTINGS COMPONENTS ######
def prior(params,window): # Priors are flat
    logprior = 0
    xs,ys = params
    xlow,xhigh,ylow,yhigh = window
    
    if ((xs<xlow) or (xs>xhigh)):
        return -np.inf

    if ((ys<ylow) or (ys>yhigh)):
        return -np.inf
    
    return logprior
    
def posterior(params, y, dy, window):
   
	# y: [xim,yim]
	# dy: Uncertainty on observed images positions (0.04 from HST)
	xs,ys = params
	xref,yref = y

    # Prior
	priors = prior(params,window)

    # Calculate model
	thetabeta = ((xfield-xs)/206264.80624709636)*((xfield-xs)/206264.80624709636) + ((yfield-ys)/206264.80624709636)*((yfield-ys)/206264.80624709636)
	avgtdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - avglpot.T)
	try:
		xrecon,yrecon = fims2(xfield,yfield,avgtdsurf)
	except ValueError:
		xrecon,yrecon = fims(xfield,yfield,avgtdsurf)

	# Sort images to make sure reconstructed images are in same order as xref!
	xrec=[]
	yrec=[]
	for i in range(len(xref)):
		distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
		xrec.append(xrecon[np.argmin(distim)])
		yrec.append(yrecon[np.argmin(distim)])
	xrec,yrec = np.array(xrec),np.array(yrec)

    # Add likelihood to prior to get posterior
	distref = np.sqrt(xref**2 + yref**2)
	distrec = np.sqrt(xrec**2 + yrec**2)
	likelihood = -0.5*sum((((xref - xrec)**2)/(dy**2)) + (((yref - yrec)**2)/(dy**2)) + (len(xrecon)-len(xref))**2)

	logpost = likelihood + priors
    
	return logpost

def metropolis(sampsize, initial_state, sigma,y,dy,window):

	n_params = len(initial_state)
    
	#trace = np.empty((sampsize+1,n_params))
	trace = [[] for x in range(sampsize+1)]

	trace[0] = initial_state[0] # Set parameters to the initial state
	logprob = posterior(trace[0],y,dy,window) # Compute p(x)

	accepted = [0]*n_params

	for i in tqdm(range(sampsize)): # while we want more samples
		iterparams = trace[i] # Current parameters in the trace
		# print(iterparams)
        
		for j in range(n_params):
			# draw x' from the proposal distribution (a gaussian)
			iterparams_j = trace[i].copy()
			#print(iterparams_j)
			xprime = norm.rvs(loc=iterparams[0],scale=sigma[j][0],size=1)[0]
			yprime = norm.rvs(loc=iterparams[1],scale=sigma[j][1],size=1)[0]
			iterparams_j = [xprime,yprime]
			#print(iterparams_j)
			
            
			# compute p(x')
			logprobprime = posterior(iterparams_j,y,dy,window)

			alpha = logprobprime - logprob
            # Draw uniform from uniform distribution
			u = np.random.uniform(0,1,1)[0]
            
            # Test sampled value
			if np.log(u) < alpha:
				#trace[i+1,j] = xprime
				trace[i+1] = iterparams_j
				logprob = logprobprime
				accepted[j] += 1
			else:
				#trace[i+1,j] = trace[i,j]
				trace[i+1] = trace[i]
                
	return np.array(trace),np.array(accepted)

# Cloud of backprojected source positions. Used as prior
xscloud = np.load('%s/xcloud.npy'%(path),allow_pickle=True)
yscloud = np.load('%s/ycloud.npy'%(path),allow_pickle=True)
file = open(f'{path}/optimsourcepos.txt','w')

# spocks = np.array([13])
# warhol = np.array([12.4])
for src in sourceID:
	# #### For v2 Rihtarsic+24 ID
	srcind=np.argwhere(IDCANUCS==src)[0][0] # index in idindex
	zsrc = sourcez[srcind][0]
	# #### For orginal Bergamini+23 ID
	# srcind = np.argwhere(sourceID==float(src))[0][0]
	# zsrc = sourcez[srcind][0]
	avglpot = np.genfromtxt('%s/AVGlenspot_%s.txt'%(path,zsrc))
	print(src,srcind)
	#################### DEFINE GRID ############################
	gridsize=500
	xlow,ylow,xhigh,yhigh = -70,-70,71,71
	nx = (xhigh-xlow)/gridsize
	ny = (yhigh-ylow)/gridsize
	xrang = np.arange(xlow,xhigh,nx)
	yrang = np.arange(ylow,yhigh,ny)
	xfield,yfield = np.meshgrid(xrang,yrang)
	
	Dspc,Ddspc = Dspc_all[srcind],Ddspc_all[srcind]

	thetabeta_test = ((xfield-xsmean[srcind])/206264.80624709636)*((xfield-xsmean[srcind])/206264.80624709636) + ((yfield-ysmean[srcind])/206264.80624709636)*((yfield-ysmean[srcind])/206264.80624709636)
	avgtdsurf_test = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta_test - avglpot.T)
	try:
		xtest,ytest = fims2(xfield,yfield,avgtdsurf_test)
	except ValueError:
		xtest,ytest = fims(xfield,yfield,avgtdsurf_test)

# Sort images to make sure reconstructed images are in same order as xqso!
	xrectest=[]
	yrectest=[]
	for i in range(len(np.array(imx[srcind]))):
		distim = np.sqrt((np.array(xtest)-np.array(imx[srcind])[i])**2 + (np.array(ytest)-np.array(imy[srcind])[i])**2)
		xrectest.append(xtest[np.argmin(distim)])
		yrectest.append(ytest[np.argmin(distim)])
	xrectest,yrectest = np.array(xrectest),np.array(yrectest)
	distrectest = np.sqrt((np.array(xrectest)-np.array(imx[srcind]))**2 + (np.array(yrectest)-np.array(imy[srcind]))**2)
	print('Initial Mean Image Separation:',np.mean(distrectest))
	##########################################################
	##########################################################
	#################### DO METROPOLIS HASTINGS ##############
	##########################################################
	##########################################################
	n_iter = 500
	xs,ys = xsmean[srcind],ysmean[srcind] 
	initstate = [[xs,ys]]
	sigma = [[0.04,0.04]]
	refx,refy = np.array(imx[srcind]),np.array(imy[srcind])
	dyobs=np.zeros(len(refx))+0.04
	priorwindow = [min(xscloud[srcind]),max(xscloud[srcind]),min(yscloud[srcind]),max(yscloud[srcind])]
	trace,accept = metropolis(n_iter,initstate,sigma,[refx,refy],dyobs,priorwindow)
	print(initstate)
	print('Old: ',xs,ys)
	print(trace[-1],accept)

	#### Check if it has been optimized
	xf,yf = statistics.mode(trace.T[0]),statistics.mode(trace.T[1])#trace[-1][0],trace[-1][1]
	thetabeta_final = ((xfield-xf)/206264.80624709636)*((xfield-xf)/206264.80624709636) + ((yfield-yf)/206264.80624709636)*((yfield-yf)/206264.80624709636)
	avgtdsurf_final = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta_final - avglpot.T)
	try:
		xfinal,yfinal = fims2(xfield,yfield,avgtdsurf_final)
	except ValueError:
		xfinal,yfinal = fims(xfield,yfield,avgtdsurf_final)

	# Sort images to make sure reconstructed images are in same order as xqso!
	xrecfinal=[]
	yrecfinal=[]
	for i in range(len(np.array(imx[srcind]))):
		distim = np.sqrt((np.array(xfinal)-np.array(imx[srcind])[i])**2 + (np.array(yfinal)-np.array(imy[srcind])[i])**2)
		xrecfinal.append(xfinal[np.argmin(distim)])
		yrecfinal.append(yfinal[np.argmin(distim)])
	xrecfinal,yrecfinal = np.array(xrecfinal),np.array(yrecfinal)
	distrecfinal = np.sqrt((np.array(xrecfinal)-np.array(imx[srcind]))**2 + (np.array(yrecfinal)-np.array(imy[srcind]))**2)
	print('Final Mean Image Separation:',np.mean(distrecfinal))
	if np.mean(distrecfinal) < np.mean(distrectest):
		print('SUCCESS!')
		file.writelines('%f  %f  \n'%(xf,yf))
		print('New: ',xf,yf)
	else:
		print('FAIL!')
		print(xs,ys)
		file.writelines('%f  %f  \n'%(xs,ys))

	fig2.clear()

file.close()
