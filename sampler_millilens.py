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
from scipy.interpolate import interp1d,RectBivariateSpline
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
import emcee
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

print('dir00 = Normal')
print('dir11 = Normal + S1/S2 and F1/F2')
print('dir12 = Normal + S1/F1 and S2/F2')
print('dir21 = Normal + S1/S2 and F1/F2')
print('dir22 = Normal + S1/F1 and S2/F2')
choice = input('Which? (dir00,dir11,dir12,dir21,dir22) ')
path = '/Users/derek/Desktop/UMN/Research/MACSJ0416/%s'%choice #CHANGE PATH NAME FOR DIFFERENT RUNS
bestgen = np.loadtxt('%s/bestfits.txt'%(path),usecols=0)
zeropointRA,zeropointDec = 64.03730721518987,-24.070971485232068 # Some random zero point
RAgal = np.loadtxt('MACSJ0416_clustermembers.txt',usecols=1) # Image positions of cluster members in RA degrees
Decgal = np.loadtxt('MACSJ0416_clustermembers.txt',usecols=2) # Image positions of cluster members in Dec degrees
xgal,ygal = RADEC_to_ARCSEC(RAgal,Decgal,zeropointRA,zeropointDec)

#### Mothra Data ####
xmoth,ymoth = RADEC_to_ARCSEC(64.03676, -24.06625,zeropointRA,zeropointDec) # Mothra Yan23
zmoth = 2.091
Dmoth = 1764803129.025831 # in pc
Dlummoth = Dmoth*((1+zmoth)**2) # in pc
absmagmoth =-7.84 # roughly Rigel in V band (intrinsic)
obsmagmoth = 28.85 # observed apparent magnitude
expmagmoth = absmagmoth + 5*np.log10(Dlummoth) - 5 # expected apparent magnitude 
fmoth_exp = 10**((expmagmoth + 48.60)/(-2.5)) # expected flux in cgs
fmoth_obs = 10**((obsmagmoth + 48.60)/(-2.5)) # observed flux in cgs
magnmoth = fmoth_obs/fmoth_exp # Required magnification for Mothra!

#### RAW DATA
# 88 Source ID names
if ((choice == 'dir00') or (choice == 'dir30') or (choice == 'dir40')):
	sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
	ID = np.loadtxt('MACSJ0416.txt',usecols=0,dtype='str')
	RAdeg = np.loadtxt('MACSJ0416.txt',usecols=2) # Image positions in RA degrees
	Decdeg = np.loadtxt('MACSJ0416.txt',usecols=3) # Image positions in Dec degrees
	zs = np.loadtxt('MACSJ0416.txt',usecols=4)
	sourceplanes=np.unique(zs)
	xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec
else:
	if choice == 'dir11':
		print('130 (D21), 131 (S1/S2), 132 (F1/F2)')
		trans = 'trans1'
		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,130,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
	if choice == 'dir12':
		print('130 (D21), 131 (S1/F1), 132 (S2/F2)')
		trans = 'trans2'
		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,130,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
	if ((choice == 'dir21') or (choice=='dir31') or (choice == 'dir41')):
		print('131 (S1/S2), 132 (F1/F2)')
		trans = 'trans21'
		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])
	if ((choice == 'dir22') or (choice=='dir32')):
		print('131 (S1/F1), 132 (S2/F2)')
		trans = 'trans22'
		sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,131,132,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211])	

	ID = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=0,dtype='str')
	RAdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=2) # Image positions in RA degrees
	Decdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=3) # Image positions in Dec degrees
	zs = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=4)
	sourceplanes=np.unique(zs)
	xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec

count=0
imx,imy,sourcez = [],[],[] # x images, y images, source redshifts
for i in range(len(sourceID)):
	xim,yim,zsrc = [],[],[]
	if str(sourceID[i])[-2:] == '.0':
		for j in range(len(ID)):
			if str(sourceID[i])[0:len(str(sourceID[i]))-len(str(sourceID[i])[-2:])] == ID[j].translate({ord(i): None for i in 'abcdefghi'}):
				# print(str(sourceID[i])[0:len(str(sourceID[i]))-len(str(sourceID[i])[-2:])],ID[j].translate({ord(i): None for i in 'abcdefghi'}))
				xim.append(xarc[j])
				yim.append(yarc[j])
				zsrc.append(zs[j])
				count+=1
		imx.append(xim)
		imy.append(yim)
		sourcez.append(zsrc)
		continue
	else:
		for j in range(len(ID)): 
			if str(sourceID[i]) == ID[j].translate({ord(i): None for i in 'abcdefghi'}):
				xim.append(xarc[j])
				yim.append(yarc[j])
				zsrc.append(zs[j])
				count+=1
		imx.append(xim)
		imy.append(yim)
		sourcez.append(zsrc)
# KEY: imx[sourceID][image]

######## DEFINE LENS POTENTIAL GRID ###########
xsmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=0)
ysmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=1)
Dspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=3)
Ddspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=4)

print(sourceID)
src = input('Which Source: ')
srcind = np.argwhere(sourceID==float(src))[0][0]
zsrc = sourcez[srcind][0]
avglpot = np.genfromtxt('%s/AVGlenspotMODEL_%s.txt'%(path,zsrc)) # lens potential in modelling window 
xref,yref = imx[srcind],imy[srcind]

###### MAIN GRID #####
gridsize=1000
xlow,ylow,xhigh,yhigh = xmoth-0.7,ymoth-0.7,xmoth+0.7,ymoth+0.7
nx = (xhigh-xlow)/gridsize
ny = (yhigh-ylow)/gridsize
stepx=(xhigh-xlow)/avglpot.shape[0]
stepy=(yhigh-ylow)/avglpot.shape[1]
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
x,y = np.meshgrid(xrang,yrang)
xl,yl,xh,yh = xmoth-0.7,ymoth-0.7,xmoth+0.7,ymoth+0.7

xs,ys = xsmean[srcind],ysmean[srcind]
thetabeta = ((x-xs)/206264.80624709636)*((x-xs)/206264.80624709636) + ((y-ys)/206264.80624709636)*((y-ys)/206264.80624709636)
Dspc,Ddspc = Dspc_all[srcind],Ddspc_all[srcind]
sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2
avgtdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - avglpot.T)

def magnmap(lpot,xrang,yrang):
	psix = np.gradient(lpot.T)[0]/(nx/206264.80624709636)
	psiy = np.gradient(lpot.T)[1]/(nx/206264.80624709636)
	psixx = np.gradient(psix)[0]/(nx/206264.80624709636)
	psiyy = np.gradient(psiy)[1]/(nx/206264.80624709636)
	psixy = np.gradient(psix)[1]/(nx/206264.80624709636)
	psiyx = np.gradient(psiy)[0]/(nx/206264.80624709636)
	kap = 0.5*(psixx + psiyy)
	gam1 = 0.5*(psixx - psiyy)
	gam2 = psixy
	avgmassdens = np.genfromtxt('%s/AVGmassdens_MODEL.txt'%(path))/sigcrit
	massinterp = RectBivariateSpline(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]),avgmassdens)
	kapinterp = RectBivariateSpline(xrang,yrang,kap)
	mref=massinterp.ev(x,y)
	kref=kapinterp.ev(x,y)
	factor = np.mean(mref/kref)#534526963912.9919#
	# print(factor)
	kap = kap*factor
	gam1 = gam1*factor
	gam2 = gam2*factor
	detmag = (1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2)

	return detmag,kap

######## Sublens Models ######
def NFWlpot(x,y,xpos,ypos,params):
	# xpos,ypos are positions of the NFW in arcsec
	# x,y are positions to calculate lpot at in arcsec
	r_s,rho_s = params # scale radius (pc), scale density (Msun/pc^3)
	theta_s = r_s/Ddpc # radians probably

	r = np.sqrt((x-xpos)**2 + (y-ypos)**2)/206264.80624709636 # distnace to NFW in radians

	thetadim = r/theta_s
	if thetadim < 1: 
		B = (np.log(thetadim/2)**2) - (np.arctanh(np.sqrt(1 - thetadim**2))**2)
	if thetadim == 1:
		B = np.log(2)**2
	if thetadim > 1:
		B = (np.log(thetadim/2)**2) + (np.arctan(np.sqrt(thetadim**2 - 1))**2)

	return (Ddspc/Dspc)*((8*np.pi*G*r_s*r_s*rho_s*theta_s)/(c_light**2))*B

def rNFW(r,params):
	# Input r in pc
	r_s,rho_s = params # scale radius (pc), scale density (Msun/pc^3)
	return rho_s/((r/r_s)*((1+(r/r_s))**2))

def coreSUB(x,y,xpos,ypos,params):
	# xpos,ypos are positions of the subhalo in arcsec
	# x,y are positions to calculate lpot at in arcsec	
	b = float(np.array(params))
	a = 0.005
	D = 3.5e-13
	r = np.sqrt((x-xpos)**2 + (y-ypos)**2) # distance to subhalo in arcsec

	return D*(b/(a-b))*((a*np.log(a+r)) - (b*np.log(b+r)))

def kappatstrip(r,params):
	b,D = params
	a = 0.005
	num = 0.5*r*(1+(a/b)) + a
	dem = (((r**2)/b) + r*(1+(a/b)) +a)**2
	return D*(num/dem)

def masststrip(rlim,params):
	b,D = params
	a = 0.005
	# r is arcsec
	integrand = lambda r : r*kappatstrip(r,[b,D])*sigcrit
	mass = 2*np.pi*integrate.quad(integrand,0,rlim)
	return mass # Solar masses

def alphapot(x,y,xpos,ypos,params):
	alpha = float(np.array(params)) 
	b=2.5e-14
	s=0.004
	q=1
	K = 0

	thetax,thetay = (x-xpos),(y-ypos) # arcsec

	return b*(s**2 + thetax**2 + (thetay/q)**2 + K*K*thetax*thetay)**(alpha/2)

detmag,avgkap = magnmap(avglpot,xrang,yrang)

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

xrecon,yrecon = fims2(x,y,avgtdsurf)
xrec=[]
yrec=[]
for i in range(len(xref)):
	distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
	xrec.append(xrecon[np.argmin(distim)])
	yrec.append(yrecon[np.argmin(distim)])
xrec,yrec = np.array(xrec),np.array(yrec)
avgdetmaginterp = RectBivariateSpline(xrang,yrang,detmag.T)
magnb0,magnb1 = 1/avgdetmaginterp.ev(xrec[0],yrec[0]),1/avgdetmaginterp.ev(xrec[1],yrec[1])
print(magnb0,magnb1) # Magnifications at arc image positions

macroCC = ax2.contour(x,y,detmag,levels=[0],colors='y',linestyles='dashed')
yccrang = np.linspace(16.8,17.4,5000)
p = macroCC.collections[0].get_paths()[0]
v = p.vertices
xCCmac = v[:,0]
yCCmac = v[:,1]
xccmacrang = interp1d(yCCmac,xCCmac)(yccrang) # Cluster CC

##### DEFINE METROPOLIS HASTINGS COMPONENTS ######
def prior(params,window): # Priors are flat
    logprior = 0
    # r_s,rho_s = params
    b = float(np.array(params))
    # alpha = float(np.array(params))
    
    # if ((r_s <= 0) or (r_s >= 50)):
    #     return -np.inf

    # if ((rho_s <= 0) or (rho_s >= 40)):
    #     return -np.inf

    return logprior
    
def posterior(params, y, dy, window):
   
	# y: [xim,yim]
	# dy: Uncertainty on observed images positions (0.04 from HST) and magnification
	# xsub,ysub,r_s,rho_s = params # Use if optimizing position
	# r_s,rho_s = params
	b = float(np.array(params))
	# alpha = float(np.array(params))
	xref,yref,xsub,ysub = y # only xref,yref if optimizing position of NFW

    # Prior
	priors = prior(params,window)

    # Calculate model
	thetabeta = ((xfield-xs)/206264.80624709636)*((xfield-xs)/206264.80624709636) + ((yfield-ys)/206264.80624709636)*((yfield-ys)/206264.80624709636)
	# sublens = np.array([NFWlpot(i,j,xsub,ysub,[r_s,rho_s]) for i in xrang for j in yrang]).reshape(gridsize,gridsize)
	sublens = np.array([coreSUB(i,j,xsub,ysub,[b]) for i in xrang for j in yrang]).reshape(gridsize,gridsize)
	# sublens = np.array([alphapot(i,j,xsub,ysub,[alpha]) for i in xrang for j in yrang]).reshape(gridsize,gridsize)
	totlpot = avglpot+sublens
	tottdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - totlpot.T)
	try:
		xrecon,yrecon = fims2(xfield,yfield,tottdsurf)
	except ValueError:
		xrecon,yrecon = np.array([0.0,0.0]),np.array([0.0,0.0])

	# Sort images to make sure reconstructed images are in same order as xref!
	xrec=[]
	yrec=[]
	for i in range(len(xref)):
		distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
		xrec.append(xrecon[np.argmin(distim)])
		yrec.append(yrecon[np.argmin(distim)])
	xrec,yrec = np.array(xrec),np.array(yrec)

	# Get magnification at Mothra
	detmagtot,kaptot = magnmap(totlpot,xrang,yrang)
	detmaginterp = RectBivariateSpline(xrang,yrang,detmagtot.T)
	magn = 1/detmaginterp.ev(xmoth,ymoth)
	magnarc0,magnarc1 = 1/detmaginterp.ev(xrec[0],yrec[0]),1/detmaginterp.ev(xrec[1],yrec[1])

	# Critical Curve displacement
	milliCC = ax2.contour(xfield,yfield,detmagtot,levels=[0])
	p = milliCC.collections[0].get_paths()[0]
	v = p.vertices
	xCCmil = v[:,0]
	yCCmil = v[:,1]
	xccmilrang = interp1d(yCCmil,xCCmil)(yccrang)

    # Add likelihood to prior to get posterior
	distref = np.sqrt(xref**2 + yref**2)
	distrec = np.sqrt(xrec**2 + yrec**2)
	likelihood = -0.5*sum((((xref - xrec)**2)/(dy[0]**2)) + (((yref - yrec)**2)/(dy[1]**2)) + (((magnmoth-magn)**2)/(dy[2]**2)) 
				+ (((magnb0-magnarc0)**2)/(dy[3]**2)) + (((magnb1-magnarc1)**2)/(dy[3]**2)) + ((max(xccmilrang-xccmacrang)**2)/(dy[4]**2)))

	logpost = likelihood + priors
    
	return logpost

def BIC(params,y,dy,N):
	K = len(params) # Number of parameters
	# N = number of iterations
	# r_s,rho_s = params
	b = float(np.array(params))
	# alpha = float(np.array(params))
	xref,yref,xsub,ysub = y # only xref,yref if optimizing position of NFW

	thetabeta = ((xfield-xs)/206264.80624709636)*((xfield-xs)/206264.80624709636) + ((yfield-ys)/206264.80624709636)*((yfield-ys)/206264.80624709636)
	# sublens = np.array([NFWlpot(i,j,xsub,ysub,[r_s,rho_s]) for i in xrang for j in yrang]).reshape(gridsize,gridsize)
	sublens = np.array([coreSUB(i,j,xsub,ysub,[b]) for i in xrang for j in yrang]).reshape(gridsize,gridsize)
	# sublens = np.array([alphapot(i,j,xsub,ysub,[alpha]) for i in xrang for j in yrang]).reshape(gridsize,gridsize)
	totlpot = avglpot+sublens
	tottdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - totlpot.T)
	try:
		xrecon,yrecon = fims2(xfield,yfield,tottdsurf)
	except ValueError:
		xrecon,yrecon = np.array([0.0,0.0]),np.array([0.0,0.0])

	# Sort images to make sure reconstructed images are in same order as xref!
	xrec=[]
	yrec=[]
	for i in range(len(xref)):
		distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
		xrec.append(xrecon[np.argmin(distim)])
		yrec.append(yrecon[np.argmin(distim)])
	xrec,yrec = np.array(xrec),np.array(yrec)

	# Get magnification at Mothra
	detmagtot,kaptot = magnmap(totlpot,xrang,yrang)
	detmaginterp = RectBivariateSpline(xrang,yrang,detmagtot.T)
	milliCC = ax2.contour(xfield,yfield,detmagtot,levels=[0])
	magn = 1/detmaginterp.ev(xmoth,ymoth)
	magnarc0,magnarc1 = 1/detmaginterp.ev(xrec[0],yrec[0]),1/detmaginterp.ev(xrec[1],yrec[1])

	# Critical Curve displacement
	p = milliCC.collections[0].get_paths()[0]
	v = p.vertices
	xCCmil = v[:,0]
	yCCmil = v[:,1]
	xccmilrang = interp1d(yCCmil,xCCmil)(yccrang)

    # Add likelihood to prior to get posterior
	distref = np.sqrt(xref**2 + yref**2)
	distrec = np.sqrt(xrec**2 + yrec**2)
	loglikelihood = -0.5*sum((((xref - xrec)**2)/(dy[0]**2)) + (((yref - yrec)**2)/(dy[1]**2)) + (((magnmoth-magn)**2)/(dy[2]**2)) 
				+ (((magnb0-magnarc0)**2)/(dy[3]**2)) + (((magnb1-magnarc1)**2)/(dy[3]**2)) + ((max(xccmilrang-xccmacrang)**2)/(dy[4]**2)))

	bic = K*np.log(N) - 2*loglikelihood
	aic = 2*K - 2*loglikelihood

	return bic,aic

def metropolis(sampsize, initial_state, sigma,y,dy,window):

	n_params = len(initial_state)
    
	trace = np.empty((sampsize+1,n_params))

	trace[0] = initial_state # Set parameters to the initial state
	logprob = posterior(trace[0],y,dy,window) # Compute p(x)

	accepted = [0]*n_params

	for i in tqdm(range(sampsize)): # while we want more samples
		iterparams = trace[i] # Current parameters in the trace
		print(iterparams)
		if i%100 == 0:
			bic = BIC(iterparams,y,dy,i+1)
			print(bic)
        
		for j in range(n_params):
			# draw x' from the proposal distribution (a gaussian)
			iterparams_j = trace[i].copy()
			#print(iterparams_j)
			xprime = norm.rvs(loc=iterparams[j],scale=sigma[j],size=1)[0]
			iterparams_j[j] = xprime
			#print(iterparams_j)
			
            
			# compute p(x')
			logprobprime = posterior(iterparams_j,y,dy,window)

			alpha = logprobprime - logprob
            # Draw uniform from uniform distribution
			u = np.random.uniform(0,1,1)[0]
            
            # Test sampled value
			if np.log(u) < alpha:
				trace[i+1,j] = xprime
				logprob = logprobprime
				accepted[j] += 1
			else:
				trace[i+1,j] = trace[i,j]

		# np.savetxt('millilenstrace_tstrip_030R.txt',trace)
                
	return np.array(trace),np.array(accepted)

##########################################################
##########################################################
#################### DO METROPOLIS HASTINGS ##############
##########################################################
##########################################################
xfield,yfield = np.meshgrid(xrang,yrang)
n_iter = 1000
xs,ys = xsmean[srcind],ysmean[srcind] 
# initstate = [xmoth+0.05,ymoth-0.05,50,30] # Use if optimizing position
initstate = [0.006]
rdist = 0.050
xdist,ydist = np.sqrt((rdist**2)/2),np.sqrt((rdist**2)/2)
xnfw,ynfw = xmoth+xdist,ymoth-ydist
# sigma = [0.2,0.2,25,15] # Sample Uncertainty # Use if optimizing position
sigma = [0.000005] # Sample Uncertainty
refx,refy = np.array(imx[srcind]),np.array(imy[srcind])
dyobs=np.array([0.04,0.04,10,1,0.01]) # Observed uncertainty in prior/likelihood
priorwindow = [xmoth-0.5,xmoth+0.5,ymoth-0.5,ymoth+0.5]
# trace,accept = metropolis(n_iter,initstate,sigma,[refx,refy,xnfw,ynfw],dyobs,priorwindow)

pos = np.array(initstate) + 1e-4 * np.random.randn(2, 1)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, posterior, args=([refx,refy,xnfw,ynfw],dyobs,priorwindow)
)
sampler.run_mcmc(pos, n_iter, progress=True)

samples = sampler.get_chain()

sys.exit()

