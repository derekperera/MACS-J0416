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
from scipy.stats import distributions,pearsonr,chisquare,norm,gaussian_kde
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
import pandas as pd
plt.ion()
rc('font', weight='bold')
############ PATHS ##########################################
c_light = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
c_light=299792.458#in km/s
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

# Kelly et. al. 22 (F1 and F2)
RAF,DecF = np.array([64.0388904,64.0386110]),np.array([-24.0701557,-24.0699715]) # Spock F1 and F2 from Kelly22
xspockF,yspockF = RADEC_to_ARCSEC(RAF,DecF,zeropointRA,zeropointDec)

# RAspock,Decspock = np.array([64.03847,64.03889,64.03874,64.03836]),np.array([-24.06984,-24.07017,-24.07004,-24.06978]) # From Yan23
# xspock,yspock = RADEC_to_ARCSEC(RAspock,Decspock,zeropointRA,zeropointDec)
RAD21,DecD21 = np.array([64.03847,64.03889]),np.array([-24.06984,-24.07017]) # D21-S1 and D21-S2 from Yan23
xD21,yD21 = RADEC_to_ARCSEC(RAD21,DecD21,zeropointRA,zeropointDec)
xD31,yD31 = RADEC_to_ARCSEC(64.03836, -24.06978,zeropointRA,zeropointDec)

RAdieg,Decdieg = np.array([64.0393337,64.0383658]),np.array([-24.0704250,-24.0697531])
xspockd,yspockd = RADEC_to_ARCSEC(RAdieg,Decdieg,zeropointRA,zeropointDec) # From Diego 22

RAS,DecS= np.array([64.038565,64.038998]),np.array([-24.069939,-24.070241])
xspockS,yspockS = RADEC_to_ARCSEC(RAS,DecS,zeropointRA,zeropointDec) # From Rodney 18 (S1 and S2)

xmoth,ymoth = RADEC_to_ARCSEC(64.03676, -24.06625,zeropointRA,zeropointDec) # Mothra Yan23
xwar,ywar = RADEC_to_ARCSEC(64.0362850,-24.0674847,zeropointRA,zeropointDec) # Warhol Chen19

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

# Mean Source Positions from backprojecting
xsmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=0)
ysmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=1)
# xsmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=1)
# ysmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=2)
Dspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=3)
Ddspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=4)
# See Kelly et. al. 2022
gspockx,gspocky = np.array([imx[6][0]+1.346153846153845]),np.array([imy[6][0]+3.384615384615387])

print(sourceID)
src = input('Which Source: ')
srcind = np.argwhere(sourceID==float(src))[0][0]
zsrc = sourcez[srcind][0]
avglpot = np.genfromtxt('%s/AVGlenspot_%s.txt'%(path,zsrc))
xref,yref = imx[srcind],imy[srcind]

#################### DEFINE GRID ############################
gridsize=500
xlow,ylow,xhigh,yhigh = -70,-70,71,71
nx = (xhigh-xlow)/gridsize
ny = (yhigh-ylow)/gridsize
stepx=(xhigh-xlow)/avglpot.shape[0]
stepy=(yhigh-ylow)/avglpot.shape[1]
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
x,y = np.meshgrid(xrang,yrang)

############## Find Images ##########################
fig2,ax2=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
fig9,ax9=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
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

xs,ys = xsmean[srcind],ysmean[srcind]
thetabeta = ((x-xs)/206264.80624709636)*((x-xs)/206264.80624709636) + ((y-ys)/206264.80624709636)*((y-ys)/206264.80624709636)
Dspc,Ddspc = Dspc_all[srcind],Ddspc_all[srcind]
sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2
avgtdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - avglpot.T)
xrecon,yrecon = fims(x,y,avgtdsurf)
try:
	xtest,ytest = fims2(x,y,avgtdsurf)
	print('Worked')
	# plt.scatter(xrecon,yrecon,color='b',label='Manual')	
	ax9.scatter(xtest,ytest,color='y',label='Reconstructed',marker='o',zorder=2,facecolors='none', edgecolors='y')
	print('Extraneous Images: ',len(xtest)-len(xref))
except ValueError:
	xtest,ytest = fims(x,y,avgtdsurf)
	print('shapely failed, go manual')
	plt.scatter(xrecon,yrecon,color='b',label='Manual')
	print('Extraneous Images: ',len(xrecon)-len(xref))

# Sort images to make sure reconstructed images are in same order as xqso!
xrec=[]
yrec=[]
for i in range(len(xref)):
	distim = np.sqrt((np.array(xtest)-xref[i])**2 + (np.array(ytest)-yref[i])**2)
	xrec.append(xtest[np.argmin(distim)])
	yrec.append(ytest[np.argmin(distim)])
xrec,yrec = np.array(xrec),np.array(yrec)

distrecon = np.sqrt((np.array(xrec)-xref)**2 + (np.array(yrec)-yref)**2)
print('Mean Image Separation:',np.mean(distrecon))

avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))/sigcrit
stdmassdens = np.genfromtxt('%s/STDmassdens.txt'%(path))/sigcrit
xm,ym = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))
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
	# avgmassdens = np.genfromtxt('%s/AVGmassdens_MODEL.txt'%(path))/sigcrit
	# massinterp = RectBivariateSpline(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]),avgmassdens)
	# kapinterp = RectBivariateSpline(xrang,yrang,kap)
	# mref=massinterp.ev(x,y)
	# kref=kapinterp.ev(x,y)
	factor = 1.0#np.mean(mref/kref)#534526963912.9919#
	# print(factor)
	kap = kap*factor
	gam1 = gam1*factor
	gam2 = gam2*factor
	detmag = (1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2)

	return detmag,kap

# im = ax9.imshow(stdmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='Greys',origin='lower')
# plt.imshow(deta,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='seismic',origin='lower',norm=colors.CenteredNorm())
# ax9.contour(xm,ym,stdmassdens,levels=30,origin='lower',alpha=0.5)

xspockarc,yspockarc = np.linspace(xref[0],xref[1],1000),np.linspace(yref[0],yref[1],1000)
spockarc = np.array([[xspockarc[i],yspockarc[i]] for i in range(len(xspockarc))])
magnspock=[]
xccdens,yccdens=[],[]
for i in tqdm(range(1,41)):
	cccrosscount = 0
	lpot = np.genfromtxt('%s/lenspot%s_%s.txt'%(path,zsrc,i))
	deta,kap = magnmap(lpot,xrang,yrang)
	detmaginterp = RectBivariateSpline(xrang,yrang,deta.T)
	magnarc = 1/detmaginterp.ev(xspockarc,yspockarc)
	magnspock.append(magnarc)
	critcurve=ax2.contour(x,y,deta,levels=[0],colors='b',linestyles='dashed')
	xcc,ycc = [],[],
	for item in critcurve.collections:
		for i in item.get_paths():
			v = i.vertices
			xcrit = v[:, 0]
			ycrit = v[:, 1]
			xcc.append(xcrit)
			ycc.append(ycrit)
	xcc,ycc = np.concatenate(xcc),np.concatenate(ycc)
	xccdens.append(xcc)
	yccdens.append(ycc)
xccdens,yccdens = np.concatenate(xccdens),np.concatenate(yccdens)
# Calculate the point density
xy = np.vstack([xccdens,yccdens])
# ccnumdens = gaussian_kde(xy)(xy)
ccnumdens,xedges,yedges,imCC = ax2.hist2d(xccdens,yccdens,density=True,bins=[350,350],cmap='Greens')
areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
ccnumdens = ccnumdens*areas

# import seaborn as sns
# data = np.array([[xccdens[i],yccdens[i]] for i in range(len(xccdens))])
# df = pd.DataFrame(data, columns = ['X','Y']) 
# sns.kdeplot(data = df, shade=True, ax=ax9, x='X', y='Y',levels=100)

### Bootstrap? ###
# Create a flat copy of the density distribution
flat = ccnumdens.flatten()

# Then, sample an index from the 1D array with the
# probability distribution from the original array
sample_coords = np.random.choice(a=flat.size, p=flat,size = 10000000)

# Take this index and adjust it so it matches the original array
adjusted_index = np.unravel_index(sample_coords, ccnumdens.shape)
adjusted_index = np.array(list(zip(*adjusted_index)))
sampxcc,sampycc = adjusted_index.T[0],adjusted_index.T[1]
ccnumdensboot,xedgesboot,yedgesboot,imCCboot = ax2.hist2d(sampxcc,sampycc,density=True,bins=[350,350],cmap='Greens')

resampxcc,resampycc=[],[]
for i in tqdm(range(30)):
	resample = np.random.randint(low=1, high=41, size=(40,))
	newavglpot = np.zeros((gridsize+1,gridsize+1))
	for j in resample:
		lpot = np.genfromtxt('%s/lenspot%s_%s.txt'%(path,zsrc,j))
		newavglpot+=lpot
	newavglpot=newavglpot/len(resample)
	deta,kap = magnmap(newavglpot,xrang,yrang)
	critcurve=ax9.contour(x,y,deta,levels=[0],colors='b',linestyles='dashed',alpha=0.3)
	xcc,ycc = [],[],
	for item in critcurve.collections:
		for i in item.get_paths():
			v = i.vertices
			xcrit = v[:, 0]
			ycrit = v[:, 1]
			xcc.append(xcrit)
			ycc.append(ycrit)
	xcc,ycc = np.concatenate(xcc),np.concatenate(ycc)
	resampxcc.append(xcc)
	resampycc.append(ycc)
resampxcc,resampycc = np.concatenate(resampxcc),np.concatenate(resampycc)
bootnumdens,xedges,yedges,imboot = ax2.hist2d(resampxcc,resampycc,density=True,bins=[200,200],cmap='Greens')
areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
bootnumdens = bootnumdens*areas
ccnumdens = bootnumdens

im = ax9.imshow(ccnumdens.T,extent=(xedges[1],xedges[len(xedges)-1],yedges[1],yedges[len(yedges)-1]),aspect='auto',cmap='Greens',origin='lower',interpolation = "gaussian")
ax9.contour(xedges[1:],yedges[1:],ccnumdens.T,levels=5)
deta,kap = magnmap(avglpot,xrang,yrang)
avgCC = ax9.contour(x,y,deta,levels=[0],colors='r',linestyles='dashed')
xavgcc,yavgcc = [],[]
for item in avgCC.collections:
	for i in item.get_paths():
		v = i.vertices
		xcrit = v[:, 0]
		ycrit = v[:, 1]
		xavgcc.append(xcrit)
		yavgcc.append(ycrit)
ccnumdensinterp = RectBivariateSpline(xedges[1:],yedges[1:],ccnumdens)
ccdenstrace = ccnumdensinterp.ev(xavgcc,yavgcc)
# sc1 = ax9.scatter(xavgcc,yavgcc,marker='s',s=10,c=ccdenstrace,cmap='YlOrRd')
start,finish = 570,680
cc1,cc2 = 610,623
# ax9.plot(xavgcc[0][start:finish],yavgcc[0][start:finish],c='b',ls='--')
# ax9.plot(xavgcc[0][cc1],yavgcc[0][cc1],c='b',marker='s')
# ax9.plot(xavgcc[0][cc2],yavgcc[0][cc2],c='b',marker='s')
# plt.scatter(xrec,yrec,color='yellow',marker='^')
ax9.scatter(xref,yref,color='lime',label='Observed',marker='o',zorder=2,facecolors='none', edgecolors='lime')
ax9.scatter(xspockF,yspockF,color='deeppink',marker='^',zorder=2,alpha=0.5,label='Kelly+22')
ax9.scatter(xD21,yD21,color='cyan',marker='^',zorder=2,alpha=0.8,label='Yan+23')
# ax9.scatter(xD23,yD23,color='cyan',marker='^',alpha=1)
ax9.scatter(xD31,yD31,color='cyan',marker='^',zorder=2,alpha=0.8)
# plt.scatter(xspockd,yspockd,color='r',marker='x',label='Diego22')
ax9.scatter(xspockS,yspockS,color='orange',marker='^',zorder=2,alpha=0.8,label='Rodney+18')
ax9.scatter(xgal,ygal,color='r',marker='x',label='Cluster Members')
ax9.legend(loc=1,ncol=2,prop={'size':9})
# ax2.plot([-2.5,2.5],[8,8],lw=5,color='k')
# ax2.annotate('5"', # this is the text
# 			(0,8.5), # this is the point to label
# 			color='k') # horizontal alignment can be left, right or center	
ax9.set_xlim(0.0,10)
ax9.set_ylim(-1.0,9.0)
distgal = np.sqrt((xgal-gspockx)**2 + (ygal-gspocky)**2)
indgalsorted = np.argsort(distgal)
spockgalx,spockgaly = np.array([xgal[indgalsorted[0]],xgal[indgalsorted[3]]]),np.array([ygal[indgalsorted[0]],ygal[indgalsorted[3]]]) 
# plt.scatter(spockgalx[0],spockgaly[0],color='g',marker='x')
spockgalind = np.array([indgalsorted[0],indgalsorted[3]])
bcgnx,bcgny = 2.6059,12.533
bcgsx,bcgsy = -17.54226,-23.25065
distbcgn = np.sqrt((xgal-bcgnx)**2 + (ygal-bcgny)**2)
distbcgs = np.sqrt((xgal-bcgsx)**2 + (ygal-bcgsy)**2)
indbcgn,indbcgs = np.argsort(distbcgn)[0],np.argsort(distbcgs)[0]
BCG_Nx,BCG_Ny = xgal[indbcgn],ygal[indbcgn]
BCG_Sx,BCG_Sy = xgal[indbcgs],ygal[indbcgs]
plt.scatter(BCG_Nx,BCG_Ny,color='g',marker='x')
plt.scatter(BCG_Sx,BCG_Sy,color='g',marker='x')
# plt.contour(x,y,avgdeta,levels=[0],colors='r',linestyles='dashed')

divider = make_axes_locatable(ax9)
cax2=divider.append_axes("right", size="5%",pad=0.05)
cax2.yaxis.set_label_position("right")
cax2.yaxis.tick_right()
cbar=fig9.colorbar(im,label=r'CC Density',cax=cax2)
cbar.set_ticks([])

ax9.minorticks_on()
ax9.set_aspect('equal')
ax9.set_anchor('C')

ax9.invert_xaxis()

plt.show()

fig7,ax7=subplots()

ax7.plot(ccdenstrace[0][start:finish],color='b')
ax7.minorticks_on()
ax7.set_anchor('C')
ax7.set_ylabel(r'CC Density',fontsize=15,fontweight='bold')
ax7.set_yticklabels([])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax7.axvline(x=cc1-start,color='r',ls='--')
ax7.axvline(x=cc2-start,color='r',ls='--')

ax7.axhline(y=np.mean(ccdenstrace[0]),color='green',linestyle='-')
ax7.fill_between(np.arange(0,finish-start,1), np.mean(ccdenstrace[0])-np.std(ccdenstrace[0]), np.mean(ccdenstrace[0])+np.std(ccdenstrace[0]),color='green',alpha=0.5)

plt.show()

spockccdens = ccnumdensinterp.ev(xspockarc,yspockarc)
distarc = np.sqrt((xspockarc-xref[0])**2 + (yspockarc-yref[0])**2) # Arcsec distance along arc starting at leftmost image

xyan,yyan = np.array([xD21[0],xD21[1],xD31]),np.array([yD21[0],yD21[1],yD31])
transmags=[]
for i in range(len(magnspock)):
	transmags.append(interp1d(distarc,abs(magnspock[i]))(np.sqrt((xyan-xref[0])**2 + (yyan-yref[0])**2)))
transmags=np.array(transmags)

print('D21-S1:',np.mean(transmags.T[0]),'+/-',np.std(transmags.T[0]))
print('D21-S2:',np.mean(transmags.T[1]),'+/-',np.std(transmags.T[1]))
print('D31-S4:',np.mean(transmags.T[2]),'+/-',np.std(transmags.T[2]))

sys.exit()