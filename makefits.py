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
sigcrit_rescaled = ((c_light**2)/(4*np.pi*G))*(1/Ddpc)*(arcsec_pc**2) # Solar Mass per arsec^2 (Ds/Dds = 1)

def RADEC_to_ARCSEC(RA,Dec,zpra,zpdec):
	# Takes you from RA,Dec in observed degrees to arcsec with respect to given zero point zpra,zpdec

	# First convert RA,Dec to real RA,Dec in deg
	RA = (RA-zpra)*np.cos(Dec*rad_per_deg)
	Dec = Dec - zpdec

	# Next convert to arcsec
	x_arc,y_arc = RA*3600,Dec*3600

	return x_arc,y_arc
def ARCSEC_to_RADEC(xarc,yarc,zpra,zpdec):
	# Takes you from arcsec to RA,Dec with respect to given zero point zpra,zpdec

	Dec = (yarc/3600)+zpdec
	RA = ((xarc/3600)/np.cos(Dec*rad_per_deg))+zpra
	# RA = (xarc/3600)+zpra

	return RA,Dec

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
elif ((choice == 'dirsub15')):
	trans = 'sub15'
	sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,24,5.1,5.2,5.3,5.4,5.5,5.6,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,203,8,29,107,109,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,207,208,108,209,32,35,33,113,
					2.1,2.2,112,210.1,210.2,210.3,210.4,211]) # Sub 15
	ID = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=0,dtype='str')
	RAdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=2) # Image positions in RA degrees
	Decdeg = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=3) # Image positions in Dec degrees
	zs = np.loadtxt('MACSJ0416%s.txt'%trans,usecols=4)
	sourceplanes=np.unique(zs)
	xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec
	print(trans)
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
# avglpot = np.loadtxt('/Users/derek/Desktop/UMN/Research/MACSJ0416/lpotdiego23.txt',usecols=2).reshape(500,500)
# avgdeta = np.genfromtxt('%s/AVGdetA_%s.txt'%(path,zsrc)).T
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
lpot_unscaled = avglpot*(Dspc/Ddspc)
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

def deltatheta(xtest,ytest,xref,yref):
	xrec=[]
	yrec=[]
	for i in range(len(xref)):
		distim = np.sqrt((np.array(xtest)-xref[i])**2 + (np.array(ytest)-yref[i])**2)
		xrec.append(xtest[np.argmin(distim)])
		yrec.append(ytest[np.argmin(distim)])
	xrec,yrec = np.array(xrec),np.array(yrec)
	distrecon = np.sqrt((np.array(xrec)-xref)**2 + (np.array(yrec)-yref)**2)
	return xrec,yrec,np.mean(distrecon)
xrec,yrec,deltasep = deltatheta(xrec,yrec,xref,yref)
print('Mean Image Separation:',deltasep)

avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))/sigcrit_rescaled
stdmassdens = np.genfromtxt('%s/STDmassdens.txt'%(path))
xm,ym = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))
magnmatrix = hessian(avgtdsurf)
deta = magnmatrix[0][0]*magnmatrix[1][1] - magnmatrix[1][0]*magnmatrix[0][1]
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
	avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))/sigcrit
	massinterp = RectBivariateSpline(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]),avgmassdens)
	kapinterp = RectBivariateSpline(xrang,yrang,kap)
	# mref=massinterp.ev(x,y)
	# kref=kapinterp.ev(x,y)
	factor = 1.0#np.mean(mref/kref)#534526963912.9919#
	# print('Fuck',np.mean(mref/kref))
	kap = kap*factor
	gam1 = gam1*factor
	gam2 = gam2*factor
	detmag = (1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2)

	return detmag,kap
deta,kap = magnmap(avglpot,xrang,yrang)

im = ax9.imshow(avgmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='Greys',origin='lower')
# plt.imshow(deta,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='seismic',origin='lower',norm=colors.CenteredNorm())
ax9.contour(xm,ym,avgmassdens,levels=30,origin='lower',alpha=0.5,colors='purple')
ax9.contour(x,y,deta,levels=[0],colors='lightcoral',linestyles='dashed')
ax9.scatter(xref,yref,color='lime',label='Observed',marker='o',zorder=2,facecolors='none', edgecolors='lime')
ax9.scatter(xgal,ygal,color='r',marker='x',label='Cluster Members')
ax9.legend(loc=1,ncol=2,prop={'size':9})
# ax9.set_xlim(0.0,10)
# ax9.set_ylim(-1.0,9.0)
ax9.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax9.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')

divider = make_axes_locatable(ax9)
cax2=divider.append_axes("right", size="5%",pad=0.05)
cax2.yaxis.set_label_position("right")
cax2.yaxis.tick_right()
fig9.colorbar(im,label=r'$\Sigma / \Sigma_{crit}$',cax=cax2)

ax9.minorticks_on()
ax9.set_aspect('equal')
ax9.set_anchor('C')

ax9.invert_xaxis()

plt.show()

# 2D Interpolation
tdsurfinterp = RectBivariateSpline(xrang,yrang,avgtdsurf.T)
tdbase = tdsurfinterp.ev(xtest[1],ytest[1])
tdelays = []
for i in range(len(xtest)):
	tdelays.append(tdsurfinterp.ev(xtest[i],ytest[i]) - tdbase)
	#print(tdsurfinterp.ev(xrecqso[i],yrecqso[i]))
tdelays = np.array(tdelays)/86400 # in days

detainterp = RectBivariateSpline(xrang,yrang,deta.T)
# avgdetainterp = RectBivariateSpline(xrang,yrang,avgdeta.T)
detamag = 1/detainterp.ev(np.array(xref),np.array(yref))
print(detamag)

####### MAKE FITS #######
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

# Example: Create a 2D array (e.g., a 100x100 image)
# image_data = np.flip(avgmassdens,axis=1) # Kappa
image_data = np.flip(lpot_unscaled,axis=1) # Unscaled Lpot
# image_data = np.flip(deta,axis=1) # detA for magnification

# Define the WCS (World Coordinate System) parameters
# RA/Dec at the center of the image (e.g., RA = 10h, Dec = -20 degrees)
# RA0,Dec0=ARCSEC_to_RADEC(xm[250][250]-0.5*np.median(np.diff(xm[0])),ym[250][250]-0.5*np.median(np.diff(xm[0])),zeropointRA,zeropointDec) # Kappa
# RA0,Dec0=ARCSEC_to_RADEC(x[250][250]-np.median(np.diff(x[0])),y[250][250]+np.median(np.diff(x[0])),zeropointRA,zeropointDec) # Lpot
RA0,Dec0=ARCSEC_to_RADEC(x[250][250],y[250][250],zeropointRA,zeropointDec) # Lpot
# RA0,Dec0=ARCSEC_to_RADEC(x[250][250],y[250][250],zeropointRA,zeropointDec) # detA
ra_center = RA0 * u.deg  # Right Ascension (in degrees)
dec_center = Dec0 * u.deg  # Declination (in degrees)

# Define the pixel scale (arcseconds per pixel)
# pixel_scale = np.median(np.diff(xm[0])) * u.arcsec / u.pixel  # Kappa
pixel_scale = np.median(np.diff(x[0])) * u.arcsec / u.pixel  # Lpot

# Set the image size (in pixels)
naxis1, naxis2 = image_data.shape

# Create a WCS object
wcs = WCS(naxis=2)
# wcs.wcs.crpix = [249.5, 249.5]  # Set the reference pixel to the center of the image KAPPA
wcs.wcs.crpix = [251, 251]  # Set the reference pixel to the center of the image Lpot
# wcs.wcs.crpix = [251, 251]  # Set the reference pixel to the center of the image detA
wcs.wcs.crval = [ra_center.to(u.deg).value, dec_center.to(u.deg).value]  # RA/Dec of reference pixel
wcs.wcs.cdelt = np.array([-pixel_scale.value/3600, pixel_scale.value/3600])  # Pixel scale deg/pix (negative for RA)
wcs.wcs.cunit = ['deg', 'deg']  # Units for RA/Dec are degrees
wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Projection type (TAN = gnomonic)

# Create a primary HDU (Header Data Unit) for the FITS file
header = wcs.to_header()  # Convert WCS to FITS header
hdu = fits.PrimaryHDU(data=image_data, header=header)

# Write the FITS file
fitspath = '/Users/derek/Desktop/UMN/Research/MACSJ0416/imagefits'
fits_file = '%s/lpot.fits'%fitspath
hdu.writeto(fits_file, overwrite=True)

from astropy import wcs
from astropy.wcs import WCS
with fits.open('%s/lpot.fits'%fitspath) as hdu:
	data=hdu[0].data
	muwcs2 = wcs.WCS(hdu[0].header)
	data = np.where(np.isnan(data)==False,data,0)
pixcoords = np.array([[i,j] for i in tqdm(range(data.shape[1])) for j in range(data.shape[0])])
ra,dec = muwcs2.all_pix2world(pixcoords,1).T
xb,yb = RADEC_to_ARCSEC(ra,dec,zeropointRA,zeropointDec) 
xb,yb = xb.reshape(data.shape[0],data.shape[1]),yb.reshape(data.shape[0],data.shape[1])
xblow,yblow,xbhigh,ybhigh = xb.min(),yb.min(),xb.max(),yb.max()
nbx = (xbhigh-xblow)/data.shape[0]
nby = (ybhigh-yblow)/data.shape[1]
xbrang = np.linspace(xblow,xbhigh,data.shape[1])#np.arange(xblow,xbhigh,nbx)
ybrang = np.linspace(yblow,ybhigh,data.shape[0])#np.arange(yblow,ybhigh,nby)
muinterp = RectBivariateSpline(ybrang,xbrang,np.rot90(data.T,3)) # Magnification
xmu,ymu = np.meshgrid(np.linspace(xlow,xhigh,avglpot.shape[0]),np.linspace(ylow,yhigh,avglpot.shape[1]))
xmurang = xmu[0]
ymurang = ymu.T[0]
lpot2 = np.array([muinterp.ev(xmurang[i],ymurang[j]) for i in tqdm(range(len(xmurang))) for j in range(len(ymurang))]).reshape(gridsize+1,gridsize+1)
detatest,kaptest = magnmap(lpot2*(Ddspc/Dspc),xmurang,ymurang)

plt.figure()
plt.contour(xbrang,ybrang,detatest,levels=[0],colors='r',linestyles='dashed')
plt.contour(x,y,deta,levels=[0],colors='b',linestyles='dashed')
plt.scatter(xtest,ytest,color='b',marker='s')
x,y = np.meshgrid(xbrang,ybrang)
thetabeta = ((x-xs)/206264.80624709636)*((x-xs)/206264.80624709636) + ((y-ys)/206264.80624709636)*((y-ys)/206264.80624709636)
fitstdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - lpot2.T*(Ddspc/Dspc))
xfits,yfits = fims2(x,y,fitstdsurf)
plt.scatter(xfits,yfits,color='r',marker='s')

sys.exit()