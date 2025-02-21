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

# ########### USE THESE FOR ORIGINAL MODEL WITH BERGAMINI+23 ID
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

# print(sourceID)
# src = input('Which Source: ')
# srcind = np.argwhere(sourceID==float(src))[0][0]
# zsrc = sourcez[srcind][0]

############ USE THESE FOR V2 MODEL WITH RIHTARSIC+24 ID
choice = 'CANUCS'
path = '/Users/derek/Desktop/UMN/Research/MACSJ0416/%s'%choice #CHANGE PATH NAME FOR DIFFERENT RUNS 
zeropointRA,zeropointDec = 64.03730721518987,-24.070971485232068 # Some random zero point

fitspath = '/Users/derek/Desktop/UMN/Research/MACSJ0416/imagefits'
idriht = np.loadtxt('%s/images_rihtarsic24.dat'%fitspath,usecols=0,dtype=str)
zsriht = np.loadtxt('%s/images_rihtarsic24.dat'%fitspath,usecols=6)
RAdeg = np.loadtxt('%s/images_rihtarsic24.dat'%fitspath,usecols=1)
Decdeg = np.loadtxt('%s/images_rihtarsic24.dat'%fitspath,usecols=2)
xarc,yarc = RADEC_to_ARCSEC(RAdeg,Decdeg,zeropointRA,zeropointDec) # image positions in arcsec
IDCANUCS = np.array([idriht[i].split('.')[0] for i in range(len(idriht))])
idindex = np.unique(IDCANUCS,return_index=True)[1]
IDCANUCS = np.array([IDCANUCS[index] for index in sorted(idindex)])
sourceID = IDCANUCS

# This step now done in dataprocess v2
imx = np.load('%s/imx.npy'%path,allow_pickle=True)
imy = np.load('%s/imy.npy'%path,allow_pickle=True)
sourcez = np.load('%s/sourcez.npy'%path,allow_pickle=True)
xarc,yarc = np.concatenate(imx),np.concatenate(imy)
# KEY: imx[sourceID][image]

print(sourceID)
idchoice = input('Source ID: ')
srcind=np.argwhere(sourceID==idchoice)[0][0] # index in idindex
zsrc = sourcez[srcind][0]
print(zsrc)

####################################################################

# Kelly et. al. 22 (F1 and F2)
RAF,DecF = np.array([64.0388904,64.0386110]),np.array([-24.0701557,-24.0699715]) # Spock F1 and F2 from Kelly22
xspockF,yspockF = RADEC_to_ARCSEC(RAF,DecF,zeropointRA,zeropointDec)

# RAspock,Decspock = np.array([64.03847,64.03889,64.03874,64.03836]),np.array([-24.06984,-24.07017,-24.07004,-24.06978]) # From Yan23
# xspock,yspock = RADEC_to_ARCSEC(RAspock,Decspock,zeropointRA,zeropointDec)
RAD21,DecD21 = np.array([64.03847,64.03889]),np.array([-24.06984,-24.07017]) # D21-S1 and D21-S2 from Yan23
xD21,yD21 = RADEC_to_ARCSEC(RAD21,DecD21,zeropointRA,zeropointDec)
xD31,yD31 = RADEC_to_ARCSEC(64.03836, -24.06978,zeropointRA,zeropointDec)
xD23,yD23 = RADEC_to_ARCSEC(64.03874, -24.07004,zeropointRA,zeropointDec)

RAdieg,Decdieg = np.array([64.0393337,64.0383658]),np.array([-24.0704250,-24.0697531])
xspockd,yspockd = RADEC_to_ARCSEC(RAdieg,Decdieg,zeropointRA,zeropointDec) # From Diego 22

RAS,DecS= np.array([64.038565,64.038998]),np.array([-24.069939,-24.070241])
xspockS,yspockS = RADEC_to_ARCSEC(RAS,DecS,zeropointRA,zeropointDec) # From Rodney 18 (S1 and S2)

xmoth,ymoth = RADEC_to_ARCSEC(64.03676, -24.06625,zeropointRA,zeropointDec) # Mothra Yan23
xwar,ywar = RADEC_to_ARCSEC(64.0362850,-24.0674847,zeropointRA,zeropointDec) # Warhol Chen19

# Mean Source Positions from backprojecting
xsmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=0)
ysmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=1)
# xsmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=1)
# ysmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=2)
Dspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=3)
Ddspc_all = np.loadtxt('%s/meansourcepos.txt'%path,usecols=4)
# See Kelly et. al. 2022
gspockx,gspocky = np.array([imx[6][0]+1.346153846153845]),np.array([imy[6][0]+3.384615384615387])


# Get Lens Potential
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
				if intersection.geom_type == 'MultiPoint':
					xrec.append(np.array([float(intersection.geoms[i].x) for i in range(len(intersection.geoms))]))
					yrec.append(np.array([float(intersection.geoms[i].y) for i in range(len(intersection.geoms))]))
				else:
					xrec.append([float(intersection.x)])
					yrec.append([float(intersection.y)])
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

distrecon = np.sqrt((np.array(xrec)-xref)**2 + (np.array(yrec)-yref)**2)
print('Mean Image Separation:',np.mean(distrecon))

avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))/sigcrit
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


# # ## Using C
# wavelpot = np.loadtxt('wavetest.txt',usecols=3).reshape((gridsize,gridsize))/(1e9*1e9)
# detawaveC,kapwaveC = magnmap(wavelpot.T,xm[0],ym.T[0])

# ## Using Python
# wavelpot = np.genfromtxt('wavetestpy.txt')
# detawavepy,kapwavepy = magnmap(wavelpot,xrang,yrang)


im = ax9.imshow(avgmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='Greys',origin='lower')
# plt.imshow(deta,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='seismic',origin='lower',norm=colors.CenteredNorm())
ax9.contour(xm,ym,avgmassdens,levels=30,origin='lower',alpha=0.5,colors='purple')
# ax9.contour(x,y,avgtdsurf,levels=100,origin='lower',alpha=0.5,colors='purple')
ax9.contour(x,y,deta,levels=[0],colors='lightcoral',linestyles='dashed')
# ax9.contour(xm,ym,detawaveC,levels=[0],colors='b',linestyles='dashed')
# ax9.contour(x,y,detawavepy,levels=[0],colors='r',linestyles='dashed')
# plt.scatter(xrec,yrec,color='yellow',marker='^')
ax9.scatter(xref,yref,color='lime',label='Observed',marker='o',zorder=2,facecolors='none', edgecolors='lime')
ax9.scatter(xspockF,yspockF,color='deeppink',marker='^',alpha=0.5,label='Kelly+22')
ax9.scatter(xD21,yD21,color='cyan',marker='^',alpha=0.5,label='Yan+23')
# ax9.scatter(xD23,yD23,color='cyan',marker='^',alpha=1)
ax9.scatter(xD31,yD31,color='cyan',marker='^',alpha=0.5)
# plt.scatter(xspockd,yspockd,color='r',marker='x',label='Diego22')
ax9.scatter(xspockS,yspockS,color='orange',marker='^',alpha=0.5,label='Rodney+18')
# plt.scatter(xmoth,ymoth,color='g',marker='>',label='Mothra',alpha=0.5)
plt.scatter(xwar,ywar,color='g',marker='>',label='Warhol',alpha=0.5)
# plt.scatter(gspockx,gspocky,color='y',marker='x',label='Gal z=0.4067')
ax9.scatter(xgal,ygal,color='r',marker='x',label='Cluster Members')
ax9.legend(loc=1,ncol=2,prop={'size':9})
# ax2.plot([-2.5,2.5],[8,8],lw=5,color='k')
# ax2.annotate('5"', # this is the text
# 			(0,8.5), # this is the point to label
# 			color='k') # horizontal alignment can be left, right or center	
ax9.set_xlim(0.0,10)
ax9.set_ylim(-1.0,9.0)
ax9.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax9.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
distgal = np.sqrt((xgal-gspockx)**2 + (ygal-gspocky)**2)
indgalsorted = np.argsort(distgal)
spockgalx,spockgaly = np.array([xgal[indgalsorted[0]],xgal[indgalsorted[3]]]),np.array([ygal[indgalsorted[0]],ygal[indgalsorted[3]]]) 
ax9.text(spockgalx[0]-0.1,spockgaly[0]+0.1,'Spock-N',color='r',fontsize='x-small')
ax9.text(spockgalx[1]-0.1,spockgaly[1]+0.1,'Spock-S',color='r',fontsize='x-small')
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
fig9.colorbar(im,label=r'$\Sigma / \Sigma_{crit}$',cax=cax2)

ax9.minorticks_on()
ax9.set_aspect('equal')
ax9.set_anchor('C')

ax9.invert_xaxis()

def region(regionx,regiony,regionsize):

	# regionx is x coordinate of center of circular region in arcsec
	# regiony is y coordinate of center of circular region in arcsec
	# regionsize is radius of circular region in arcsec

	dist = np.sqrt((x-regionx)**2 + (y-regiony)**2)
	densreginit = np.array([avgmassdens[i][j]*(stepx**2) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	regpos = np.array([(i,j) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	for i,item in enumerate(densreginit):
		if item == max(densreginit):
			xregpos,yregpos = regpos[i][0],regpos[i][1]
			break
	xclump = x[xregpos][yregpos] # Peak of Clump x
	yclump = y[xregpos][yregpos] # Peak of Clump y

	# xclump,yclump = regionx,regiony
	phi = np.linspace(0, 2*np.pi, 100)
	x1 = regionsize*np.cos(phi) + xclump
	x2 = regionsize*np.sin(phi) + yclump
	ax2.errorbar(x1,x2,color='g',label='South-East Clump')
	# ax1.errorbar([regionx],[regiony],color='b',marker='*')
	# ax1.errorbar([xclump],[yclump],color='g',marker='*')

	dist = np.sqrt((x-xclump)**2 + (y-yclump)**2)
	densregion = np.array([(avgmassdens[i][j]-5e10)*(stepx**2) for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	massregion = sum(densregion)
	stddensregion = np.array([((stdmassdens[i][j])*(stepx**2))**2 for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	stdmassregion = np.sqrt(sum(stddensregion))

	return xclump,yclump,massregion,stdmassregion
xclump,yclump,massclump,stdmassclump = region(spockgalx[0],spockgaly[0],10*1.88/arcsec_kpc) # spock N
print('Mass in Region (Solar Masses): ',massclump ,'+/-',stdmassclump)
xclump,yclump,massclump,stdmassclump = region(spockgalx[1],spockgaly[1],10*1.08/arcsec_kpc) # spock S
# xclump,yclump,massclump = region(BCG_Nx,BCG_Ny,8.81/arcsec_kpc)
print('Mass in Region (Solar Masses): ',massclump ,'+/-',stdmassclump)

plt.show()

for i in range(len(xspockF)):
	print('F1/F2:',xspockF[i],yspockF[i],RAF[i],DecF[i])
for i in range(len(xD21)):
	print('D21-S1/S2:',xD21[i],yD21[i],RAD21[i],DecD21[i])
for i in range(len(xspockS)):
	print('S1/S2:',xspockS[i],yspockS[i],RAS[i],DecS[i])

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
# avgdetamag = 1/avgdetainterp.ev(np.array(xref),np.array(yref))
# print(avgdetamag)

# psix = np.gradient(avglpot.T)[0]/(nx/206264.80624709636)
# psiy = np.gradient(avglpot.T)[1]/(nx/206264.80624709636)
# psixx = np.gradient(psix)[0]/(nx/206264.80624709636)
# psiyy = np.gradient(psiy)[1]/(nx/206264.80624709636)
# psixy = np.gradient(psix)[1]/(nx/206264.80624709636)
# psiyx = np.gradient(psiy)[0]/(nx/206264.80624709636)

# # Idk where the factor comes from but it is needed
# kap = 0.5*(psixx + psiyy)
# gam1 = 0.5*(psixx - psiyy)
# gam2 = psixy
# # avgmassdens = avgmassdens/sigcrit
# massinterp = RectBivariateSpline(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]),avgmassdens)
# kapinterp = RectBivariateSpline(xrang,yrang,kap)
# mref=massinterp.ev(np.array(xref),np.array(yref))
# kref=kapinterp.ev(np.array(xref),np.array(yref))
# factor = np.mean(mref/kref)
# kap = kap*factor
# gam1 = gam1*factor
# gam2 = gam2*factor
# detmag = (1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2)
# magmat = np.array([[[1-kap-gam1],[-gam2]],[[-gam2],[1-kap+gam1]]])
# detmat = magmat[0][0]*magmat[1][1] - magmat[0][1]*magmat[1][0]
# sys.exit()
# def getlpot(kappa,xl,yl,xk,yk):
# 	# xl,yl form the x and y array of psi; is not same shape as kappa
# 	# xk,yk form the x and y array of kappa
# 	# psi = np.empty([1,100]) # initialize array
# 	dxi,dyi = np.diff(xk).mean()/206264.80624709636,np.diff(yk.T).mean()/206264.80624709636
# 	# for i in tqdm(range(1)):
# 	# 	for j in range(100):
# 	# 		psi[i][j] = sum(kappa*np.log(np.sqrt((xl[i]-xk)**2 + (yl[j]-yk)**2)+ 1e-20)*dxi*dyi)

# 	# kappa,xk,yk = kappa.flatten(),xk.flatten(),yk.flatten()
# 	# psi = np.array([sum(kappa*np.log(np.sqrt((xl[i]-xk)**2 + (yl[j]-yk)**2) + 1e-10)*dxi*dyi) for i in tqdm(range(len(xl))) for j in range(len(yl))])
# 	psi = np.array([np.sum(kappa*np.log(np.sqrt((xl[i]-xk)**2 + (yl[j]-yk)**2) + 1e-20)*dxi*dyi) for i in tqdm(range(len(xl))) for j in range(len(yl))])

# 	return (1/np.pi)*psi.reshape(len(xl),len(yl))

# testlpot = getlpot(avgmassdens,xrang,yrang,x,y)
# # testlpot = getlpot(avgkap,xrang,yrang,x,y)
# np.savetxt('waveDMlpot.txt',testlpot)

# file = open(f'{path}/kappamothra.txt','w')
# for i in tqdm(range(len(x.flatten()))):
# 	file.writelines('%f  %f  %f \n'%(x.flatten()[i],y.flatten()[i],avgmassdens.flatten()[i]))
# file.close()

# magns=[]
# for i in tqdm(range(1,41)):
# 	lpot = np.genfromtxt('%s/lenspot%s_%s.txt'%(path,zsrc,i))
# 	deta,kap = magnmap(lpot,xrang,yrang)
# 	detmaginterp = RectBivariateSpline(xrang,yrang,deta.T)
# 	magnarc = 1/detmaginterp.ev(xwar,ywar)
# 	magns.append(magnarc)
# magns=np.array(magns)

sys.exit()
# ###### Refined Spock GRID #####
avglpot = np.genfromtxt('%s/AVGlenspotMODEL_%s.txt'%(path,zsrc))
gridsize=1000
xlow,ylow,xhigh,yhigh = 1.5,0.5,8.5,7.5 # Spock
nx = (xhigh-xlow)/gridsize # Effectively arcsec/pixel
ny = (yhigh-ylow)/gridsize
stepx=(xhigh-xlow)/avglpot.shape[0]
stepy=(yhigh-ylow)/avglpot.shape[1]
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
x,y = np.meshgrid(xrang,yrang)

thetabeta = ((x-xs)/206264.80624709636)*((x-xs)/206264.80624709636) + ((y-ys)/206264.80624709636)*((y-ys)/206264.80624709636)
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
	# avgmassdens = np.genfromtxt('%s/AVGmassdens_MODEL.txt'%(path))/sigcrit
	# massinterp = RectBivariateSpline(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]),avgmassdens)
	# kapinterp = RectBivariateSpline(xrang,yrang,kap)
	# mref=massinterp.ev(x,y)
	# kref=kapinterp.ev(x,y)
	factor = 1.0#np.mean(mref/kref)#534526963912.9919#
	# print('Fuck',np.mean(mref/kref))
	kap = kap*factor
	gam1 = gam1*factor
	gam2 = gam2*factor
	detmag = (1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2)

	return detmag,kap
detmag,avgkap = magnmap(avglpot,xrang,yrang)
# Magnification
fig3,ax3=subplots()
im3 = ax3.imshow(detmag,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='BrBG',origin='lower',norm=colors.CenteredNorm())
# divider = make_axes_locatable(ax3)
# cax3=divider.append_axes("right", size="5%",pad=0.000005)
# cax3.yaxis.set_label_position("right")
# cax3.yaxis.tick_right()
fig3.colorbar(im3,label=r'$1/\mu$',pad=0.02)
critcurve = ax3.contour(x,y,detmag,levels=[0],colors='lightcoral',linestyles='dashed')
# for item in critcurve.collections:
# 	for i in item.get_paths():
# 		v = i.vertices
# 		xcrit = v[:, 0]
# 		ycrit = v[:, 1]
# 		ax3.scatter(xcrit,ycrit)
# ax3.contour(x,y,deta,levels=[0],colors='r',linestyles='dashed')
# ax3.scatter(xref,yref,color='magenta',marker='^',label='Observed',s=55)
# ax3.scatter(xtest,ytest,color='k',marker='*',label='Recon.',s=55)
# ax3.scatter(xspockF,yspockF,color='b',marker='+',label='F1 & F2',alpha=0.5,s=55)
# ax3.scatter(xD21,yD21,color='cyan',marker='d',label='D21-S1 & D21-S2',alpha=0.5,s=55)
# ax3.scatter(xspockS,yspockS,color='g',marker='>',label='S1 & S2',alpha=0.5,s=55)

ax3.scatter(xref,yref,color='lime',label='Observed',marker='o',zorder=2,facecolors='none', edgecolors='lime')
ax3.scatter(xtest,ytest,color='y',label='Reconstructed',marker='o',zorder=2,facecolors='none', edgecolors='y')
ax3.scatter(xspockF,yspockF,color='deeppink',marker='^',alpha=0.5,label='Kelly+22')
ax3.scatter(xD21,yD21,color='cyan',marker='^',alpha=0.5,label='Yan+23')
ax3.scatter(xD31,yD31,color='cyan',marker='^',alpha=0.5)
ax3.scatter(xspockS,yspockS,color='orange',marker='^',alpha=0.5,label='Rodney+18')
# plt.scatter(xwar,ywar,color='g',marker='>',label='Warhol',alpha=0.5)
ax3.minorticks_on()
ax3.set_aspect('equal')
ax3.set_anchor('C')
ax3.legend()
# plt.xlim(xref[0]-8,xref[0]+6)
# plt.ylim(yref[0]-6,yref[0]+8)
xlow,ylow,xhigh,yhigh = 1.5,0.5,8.5,7.5 # Spock
plt.xlim(xlow+0.1,xhigh-0.1)
plt.ylim(ylow+0.1,yhigh-0.1)
xspockarc,yspockarc = np.linspace(xref[0],xref[1],10000),np.linspace(yref[0],yref[1],10000)
# ax3.errorbar(xspockarc,yspockarc)

# const=((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv

# tauxx=np.gradient(np.gradient(avgtdsurf)[0])[0]
# tauyy=np.gradient(np.gradient(avgtdsurf)[1])[1]
# tauxy=np.gradient(np.gradient(avgtdsurf)[0])[1]
# tauyx=np.gradient(np.gradient(avgtdsurf)[1])[0]
# Adet = (tauxx*tauyy - tauxy*tauyx)/(const**2)
ax3.invert_xaxis()
plt.show()
sys.exit()

detmaginterp = RectBivariateSpline(xrang,yrang,detmag.T)
magnarc = 1/detmaginterp.ev(xspockarc,yspockarc)
print(1/detmaginterp.ev(np.array(xref),np.array(yref)))
distarc = np.sqrt((xspockarc-xref[0])**2 + (yspockarc-yref[0])**2) # Arcsec distance along arc starting at leftmost image
fig4,ax4=subplots()
ax4.errorbar(distarc,abs(magnarc),color='b',label='FF00')
ax4.minorticks_on()
ax4.set_ylabel(r'$\mu$',fontsize=15,fontweight='bold')
ax4.set_xlabel(r'r [arcsec]',fontsize=15,fontweight='bold')
ax4.set_yscale('log')

ax4.axvline(np.sqrt((xspockS[0]-xref[0])**2 + (yspockS[0]-yref[0])**2),color='seagreen',ls='--')
ax4.axvline(np.sqrt((xspockF[0]-xref[0])**2 + (yspockF[0]-yref[0])**2),color='darkorange',ls='--')
ax4.axvline(np.sqrt((xD21[0]-xref[0])**2 + (yD21[0]-yref[0])**2),color='darkorchid',ls='--')
ax4.axvline(np.sqrt((xspockS[1]-xref[0])**2 + (yspockS[1]-yref[0])**2),color='seagreen',ls='--',label='S1 & S2')
ax4.axvline(np.sqrt((xspockF[1]-xref[0])**2 + (yspockF[1]-yref[0])**2),color='darkorange',ls='--',label='F1 & F2')
ax4.axvline(np.sqrt((xD21[1]-xref[0])**2 + (yD21[1]-yref[0])**2),color='darkorchid',ls='--',label='D21-S1 & D21-S2')
# ax4.axhline(40)

# ax4.errorbar(distarccompa,abs(magncompa),color='r',label='H-NFW')
ax4.legend()

# Calculation for limits on Luminosity of Spock Transients
xyans,yyans = np.concatenate([xD21,[xD31]]),np.concatenate([yD21,[yD31]])
magn_yans = abs(1/detmaginterp.ev(xyans,yyans)) # Magnifaction of Yan23 transients
magobs_jwst200W = unumpy.uarray(np.array([29.44,29.13,29.06]),np.array([0.17,0.17,0.17])) # D21-S1,D21-S2,D31-S4
fobs200W = 10**((magobs_jwst200W+48.60)/(-2.5)) # observed flux in cgs
fexp200W = (fobs200W/magn_yans)#/(1+zsrc) # Expected flux if no microlens -> UPPER LIMIT
magexp200W = -2.5*unumpy.log10(fexp200W) - 48.60 # Expected apparent magnitude
Dlumpc = Dspc*((1+zsrc)**2) # in pc
absmagexp200W = magexp200W - (5*np.log10(Dlumpc)) + 5 # Absolute magnitude expected
Msun200W = 4.93 # Absolute magnitude of the Sun in JWST F200W Willmer+18
lum = 10**(0.4*(Msun200W - absmagexp200W)) # Luminosity in Solar Luminositites

fig5,ax5=subplots()
absmagnarc = abs(magnarc)
distCC = distarc[np.argmax(absmagnarc)] # position of CC
distpix = distarc-distCC 
absmagnarc = absmagnarc[(distpix>=-0.25) & (distpix<=0.25)]
distpix = distpix[(distpix>=-0.25) & (distpix<=0.25)]


ax5.errorbar(distpix,absmagnarc,color='b',label='FF00')
ax5.minorticks_on()
ax5.set_ylabel(r'$\mu$',fontsize=15,fontweight='bold')
ax5.set_xlabel(r'r [arcsec]',fontsize=15,fontweight='bold')

def magn_model(d,mu_0):
	return mu_0/abs(d+0.0001)

pfit,results = curve_fit(magn_model,distpix,absmagnarc)
ax5.errorbar(distpix,magn_model(distpix,pfit[0]),color='r')
ax5.errorbar(distpix,magn_model(distpix,7.8),color='g')

def chi2(params,d,mu,err_mu):
	mu_0 = params 
	return sum(((mu - magn_model(d,mu_0))/(np.array(err_mu)))**2)

res = minimize(chi2,[10],args=(distpix,absmagnarc,np.zeros(len(absmagnarc))+0.1),method='Nelder-Mead',options = {'maxiter':10000})
print(res.x[0])

ax5.set_yscale('log')

plt.show()

fig6,ax6=subplots()
im6 = ax6.imshow(np.log10(abs(1/detmag)),extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='RdPu',origin='lower')
fig6.colorbar(im6,label=r'$log_{10}(|\mu|)$',pad=0.02)
milliCC = ax6.contour(x,y,detmag,levels=[0],colors='g',linestyles='dashed')

ax6.scatter(xref,np.array(yref),color='yellow',marker='^',label='Observed')
ax6.scatter(xtest,ytest,color='k',marker='*',label='Recon.',s=55)
ax6.scatter(xspockF,yspockF,color='b',marker='+',label='F1 & F2',alpha=0.5,s=55)
ax6.scatter(xD21,yD21,color='cyan',marker='d',label='D21-S1 & D21-S2',alpha=0.5,s=55)
ax6.scatter(xspockS,yspockS,color='g',marker='>',label='S1 & S2',alpha=0.5,s=55)
ax6.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax6.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')

ax6.minorticks_on()
ax6.set_aspect('equal')
ax6.set_anchor('C')
xlow,ylow,xhigh,yhigh = 1.5,0.5,8.5,7.5 # Spock
plt.xlim(xlow+0.1,xhigh-0.1)
plt.ylim(ylow+0.1,yhigh-0.1)

ax6.legend(loc=2,title='',ncol=2,prop={'size':9})
ax6.invert_xaxis()
plt.show()