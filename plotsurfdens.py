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
plt.ion()
rc('font', weight='bold')
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
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

# Need to Generalize this later
Ddpc,Dspc,Ddspc = 1110940809.0062053, 1655189095.108516, 855769832.5452876 # in pc at zs = 0.94(?)
arcsec_pc = Ddpc*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec
sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2
sigcrit_rescaled = ((c_light**2)/(4*np.pi*G))*(1/Ddpc)*(arcsec_pc**2) # Solar Mass per arsec^2 (Ds/Dds = 1)

#### RAW DATA
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
zeropointRA,zeropointDec = 64.03730721518987,-24.070971485232068 # Some random zero point
RAgal = np.loadtxt('MACSJ0416_clustermembers.txt',usecols=1) # Image positions of cluster members in RA degrees
Decgal = np.loadtxt('MACSJ0416_clustermembers.txt',usecols=2) # Image positions of cluster members in Dec degrees
xgal,ygal = RADEC_to_ARCSEC(RAgal,Decgal,zeropointRA,zeropointDec)
xbcgn,ybcgn,xbcgs,ybcgs = xgal[0],ygal[0],xgal[72],ygal[72]

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
# idchoice = input('Source ID: ')
# srcind=np.argwhere(IDCANUCS==idchoice)[0][0] # index in idindex
# zsrc = sourcez[srcind][0]
# print(zsrc)
sourceID = IDCANUCS

# This step now done in dataprocess v2
imx = np.load('%s/imx.npy'%path,allow_pickle=True)
imy = np.load('%s/imy.npy'%path,allow_pickle=True)
sourcez = np.load('%s/sourcez.npy'%path,allow_pickle=True)
xarc,yarc = np.concatenate(imx),np.concatenate(imy)
# KEY: imx[sourceID][image]

# Bounds of Window
xlow,ylow,xhigh,yhigh = -70,-70,71,71

avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))
stdmassdens = np.genfromtxt('%s/STDmassdens.txt'%(path))

stepx=(xhigh-xlow)/(avgmassdens.shape[0])
stepy=(yhigh-ylow)/(avgmassdens.shape[1])
# x,y = np.meshgrid(np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy))
x,y = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))

# CREATE PLOT
fig1,ax1=subplots(1,figsize=(24,24),sharex=False,sharey=False,facecolor='w', edgecolor='k')

avgmassdens = avgmassdens/sigcrit
# avgmassdens = np.log10(avgmassdens)
# im = ax1.imshow(np.log10(avgmassdens),extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower')
im = ax1.imshow(avgmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='Greys',origin='lower')

# contourmass=ax1.contour(x,y,np.log10(avgmassdens),levels=40,origin='lower',alpha=1,linewidths=0.75)
contourmass=ax1.contour(x,y,avgmassdens,levels=np.arange(0,1.4,0.1),origin='lower',colors='purple',alpha=0.25,zorder=2)
# ax1.clabel(contourmass, inline=True)

xrecon = np.load('%s/xrecon_optim.npy'%(path),allow_pickle=True)
yrecon = np.load('%s/yrecon_optim.npy'%(path),allow_pickle=True)
ax1.scatter(np.concatenate(xrecon),np.concatenate(yrecon),color='r',label='Reconstructed',zorder=1,marker='^',alpha=0.5,s=70)

divider = make_axes_locatable(ax1)
ax1.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax1.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax1=divider.append_axes("right", size="5%",pad=0.25)

cax1.yaxis.set_label_position("right")
cax1.yaxis.tick_right()

# fig1.colorbar(im,label=r'Surface Mass Density [M$_{\odot}$ arcsec$^{-2}$]',cax=cax1)
# fig1.colorbar(im,label=r'log$_{10}$($\Sigma / \Sigma_{crit}$)',cax=cax1)
fig1.colorbar(im,label=r'$\Sigma / \Sigma_{crit}$',cax=cax1)
ax1.minorticks_on()
ax1.set_aspect('equal')
ax1.set_anchor('C')
#plt.tight_layout()

ax1.grid(False)

ax1.tick_params(axis='x',labelsize='10')
ax1.tick_params(axis='y',labelsize='10')

RAimdieg = np.loadtxt('MACSJ0416points_Diego24.txt',usecols=1) # Image positions in RA degrees from Diego
Decimdieg = np.loadtxt('MACSJ0416points_Diego24.txt',usecols=2) # Image positions in Dec degrees from Diego
xdieg,ydieg = RADEC_to_ARCSEC(RAimdieg,Decimdieg,zeropointRA,zeropointDec)

# Syst82 in Diego24 
xmyst1,ymyst1 = RADEC_to_ARCSEC(np.array([64.027115,64.027580,64.036530]),np.array([-24.078629,-24.079121,-24.084009]),zeropointRA,zeropointDec)
# Syst81 in Diego24 
xmyst2,ymyst2 = RADEC_to_ARCSEC(np.array([64.032738,64.033119,64.022079]),np.array([-24.081612,-24.081804,-24.071712]),zeropointRA,zeropointDec)

ax1.scatter(xarc,yarc,color='b',alpha=1,marker='.',label='Observed',zorder=1.5,s=15)
# ax1.scatter(xdieg,ydieg,color='cyan',alpha=1,marker='.',label='Observed (Diego24)',zorder=1.5,s=15)
# ax1.scatter(xmyst1,ymyst1,color='purple',alpha=1,marker='x',label='Observed (Diego24)',zorder=1.5,s=15)
# ax1.scatter(xmyst2,ymyst2,color='purple',alpha=1,marker='x',label='Observed (Diego24)',zorder=1.5,s=15)
ax1.errorbar(xgal,ygal,color='mediumspringgreen',marker='d',linestyle='None',label='Cluster Galaxy',alpha=0.5)
ax1.text(-12,-6,'M1',color='lime',alpha=0.9,size='x-small')
ax1.text(-20,-32,'M2',color='lime',alpha=0.9,size='x-small')


avgmassdens = avgmassdens*sigcrit
def region(regionx,regiony,regionsize):

	# regionx is x coordinate of center of circular region in arcsec
	# regiony is y coordinate of center of circular region in arcsec
	# regionsize is radius of circular region in arcesec
	phi = np.linspace(0, 2*np.pi, 100)
	dist = np.sqrt((x-regionx)**2 + (y-regiony)**2)
	densreginit = np.array([avgmassdens[i][j]*(stepx**2) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	regpos = np.array([(i,j) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	for i,item in enumerate(densreginit):
		if item == max(densreginit):
			xregpos,yregpos = regpos[i][0],regpos[i][1]
			break
	xclump = x[xregpos][yregpos] # Peak of Clump x
	yclump = y[xregpos][yregpos] # Peak of Clump y

	# xbkrd,ybkrd,rsize = 8.5,0.0,2.0 # For Spocks
	xbkrd,ybkrd,rsize = -19,-3,2.0 # For M1
	# xbkrd,ybkrd,rsize = -16,-38,2.0 # For M2

	distbkrd = np.sqrt((x-xbkrd)**2 + (y-ybkrd)**2)
	bkrddens = np.mean(np.array([avgmassdens[i][j] for i in tqdm(range(distbkrd.shape[0])) for j in range(distbkrd.shape[1]) if distbkrd[i][j] <= rsize]))
	x1 = rsize*np.cos(phi) + xbkrd
	x2 = rsize*np.sin(phi) + ybkrd
	ax1.errorbar(x1,x2,color='magenta',label='Bckgrd')

	# xclump,yclump = regionx,regiony
	x1 = regionsize*np.cos(phi) + xclump
	x2 = regionsize*np.sin(phi) + yclump
	ax1.errorbar(x1,x2,color='r',label='region')
	ax1.errorbar([regionx],[regiony],color='b',marker='*')
	ax1.errorbar([xclump],[yclump],color='r',marker='*')

	dist = np.sqrt((x-xclump)**2 + (y-yclump)**2)
	densregion = np.array([(avgmassdens[i][j]-bkrddens)*(stepx**2) for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	massregion = sum(densregion)
	stddensregion = np.array([((stdmassdens[i][j])*(stepx**2))**2 for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	stdmassregion = np.sqrt(sum(stddensregion))

	return xclump,yclump,massregion,stdmassregion
def circprof(xc,yc,binvals,avgmassdens):
	xbkrd,ybkrd,rsize = -19,-3,2.0 # For M1
	# xbkrd,ybkrd,rsize = -16,-38,2.0 # For M2
	distbkrd = np.sqrt((x-xbkrd)**2 + (y-ybkrd)**2)
	bkrddens = np.mean(np.array([avgmassdens[i][j] for i in tqdm(range(distbkrd.shape[0])) for j in range(distbkrd.shape[1]) if distbkrd[i][j] <= rsize]))
	distbins = [[] for x in range(len(binvals))] # Bins of circular profile
	for i in tqdm(range(len(x))):
		for j in range(len(y)):
			xpix = x[i][j]
			ypix = y[i][j]
			massdenspix = avgmassdens[i][j]-bkrddens
			dist = np.sqrt(((xpix-xc)**2 + (ypix-yc)**2)) # Distance from pixel to BCG
			if (dist>max(binvals)):
				distbins[nbins-1].append(massdenspix)
				continue
			for k in range(len(binvals)-1):
				if (dist>binvals[k] and dist<=binvals[k+1]):
					distbins[k].append(massdenspix)

	circprof = np.array([np.mean(np.array(bins)) for bins in distbins])
	std_circprof = np.array([np.std(np.array(bins)) for bins in distbins])

	return circprof,std_circprof
nbins=30
binvals = np.linspace(1,4,nbins) # Values of bins in arcsec
# densprof,stddensprof = circprof(-12.074148296593187,-5.292585170340686,binvals,avgmassdens) # M1
# densprof,stddensprof = circprof(-20.2685370741483,-32.13627254509018,binvals,avgmassdens) # M2
# xd,yd,mass,stdm1 = region(-12.074148296593187,-5.292585170340686,3) # M1
# xd,yd,mass,stdm2 = region(-20.2685370741483,-32.13627254509018,3) # M2
# ########## PLOT ##################
# logsigma = np.log10(np.where(densprof>0,densprof,1))[:-1]
# logr = np.log10(binvals)[:-1]
# from scipy.interpolate import splrep, splev
# f = splrep(logr, logsigma, k=5, s=3)
# logsigma = splev(logr,f)
# dkap= np.diff(logsigma)/np.diff(logr)
# xderiv = (logr[:-1] + logr[1:]) / 2
# drint = interp1d(dkap,xderiv)([-1])[0] # radius at dlog(kappa)/dr = -1 (isothermal)
# rcore = 10**drint # in arcsec
# print(rcore*arcsec_pc)
# clusternames = ['BCG-N','spock-N','spock-S','BCG-S']
# modelgalind = np.array([0,17,22,72])
# galposx,galposy = xgal[modelgalind],ygal[modelgalind]

# # Calculate the uncertainty in the slope
# slope_uncertainty = np.std(dkap)

# # Find the index where the slope is closest to -1
# closest_index = np.argmin(np.abs(dkap + 1))
# indices = np.where(np.abs(dkap + 1) < slope_uncertainty)[0]

# # Estimate x uncertainty
# x_value = xderiv[closest_index]
# x_uncertainty = (xderiv[indices].max() - xderiv[indices].min()) / 2
# print((10**x_uncertainty)*arcsec_pc)

# # Interpolate y value
# y_value = np.interp(x_value, logr, logsigma)

# # Estimate dy/dx at x_value
# dy_dx = np.interp(x_value, xderiv, dkap)

# # Estimate uncertainty in y_value
# y_uncertainty = abs(dy_dx) * x_uncertainty

# xclump,yclump,massclump,stdmassclump = region(-12.074148296593187,-5.292585170340686,rcore) # M1
# print('M1 : ', massclump , '+/-', stdmassclump)
# xclump,yclump,massclump,stdmassclump = region(-20.2685370741483,-32.13627254509018,rcore) # M2
# print('M2 : ', massclump , '+/-', stdmassclump)

# xclump,yclump,massclump,stdmassclump = region(xbcgn,ybcgn,200/arcsec_kpc) # BCG-N
# print('BCG-N : ', massclump , '+/-', stdmassclump)

# xclump,yclump,massclump,stdmassclump = region(xbcgs,ybcgs,200/arcsec_kpc) # BCG-N
# print('BCG-S : ', massclump , '+/-', stdmassclump)

# ax1.text(galposx[0]+10,galposy[0]-5,clusternames[0],color='cyan',alpha=0.9,size='x-small')
# ax1.text(galposx[3]-7.5,galposy[3]+5,clusternames[3],color='cyan',alpha=0.9,size='x-small')
# ax1.text(galposx[3],galposy[3],clusternames[3],color='cyan',alpha=0.9,size='x-small')
# ax1.set_xlim(galposx[3]-30,galposx[3]+30)
# ax1.set_ylim(galposy[3]-20,galposy[3]+30)
ax1.invert_xaxis()

# Sort images to make sure reconstructed images are in same order as observed!
xdel=[] # Delta x between recon and obs
ydel=[] # Delta y between recon and obs
zdel=[] # Corresponding redshifts
xextra,yextra = [],[]
count=0
for i in range(len(imx)):
	mask=[]
	for j in range(len(imx[i])):
		xref,yref = imx[i][j],imy[i][j]
		distim = np.sqrt((xrecon[i]-xref)**2 + (yrecon[i]-yref)**2) # Distances from obs. image to all recon. images
		# if (np.argmin(distim) in mask):
		# 	if ((xrecon[i][np.argsort(distim)[1]]-xref<100) and (yrecon[i][np.argsort(distim)[1]]-yref<100)):
		# 		xdel.append(xrecon[i][np.argsort(distim)[1]]-xref)
		# 		ydel.append(yrecon[i][np.argsort(distim)[1]]-yref)	
		# 		mask.append(np.argsort(distim)[1])	
		# 	else:
		# 		continue
		# else:
		# 	xdel.append(xrecon[i][np.argmin(distim)]-xref)
		# 	ydel.append(yrecon[i][np.argmin(distim)]-yref)
		# 	mask.append(np.argmin(distim))		
		xdel.append(xrecon[i][np.argmin(distim)]-xref)
		ydel.append(yrecon[i][np.argmin(distim)]-yref)
		mask.append(np.argmin(distim))
		zdel.append(sourcez[i][0])
	# print(mask,sourceID[i])
	if len(xrecon[i])>len(imx[i]):
		# print(sourceID[i])
		maskarray=np.ones(xrecon[i].shape, bool)
		maskarray[np.array(mask)]=False
		xextra.append(xrecon[i][maskarray])
		yextra.append(yrecon[i][maskarray])
		count+=1
xdel,ydel,zdel = np.array(xdel),np.array(ydel),np.array(zdel) # x and y displacement between observed and recon images
xextra,yextra = np.array(xextra,dtype=object),np.array(yextra,dtype=object)

# ax1.scatter(np.concatenate(xextra),np.concatenate(yextra),color='y',label='Extraneous',marker='.',s=15)
# ax1.scatter(xgal,ygal,color='y',marker='x',label='Cluster Members')
ax1.legend(loc=1,title='',ncol=2,prop={'size':9})
plt.show()

fig2,ax2=subplots(1,figsize=(8,12),sharex=False,sharey=False,facecolor='w', edgecolor='k')
# sc2 = ax2.scatter(xdel,ydel,marker='o',s=35,c=zdel,cmap='YlOrRd') 
sc2 = ax2.scatter(xdel,ydel,marker='o',s=35,color='r') 
ax2.set_ylabel(r'$\Delta$ y [arcsec]',fontsize=13,fontweight='bold')
ax2.set_xlabel(r'$\Delta$ x [arcsec]',fontsize=13,fontweight='bold')

# divider2 = make_axes_locatable(ax2)
# cax1=divider2.append_axes("right", size="5%",pad=0.25)
# fig2.colorbar(sc2,label=r'$z_s$',cax=cax1)
ax2.tick_params(axis='x',labelsize='10')
ax2.tick_params(axis='y',labelsize='10')
ax2.minorticks_on()
ax2.set_aspect('equal')
ax2.set_anchor('C')
ax2.grid(False)
ax2.axvline(x=0,color='blue',linestyle='--')
ax2.axhline(y=0,color='blue',linestyle='--')
ax2.set_aspect('equal')
ax2.set_anchor('C')
plt.show()

# Calculate RMS displacement of recon vs obs
Delta = np.sqrt(xdel**2 + ydel**2)
delrms = np.sqrt(np.mean(Delta**2))

fig3,ax3=subplots(1,figsize=(6,10),sharex=False,sharey=False,facecolor='w', edgecolor='k')
num_bins=50
ax3.hist(Delta,bins=num_bins,color='seagreen')
ax3.set_ylabel(r'Count',fontsize=13,fontweight='bold')
ax3.set_xlabel(r'$\Delta$ [arcsec]',fontsize=13,fontweight='bold')
ax3.tick_params(axis='x',labelsize='10')
ax3.tick_params(axis='y',labelsize='10')
ax3.minorticks_on()
ax3.set_anchor('C')
ax3.set_title(r'$\Delta_{RMS}$ = %s" '%round(delrms,4),fontsize=13,fontweight='bold')

plt.show()

# Start with a square Figure.
# fig = plt.figure(figsize=(7.25, 7.25))
fig, ax = plt.subplots(2,2,figsize=(7.25, 7.25),gridspec_kw={'height_ratios': [1, 4], 'width_ratios': [4, 1]})
ax[0][0].sharex(ax[1][0])
ax[1][1].sharey(ax[1][0])
ax[0][1].grid(False)
ax[0][1].axis('off')
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
import matplotlib.gridspec as gridspec
# gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
#                       left=0.1, right=0.9, bottom=0.1, top=0.9,
#                       wspace=0.05, hspace=0.05)
# gs = gridspec.GridSpec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
#                       left=0.1, right=0.9, bottom=0.1, top=0.9,
#                       wspace=0.05, hspace=0.05)
# Create the Axes.
# ax = fig.add_subplot(gs[1, 0])
# ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
# ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
ax[0][0].tick_params(axis="x", labelbottom=False)
ax[1][1].tick_params(axis="y", labelleft=False)
ax[0][0].set_ylabel('N',fontsize=13,fontweight='bold')
ax[1][1].set_xlabel('N',fontsize=13,fontweight='bold')
# now determine nice limits by hand:
binwidth = 0.05
xymax = max(np.max(np.abs(xdel)), np.max(np.abs(ydel)))
lim = (int(xymax/binwidth) + 1) * binwidth
bins = np.arange(-lim, lim + binwidth, binwidth)
ax[0][0].hist(xdel, bins=bins,color='seagreen')
ax[1][1].hist(ydel, bins=bins, orientation='horizontal',color='seagreen',)
# the scatter plot:
ax[1][0].scatter(xdel, ydel, marker='o',s=35,color='seagreen')
ax[1][0].scatter(np.mean(xdel), np.mean(ydel), marker='d',s=35,color='cyan',zorder=2)
ax[1][0].axvline(x=0,color='red',linestyle='--')
ax[1][0].axhline(y=0,color='red',linestyle='--')
ax[1][0].minorticks_on()
ax[1][0].set_ylabel(r'$\Delta$ y [arcsec]',fontsize=13,fontweight='bold')
ax[1][0].set_xlabel(r'$\Delta$ x [arcsec]',fontsize=13,fontweight='bold')
# ax[1][0].set_aspect('equal')
ax[1][0].set_anchor('C')
fig.tight_layout()

plt.show()