import grale.lenses as lenses
import grale.cosmology as cosmology
import grale.images as images
import grale.util as util
import grale.plotutil as plotutil
from grale.constants import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import *
from scipy import *
import os
import itertools
import math as maths
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm as tqdm
from shapely import geometry
plt.ion()
plt.rcParams['legend.numpoints']=1
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 5
############## CONSTANTS ######################
pc_per_m = 3.24078e-17 # pc in a m
c_light = 299792.458 # speed of light in km/s
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2
rad_per_deg = np.pi/180 # radians per degree
pcconv = 30856775812800 # Number of km in a pc

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
choice = input('Which? (dir00,dir11,dir12) ')
path = '/Users/derek/Desktop/UMN/Research/MACSJ0416/%s'%choice #CHANGE PATH NAME FOR DIFFERENT RUNS
bestgen = np.loadtxt('%s/bestfits.txt'%(path),usecols=0)
zeropointRA,zeropointDec = 64.03730721518987,-24.070971485232068 # Some random zero point

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
	imgList = images.readInputImagesFile("MACSJ0416points.txt", True) 
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
	imgList = images.readInputImagesFile("MACSJ0416points_subset15.txt", True) 
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
	imgList = images.readInputImagesFile("MACSJ0416points_%s.txt"%trans, True) 
	print(trans)
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

cosm = cosmology.Cosmology(0.7, 0.27, 0, 0.73)
cosmology.setDefaultCosmology(cosm)
zd = 0.396
Dd = cosm.getAngularDiameterDistance(zd)
Ddpc = Dd*pc_per_m
arcsec_pc = Ddpc*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec

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

	if (xint==[] or yint==[]):
		return np.array([]),np.array([])
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

####################################################################
############ FORWARD PROJECT TO GET RECONSTRUCTED IMAGES############
####################################################################

# # Mean Source Positions from backprojecting
# xsmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=1)
# ysmean = np.loadtxt('%s/meansourcepos.txt'%path,usecols=2)
# Optimized Source Positions from backprojecting
xsmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=0)
ysmean = np.loadtxt('%s/optimsourcepos.txt'%path,usecols=1)

#################### DEFINE GRID ############################
gridsize=500
xlow,ylow,xhigh,yhigh = -70,-70,71,71
nx = (xhigh-xlow)/gridsize
ny = (yhigh-ylow)/gridsize
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
thetas = np.array([[x,y] for x in xrang for y in yrang])*ANGLE_ARCSEC
x,y = np.meshgrid(xrang,yrang)
xrectot,yrectot = [],[] # All reconstructed images

### KEY ###
# x[i][j][k]: i = run number 1-40; j = which source; k = which source image

for j in range(len(sourceID)):
	print('Source: %s'%sourceID[j])
	zs = imgList[j]['z']
	Ds, Dds = cosm.getAngularDiameterDistance(zs), cosm.getAngularDiameterDistance(zd, zs) # in meters
	Dspc,Ddspc = Ds*pc_per_m,Dds*pc_per_m # in pc
	sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2
	imgpts = imgList[j]['imgdata'].getAllImagePoints()
	xref,yref = np.array(imx[j]),np.array(imy[j])
	avglpot = np.genfromtxt('%s/AVGlenspot_%s.txt'%(path,zs))
	thetabeta = ((x-xsmean[j])/206265.0)*((x-xsmean[j])/206265.0) + ((y-ysmean[j])/206265.0)*((y-ysmean[j])/206265.0)
	avgtdsurf = ((1+zd)/c_light)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - avglpot.T)
	try:
		xrecon,yrecon = fims2(x,y,avgtdsurf)
		print('Ref: ',xref,yref)
		print('Recon: ',xrecon,yrecon)
	except ValueError:
		xrecon,yrecon = fims(x,y,avgtdsurf)
		print('shapely failed, go manual')
		print(xref)
		print(xrecon)
		if len(xref)!=len(xrecon):
			print('DO IT MANUALLY')
			print('Ref: ',xref,yref)
			print('Recon: ',xrecon,yrecon)
			plt.figure()
			contourx=plt.contour(x,y,np.gradient(avgtdsurf)[0],levels=[0],origin='lower',colors='pink')
			contoury=plt.contour(x,y,np.gradient(avgtdsurf)[1],levels=[0],origin='lower')
			plt.scatter(xrecon,yrecon,color='b',label='Recon')
			plt.scatter(xref,yref,color='yellow',label='Ref')
			plt.legend()
			initcheck = input('OK? (Y/N): ')
			if initcheck == 'N':
				numim = input('How many images? ')
				xrecon,yrecon=[],[]
				for i in range(int(numim)):
					xpos = input('x image %s = '%i)
					ypos = input('y image %s = '%i)
					xrecon.append(float(xpos))
					yrecon.append(float(ypos))
				xrecon,yrecon = np.array(xrecon),np.array(yrecon)
			plt.close()

	xrectot.append(xrecon)
	yrectot.append(yrecon)


np.save('%s/xrecon_optim'%path,xrectot) # use np.load('run2_xforproj.npy',allow_pickle=True) to reload
np.save('%s/yrecon_optim'%path,yrectot) # use np.load('run2_yforproj.npy',allow_pickle=True) to reload
