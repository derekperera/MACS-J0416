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
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
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

path = '/Users/derek/Desktop/UMN/Research/MACSJ0416/dir00' #CHANGE PATH NAME FOR DIFFERENT RUNS

# bestgen=[]
# for i in np.arange(10,50,10):
# 	bestgen.append(np.loadtxt('%s/beststeps%s_20Gen.txt'%(path,i),usecols=0))
# bestgen=np.array([int(i) for i in np.concatenate((bestgen))])
# bestgen = np.array([17]) #np.loadtxt('%s/beststeps1_20Gen.txt'%(path),usecols=0)# Best fitness subdivisions
bestgen = np.loadtxt('%s/bestfits.txt'%(path),usecols=0)

#### RAW DATA
# 88 Source ID names
sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,201,24,5.1,5.2,5.3,5.4,5.5,5.6,36,15.1,15.2,
					15.4,7,202.1,202.2,10,16.1,16.2,4,37,203,8,29,23,107,109,26,25,14.1,14.2,20.1,
					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
					19.1,19.2,19.3,103,106,31,110,101,207,208,108,209,21.1,21.2,21.3,32,35,33,113,
					102,2.1,2.2,112,210.1,210.2,210.3,210.4,211]) # FULL
# sourceID = np.array([12.1,12.2,12.3,12.4,12.5,12.6,13,24,5.1,5.2,5.3,5.4,5.5,5.6,15.1,15.2,
# 					15.4,7,202.1,202.2,10,16.1,16.2,4,203,8,29,107,109,25,14.1,14.2,20.1,
# 					20.3,20.4,20.5,20.6,20.7,1,28,3,204,11,9.1,9.2,9.3,30,27,6,205,18,34,17,104,105,
# 					19.1,19.2,19.3,103,106,31,110,207,208,108,209,32,35,33,113,
# 					2.1,2.2,112,210.1,210.2,210.3,210.4,211]) # Sub 15
ID = np.loadtxt('MACSJ0416.txt',usecols=0,dtype='str')
RAdeg = np.loadtxt('MACSJ0416.txt',usecols=2) # Image positions in RA degrees
Decdeg = np.loadtxt('MACSJ0416.txt',usecols=3) # Image positions in Dec degrees
zs = np.loadtxt('MACSJ0416.txt',usecols=4)
sourceplanes=np.unique(zs)
zeropointRA,zeropointDec =  np.mean(RAdeg),np.mean(Decdeg) # Some random zero point
RA = (RAdeg - zeropointRA)*np.cos(Decdeg*rad_per_deg) # Real RA in deg
Dec = Decdeg - zeropointDec # Real Dec in deg
xarc,yarc = RA*3600,Dec*3600 # image positions in arcsec

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

#################### DEFINE GRID ############################
gridsize=500
xlow,ylow,xhigh,yhigh = -70,-70,71,71
# xmoth,ymoth = -1.798733738507787,16.997346835444205
# xlow,ylow,xhigh,yhigh = xmoth-0.7,ymoth-0.7,xmoth+0.7,ymoth+0.7 # Mothra
# xlow,ylow,xhigh,yhigh = 1.5,0.5,8.5,7.5 # Spock
nx = (xhigh-xlow)/gridsize
ny = (yhigh-ylow)/gridsize
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
thetas = np.array([[x,y] for x in xrang for y in yrang])*ANGLE_ARCSEC
x,y = np.meshgrid(xrang,yrang)

imgList = images.readInputImagesFile("MACSJ0416points_CANUCS24.txt", True) 

##### GET ALL PLUMMERS FROM RUN ########
plummermass,plummerwidth,plummerx,plummery = [],[],[],[]
avgmassdens = np.zeros((gridsize,gridsize))
avgtdelay = np.zeros((gridsize,gridsize))
avglpot = np.array([np.zeros((gridsize,gridsize)) for i in range(len(sourceplanes))])
avgdetA = np.zeros((gridsize+1,gridsize+1))
massobj,tdelayobj,lpotobj,detaobj = [],[],[],[]
sys.exit()
sourceplanes=np.array([2.921]) # ID 107
rcores=[]
for i in range(1,41): # 1,41
	print(path)
	lensX = lenses.GravitationalLens.load("%s/best_step%s_%s.lensdata"%(path,i,int(bestgen[i-1])))

	# Get all lens potential for zs
	zs = sourceplanes[0]
	Ds, Dds =  cosm.getAngularDiameterDistance(zs), cosm.getAngularDiameterDistance(zd, zs) # in meters
	Dspc,Ddspc = Ds*pc_per_m,Dds*pc_per_m # in pc
	sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2
	lpotX = lensX.getProjectedPotential(Ds,Dds,thetas)
	lpot = lpotX.reshape(gridsize+1,gridsize+1)
	np.savetxt('%s/lenspot2.921_%s.txt'%(path,i),lpot)

	# # Average Surface Mass Density (kg/m^2)
	# surfdensplum = lensX.getSurfaceMassDensityMap((xlow*ANGLE_ARCSEC,ylow*ANGLE_ARCSEC),(xhigh*ANGLE_ARCSEC,yhigh*ANGLE_ARCSEC),gridsize+1,gridsize+1)
	# avgmassdens += surfdensplum*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2) # Solar Mass per square arcsec
	# massobj.append(surfdensplum*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2))
	# massdens = surfdensplum*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2)
	print('Run: %s'%i)

	# Average Time Delay Surface (s?) For the QSO 
	# tdelayX = lensX.getTimeDelay(zd,Ds,Dds,thetas,np.array([xs,ys])*ANGLE_ARCSEC)
	# tdsurf = tdelayX.reshape(gridsize,gridsize)
	# avgtdelay += tdsurf
	# tdelayobj.append(tdsurf)

	# # Average Lens Potential
	# lpotsrun=[]
	# count=0
	# for j in range(len(sourceplanes)):
	# 	zs = sourceplanes[j]
	# 	Ds, Dds =  cosm.getAngularDiameterDistance(zs), cosm.getAngularDiameterDistance(zd, zs) # in meters
	# 	Dspc,Ddspc = Ds*pc_per_m,Dds*pc_per_m # in pc
	# 	sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2

	# 	lpotX = lensX.getProjectedPotential(Ds,Dds,thetas)
	# 	lpot = lpotX.reshape(gridsize,gridsize)
	# 	# lpot = 0.5*(((x-xs)*ANGLE_ARCSEC)**2 + ((y-ys)*ANGLE_ARCSEC)**2) - lpot
	# 	avglpot[j] += lpot
	# 	lpotsrun.append(lpot)
	# 	# for z in range(len(sourcez)):
	# 	# 	if np.array(sourcez[z])[0] == zs:
	# 	# 		print(zs,sourceID[z],count)
	# 	# 		count+=1 
	# lpotobj.append(lpotsrun)

	# Average Shear 
	# alphavec=lensX.getAlphaVectorDerivatives(thetas)
	# alphaxx,alphayy,alphaxy = alphavec.T[0].reshape(gridsize,gridsize),alphavec.T[1].reshape(gridsize,gridsize),alphavec.T[2].reshape(gridsize,gridsize)
	# gamma1 = 0.5*(alphaxx-alphayy)
	# gamma2 = alphaxy
	# gammasquared = gamma1**2 + gamma2**2
	# kappa = 0.5*(alphaxx+alphayy)
	# deta = (1-kappa)**2 - gammasquared
	# avgdetA += deta

	# Average Magnification
	# count=0
	# zspock = 1.005
	# for j in range(len(sourceplanes)):
	# 	zs = sourceplanes[j]
	# 	if zs == zspock:
	# 		print(zs)
	# 		Ds, Dds =  cosm.getAngularDiameterDistance(zs), cosm.getAngularDiameterDistance(zd, zs) # in meters
	# 		Dspc,Ddspc = Ds*pc_per_m,Dds*pc_per_m # in pc
	# 		sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2

	# 		detaX = lensX.getInverseMagnification(Ds, Dds, thetas)
	# 		deta = detaX.reshape(gridsize+1,gridsize+1)
	# 		avgdetA += deta
	# 		for z in range(len(sourcez)):
	# 			if np.array(sourcez[z])[0] == zs:
	# 				print(zs,sourceID[z],count)
	# 				count+=1 

sys.exit()
avgmassdens = avgmassdens/40
# avgtdelay = avgtdelay/40
# avglpot = avglpot/40
# avgdetA = avgdetA/40

# massobj,tdelayobj,lpotobj = np.array(massobj),np.array(tdelayobj),np.array(lpotobj)
# stdmass,stdtdelay,stdlpot = massobj.std(axis=0),tdelayobj.std(axis=0),lpotobj.std(axis=0) # Standard Deviation at all points in the field

# massobj = np.array(massobj)
# stdmass = massobj.std(axis=0)
# lpotobj = np.array(lpotobj)
# stdlpot = lpotobj.std(axis=0)
# detaobj = np.array(detaobj)
# stddeta = detaobj.std(axis=0)

np.savetxt('%s/AVGmassdens100.txt'%(path),avgmassdens) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDmassdens.txt'%(path),stdmass) # Use np.genfromtxt to get array back!
# np.savetxt('%s/AVGtimedelaytest.txt'%(path),avgtdelay) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDtimedelaytest.txt'%(path),stdtdelay) # Use np.genfromtxt to get array back!
# np.savetxt('%s/AVGdetA_%s.txt'%(path,zspock),avgdetA) # Use np.genfromtxt to get array back!
# for z in range(len(sourceplanes)):
# 	np.savetxt('%s/AVGlenspotMODEL100_%s.txt'%(path,sourceplanes[z]),avglpot[z]) # Use np.genfromtxt to get array back!
# for z in range(len(sourceplanes)):
# 	np.savetxt('%s/AVGdetA_%s.txt'%(path,sourceplanes[z]),avgdetA[z]) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDlenspot_B.txt'%(path),stdlpot) # Use np.genfromtxt to get array back!
# np.savetxt('%s/AVGdetA_B.txt'%(path),avgdetA) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDdetA_B.txt'%(path),stddeta) # Use np.genfromtxt to get array back!

# 	# Get All Plummers! (Optional I think)
# 	lensparams = lensX.getLensParameters()
# 	massplum,widthplum,xplum,yplum = [],[],[],[]
# 	for j in range(len(lensparams)-1):
# 		plummerparams = lensparams[j]['lens'].getLensParameters()
# 		massplum.append(plummerparams['mass']*lensparams[j]['factor'])
# 		widthplum.append(plummerparams['width'])
# 		xplum.append(lensparams[j]['x'])
# 		yplum.append(lensparams[j]['y'])
# 	plummermass.append(np.array(massplum)/MASS_SUN) # Mass in Solar Mass
# 	plummerwidth.append(np.array(widthplum)/ANGLE_ARCSEC) # Width in Arcsec
# 	plummerx.append(np.array(xplum)/ANGLE_ARCSEC) # x plummer in Arcsec
# 	plummery.append(np.array(yplum)/ANGLE_ARCSEC) # y plummer in Arcsec
# plummermass,plummerwidth,plummerx,plummery = np.array(plummermass),np.array(plummerwidth),np.array(plummerx),np.array(plummery)