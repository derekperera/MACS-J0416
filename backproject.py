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
def RADEC_to_ARCSEC(RA,Dec,zpra,zpdec):
	# Takes you from RA,Dec in observed degrees to arcsec with respect to given zero point zpra,zpdec

	# First convert RA,Dec to real RA,Dec in deg
	RA = (RA-zpra)*np.cos(Dec*rad_per_deg)
	Dec = Dec - zpdec

	# Next convert to arcsec
	x_arc,y_arc = RA*3600,Dec*3600

	return x_arc,y_arc

print('dir00 = Normal')
print('dir11 = Normal + Yans + S1/S2 and F1/F2')
print('dir12 = Normal + Yans + S1/F1 and S2/F2')
print('dir21 = Normal + S1/S2 and F1/F2')
print('dir22 = Normal + S1/F1 and S2/F2')
choice = input('Which? ')
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

########################################################
###################### BACKPROJECTING ##################
xstot,ystot = [],[]
### KEY ###
# xs[i][j][k]: i = run number 1-40; j = which source; k = which source image

for i in range(1,41): # 1,41
	lensX = lenses.GravitationalLens.load("%s/best_step%s_%s.lensdata"%(path,i,int(bestgen[i-1])))

	xssrc,yssrc = [],[]
	for j in range(len(sourceID)):
		zs = imgList[j]['z']
		Ds, Dds = cosm.getAngularDiameterDistance(zs), cosm.getAngularDiameterDistance(zd, zs) # in meters
		Dspc,Ddspc = Ds*pc_per_m,Dds*pc_per_m # in pc
		sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2

		imgpts = imgList[j]['imgdata'].getAllImagePoints()

		betasrcx,betasrcy = [],[]
		for k in range(len(imgpts)):
			theta = imgpts[k][0]['position']
			thetax,thetay = theta/ANGLE_ARCSEC
			alphax,alphay = (Dds/Ds)*lensX.getAlphaVector(theta)/ANGLE_ARCSEC
			betax,betay = thetax-alphax,thetay-alphay
			betasrcx.append(betax)
			betasrcy.append(betay)

		xssrc.append(np.array(betasrcx,dtype='object'))
		yssrc.append(np.array(betasrcy,dtype='object'))

	xstot.append(np.array(xssrc,dtype='object'))
	ystot.append(np.array(yssrc,dtype='object'))

xstot,ystot = np.array(xstot,dtype='object'),np.array(ystot,dtype='object') # The Backprojected source positions

##################################
# CREATE PLOT
meanxs,meanys=[],[] # The mean source positions means[run number][sourceID]
for i in range(0,40):
	xs,ys = [],[]
	for j in range(len(sourceID)):
		xs.append(np.mean(xstot[i][j]))
		ys.append(np.mean(ystot[i][j]))
	meanxs.append(xs)
	meanys.append(ys)

meanxs,meanys=np.array(meanxs),np.array(meanys)
xscloud,yscloud = meanxs.T,meanys.T # The cloud[i] values are all the reconstructed source positions for their respective ID
np.save('%s/xcloud'%path,xscloud) # use np.load('run2_xforproj.npy',allow_pickle=True) to reload
np.save('%s/ycloud'%path,yscloud) # use np.load('run2_yforproj.npy',allow_pickle=True) to reload
xsmean = np.array([np.mean(xscloud[i]) for i in range(len(xscloud))]) # Mean xs over all reconstructions
ysmean = np.array([np.mean(yscloud[i]) for i in range(len(yscloud))]) # Mean xs over all reconstructions
for i in range(len(sourceID)):
	plt.scatter(xscloud[i],yscloud[i],color='b')
	plt.scatter(xsmean[i],ysmean[i],color='r')

file = open(f'{path}/meansourcepos.txt','w')
for i in range(len(xsmean)):
	file.writelines('%f  %f  %f  %f  %f \n'%(sourceID[i],xsmean[i],ysmean[i],cosm.getAngularDiameterDistance(imgList[i]['z'])*pc_per_m, cosm.getAngularDiameterDistance(zd, imgList[i]['z'])*pc_per_m))
file.close()
sys.exit()