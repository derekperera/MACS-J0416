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
import matplotlib.colors as colours
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy.visualization import astropy_mpl_style
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
Ddpc,Dspc,Ddspc = 1110940809.0062053, 1655189095.108516, 855769832.5452876 # in pc
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

def NFW(params,r):
	# Input array r of pc distances
	# Return Solar Mass per pc^2 if r is given in pc
	r_s,c = params 

	deltac = (200/3)*((c**3)/(np.log(1+c) - (c/(1+c))))
	xdim = r/r_s

	sigmanfw = []
	for i in range(len(xdim)):
		if xdim[i] == 1:
			sigmanfw.append((2*r_s*deltac*critdens)/3)
		if xdim[i] < 1:
			sigmanfw.append(((2*r_s*deltac*critdens)/(xdim[i]**2 - 1))*(1 - (2/np.sqrt(1 - xdim[i]**2))*np.arctanh(np.sqrt((1-xdim[i])/(1+xdim[i])))))
		if xdim[i] > 1:
			sigmanfw.append(((2*r_s*deltac*critdens)/(xdim[i]**2 - 1))*(1 - (2/np.sqrt(xdim[i]**2 - 1))*np.arctan(np.sqrt((xdim[i]-1)/(1+xdim[i])))))

	return np.array(sigmanfw)

def meanNFW(params,r):
	# Input array r of pc distances
	# Return Solar Mass per pc^2 if r is given in pc
	r_s,c = params 

	deltac = (200/3)*((c**3)/(np.log(1+c) - (c/(1+c))))
	xdim = r/r_s

	sigmanfw = []
	for i in range(len(xdim)):
		if xdim[i] == 1:
			sigmanfw.append((4*r_s*deltac*critdens)*(1+np.log(0.5)))
		if xdim[i] < 1:
			sigmanfw.append(((4*r_s*deltac*critdens)/(xdim[i]**2))*((2/np.sqrt(1 - xdim[i]**2))*np.arctanh(np.sqrt((1-xdim[i])/(1+xdim[i]))) + np.log(xdim[i]/2)))
		if xdim[i] > 1:
			sigmanfw.append(((4*r_s*deltac*critdens)/(xdim[i]**2))*((2/np.sqrt(xdim[i]**2 - 1))*np.arctan(np.sqrt((xdim[i]-1)/(1+xdim[i]))) + np.log(xdim[i]/2)))

	return np.array(sigmanfw)

def chi2(params,r,sig,err_sig):
	r_s,c = params 
	return sum(((sig - NFW([r_s,c],r))/(np.array(err_sig)))**2)

fig4,ax4=subplots(1,sharex=False,sharey=False,facecolor='w', edgecolor='k')
fig4.subplots_adjust(hspace=0)

##### SURFACE MASS DENSITY MAP #####
avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))
stdmassdens = np.genfromtxt('%s/STDmassdens.txt'%(path))
# Bounds of Window
gridsize=len(avgmassdens)
xlow,ylow,xhigh,yhigh = -70,-70,71,71
stepx=(xhigh-xlow)/(avgmassdens.shape[0])
stepy=(yhigh-ylow)/(avgmassdens.shape[1])
nx = (xhigh-xlow)/gridsize # arcsec per pixel
ny = (yhigh-ylow)/gridsize
x,y = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))
avgkappa = avgmassdens/sigcrit_rescaled # Our convergence (Ds/Dds = 1)

bergsize=2000
fitspath = '/Users/derek/Desktop/UMN/Research/MACSJ0416/imagefits'
Nim = [237,237,95,101,116,96,97,102,236,303,343,198,202,94]
compamodels = ['Perera24','Bergamini23','Keeton20','Williams18','CATS16','Diego18','Sharon17','Caminha16','MARS24',
			   'Rihtarsic24','Diego23','Richard21','Glafic17','Zitrin15']
modeltype=['G','P','P','G','P','H','P','P','FF','P','H','P','P','P']
kappamodels = [avgkappa,
			   '%s/Mass_zc0.396_Npix%s.fits'%(fitspath,bergsize),
			   '%s/hlsp_frontier_model_macs0416_keeton_v4_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_williams_v4_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_cats_v4.1_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_diego_v4.1_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_sharon_v4_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_caminha_v4_kappa.fits'%fitspath,
			   '%s/result_kappa_MACSJ0416_w_header_MARS.fits'%fitspath,
			   '%s/macs0416clu-kappa-best-100mas.fits'%fitspath,
			   avgkappa,
			   '%s/MACS0416_MUSE_DR_v1.1_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_glafic_v4_kappa.fits'%fitspath,
			   '%s/hlsp_frontier_model_macs0416_zitrin-nfw_v3_kappa.fits'%fitspath]
xm,ym = np.meshgrid(np.linspace(xlow,xhigh,avgkappa.shape[0]),np.linspace(ylow,yhigh,avgkappa.shape[1]))
xmrang = xm[0]
ymrang = ym.T[0]
def getKappa(kappafile):
	ind = compamodels.index(kappafile)
	if compamodels[ind]=='Perera24':
		kappa=avgkappa
	elif compamodels[ind]=='Diego23':
		Dspc,Ddspc = 956013768.6823735,800926431.7451072 # at z=9
		with fits.open("%s/AlphaY_DdsDs1_arcsec_MACS0416_Case6.fits"%fitspath) as hdu:
			alphax = hdu[0].data*(Ddspc/Dspc)
			xwcs2 = wcs.WCS(hdu[0].header)
		with fits.open("%s/AlphaX_DdsDs1_arcsec_MACS0416_Case6.fits"%fitspath) as hdu:
			alphay = hdu[0].data*(Ddspc/Dspc)
			ywcs2 = wcs.WCS(hdu[0].header)
		pixcoords = np.array([[i,j] for i in tqdm(range(alphax.shape[0])) for j in range(alphay.shape[1])])
		ra,dec = xwcs2.all_pix2world(pixcoords,1).T
		xb,yb = RADEC_to_ARCSEC(ra,dec,zeropointRA,zeropointDec) 
		xb,yb = xb.reshape(alphax.shape[0],alphax.shape[1]),yb.reshape(alphay.shape[0],alphax.shape[1])
		xblow,yblow,xbhigh,ybhigh = xb.min(),yb.min(),xb.max(),yb.max()
		nbx = (xbhigh-xblow)/alphax.shape[0]
		nby = (ybhigh-yblow)/alphay.shape[1]
		xbrang = np.linspace(xblow,xbhigh,alphax.shape[0])#np.arange(xblow,xbhigh,nbx)
		ybrang = np.linspace(yblow,ybhigh,alphay.shape[1])#np.arange(yblow,ybhigh,nby)
		# alphax,alphay = alphax*(nbx/206264.80624709636),alphay*(nby/206264.80624709636)
		psixx = np.gradient(alphax)[0]
		psiyy = np.gradient(alphay)[1]
		psixy = np.gradient(alphax)[1]
		psiyx = np.gradient(alphay)[0]
		kap = 0.5*(psixx + psiyy)
		kap = np.rot90(kap.T,3)/(nbx)
		gam1 = 0.5*(psixx - psiyy)
		gam1 = np.rot90(gam1.T,3)/(nbx)
		gam2 = psixy
		gam2 = np.rot90(gam2.T,3)/(nbx)
		magntest = 1/((1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2))

		massinterp = RectBivariateSpline(ybrang,xbrang,kap*(Dspc/Ddspc)) # Kappa
		kappa = np.array([massinterp.ev(xmrang[i],ymrang[j]) for i in tqdm(range(len(xmrang))) for j in range(len(ymrang))]).reshape(gridsize,gridsize)

		# muinterp = RectBivariateSpline(ybrang,xbrang,magntest) # Magnification
		# xmu,ymu = np.meshgrid(np.linspace(xlow,xhigh,magn.shape[0]),np.linspace(ylow,yhigh,magn.shape[1]))
		# xmurang = xmu[0]
		# ymurang = ymu.T[0]
		# magn2 = np.array([muinterp.ev(xmurang[i],ymurang[j]) for i in tqdm(range(len(xmurang))) for j in range(len(ymurang))]).reshape(gridsize+1,gridsize+1)

	elif compamodels[ind]=='Pererasub15':
		kappa=kappamodels[ind]
	else:	
		with fits.open(kappamodels[ind]) as hdu:
			data = hdu[0].data
			wcs2 = wcs.WCS(hdu[0].header)
			data = np.where(np.isnan(data)==False,data,0)
		pixcoords = np.array([[i,j] for i in tqdm(range(data.shape[1])) for j in range(data.shape[0])])
		ra,dec = wcs2.all_pix2world(pixcoords,1).T
		xb,yb = RADEC_to_ARCSEC(ra,dec,zeropointRA,zeropointDec) 
		xb,yb = xb.reshape(data.shape[0],data.shape[1]),yb.reshape(data.shape[0],data.shape[1])
		xblow,yblow,xbhigh,ybhigh = xb.min(),yb.min(),xb.max(),yb.max()
		nbx = (xbhigh-xblow)/data.shape[0]
		nby = (ybhigh-yblow)/data.shape[1]
		xbrang = np.linspace(xblow,xbhigh,data.shape[1])#np.arange(xblow,xbhigh,nbx)
		ybrang = np.linspace(yblow,ybhigh,data.shape[0])#np.arange(yblow,ybhigh,nby)
		massinterp = RectBivariateSpline(ybrang,xbrang,np.rot90(data.T,3)) # Kappa
		kappa = np.array([massinterp.ev(xmrang[i],ymrang[j]) for i in tqdm(range(len(xmrang))) for j in range(len(ymrang))]).reshape(gridsize,gridsize)
		if compamodels[ind]=='Bergamini23':
			kappa = ((kappa*1e12)*(arcsec_kpc**2))/sigcrit_rescaled

	return kappa

# bergmassdens=getKappa('Bergamini23')*sigcrit_rescaled
# diegmassdens=getKappa('Diego23')*sigcrit_rescaled

##### CIRCULAR AVERAGE MASS DENSITY #####
### Circular average about the BCG? ####
nbins = 75
clusternames = ['BCG-N','BCG-S']
modelgalind = np.array([0,17,22,72])
xbcgn,ybcgn,xbcgs,ybcgs = xgal[0],ygal[0],xgal[72],ygal[72]
xbcgtot,ybcgtot = [xbcgn,xbcgs],[ybcgn,ybcgs]
circproftot=[]
def circprof(x,y,xcen,ycen,massdens,nbins):
	# x : x meshgrid
	# y : y meshgrid
	# xcen,ycen : coordinates of center
	# massdens : 2D density distribution
	# nbins : Number of bins
	binvals = np.linspace(1e-14,30,nbins) # Values of bins
	distbins = [[] for i in range(nbins)] # Bins of circular profile
	for i in tqdm(range(len(x))):
		for j in range(len(y)):
			xpix = x[i][j]
			ypix = y[i][j]
			massdenspix = massdens[i][j]
			dist = np.sqrt(((xpix-xcen)**2 + (ypix-ycen)**2)) # Distance from pixel to BCG
			if (dist>max(binvals)):
				distbins[nbins-1].append(massdenspix)
				continue
			for k in range(len(binvals)-1):
				if (dist>binvals[k] and dist<=binvals[k+1]):
					distbins[k].append(massdenspix)

	circprof = np.array([np.mean(np.array(bins)) for bins in distbins])
	std_circprof = np.array([np.std(np.array(bins)) for bins in distbins])
	circprof = circprof[1:-1]/(arcsec_kpc**2) # Solar mass per kpc^2
	std_circprof = std_circprof[1:-1]/(arcsec_kpc**2)
	rkpc = binvals[1:-1]*arcsec_kpc
	return circprof,std_circprof,rkpc

# returns log-log plot of density
def densityPlot(x, y, x0, y0, kappa, bins):
	#> scale declaration
	pix_kpc = 1/(arcsec_kpc*nx) # pixels per kpc

	#> translating to polar
	xx = x.flatten(); yy = y.flatten()
	xx = (xx-int(x0)) * arcsec_kpc; yy = (yy-int(y0)) * arcsec_kpc
	r = np.sqrt(xx**2 + yy**2)
	kap = kappa.flatten()/(arcsec_kpc**2)

	#> radius bins declarations
	rmin = np.min(r)
	if rmin == 0: rmin = 1 / pix_kpc
	rmax = np.max(r) + 1e-3
	# print(rmin, rmax)

	# creating bins & bin width
	rbins = np.arange(rmin, rmax, (rmax-rmin)/bins)
	width = (rmax-rmin)/bins

	#> integrating
	kapTot = np.zeros(shape=(2, bins))
	for i, radius in tqdm(enumerate(r)):
		if radius == np.inf: continue 
		index = int( (radius - rmin) / width )
		binKap, binTot = kapTot[0][index], kapTot[1][index]
		kapTot[0][index] = ((binKap * binTot) + kap[i]) / (binTot + 1) 
		kapTot[1][index] += 1  # bin total
	return rbins, kapTot[0,:],kapTot

for h in range(2):
	xbcg,ybcg = xbcgtot[h],ybcgtot[h]
	# ########## PLOT ##################
	# logsigma = np.log10(circprof)
	# logr = np.log10(binvals)

	colors = ['b','r','k','y']

	cprof,std_circprof,rkpc = circprof(x,y,xbcg,ybcg,avgmassdens,nbins)
	rjohn,massjohn,kaptot = densityPlot(x,y,xbcg,ybcg,avgmassdens,nbins)
	# diegcprof,diegstd_circprof,rkpc = circprof(x,y,xbcg,ybcg,diegmassdens,nbins)
	# bergcprof,bergstd_circprof,rkpc = circprof(x,y,xbcg,ybcg,bergmassdens,nbins)

	ax4.errorbar(rkpc,cprof,color=colors[h],linestyle='-',label=clusternames[h])
	ax4.errorbar(rjohn,massjohn,color='g')
	# ax4.errorbar(rkpc,diegcprof,color=colors[h],linestyle=':')
	# ax4.errorbar(rkpc,bergcprof,color=colors[h],linestyle='--')
	# ax4.errorbar(logr,logsigma,color=colors[permutation],linestyle='-',label='%s sersic / %s im.'%(srs,allrev))
	# y += np.random.normal(0, 0.1, size=y.shape)
	ax4.fill_between(rkpc, cprof-std_circprof, cprof+std_circprof,color=colors[h],alpha=0.5)
	imdist= np.sqrt((xarc-xbcg)**2 + (yarc-ybcg)**2)*arcsec_kpc
	for imag in imdist:
		if imag <= 30*arcsec_kpc:
			ax4.axvline(x=imag,ymin=0,ymax=0.05,color=colors[h])

	# scalerad,cvals = [],[]
	# for i in tqdm(range(500)):
	# 	cprof = np.array([np.random.normal(circprof[j],std_circprof[j]) for j in range(len(circprof))])
	# 	res = minimize(chi2,[100,10],args=(binvals[1:-1]*arcsec_pc,cprof,std_circprof),method='Nelder-Mead',options = {'maxiter':10000})
	# 	scalerad.append(res.x[0])
	# 	cvals.append(res.x[1])
	# scalerad,cvals = np.array(scalerad),np.array(cvals)
	# signfw = NFW([np.mean(scalerad),np.mean(cvals)],binvals[1:-1]*arcsec_pc) # Solar mass per pc^2
	# signfw = NFW([640000,100],binvals[1:-1]*arcsec_pc) # Solar mass per pc^2
	# signfw = signfw*arcsec_pc*arcsec_pc # Solar mass per arcsec^2
	# ax4.errorbar(binvals[1:-1],signfw,color=colors[h],linestyle='--',label='Best Fit NFW %s'%clusternames[h])
	# ax4.errorbar(binvals,NFW([39*arcsec_pc,6.1],binvals*arcsec_pc)*arcsec_pc*arcsec_pc,color='r',linestyle='--')

	# print('R_s: ',np.mean(scalerad)/arcsec_pc,'+/-',np.std(scalerad)/arcsec_pc) # arcsec
	# print('c: ',np.mean(cvals),'+/-',np.std(cvals))
	# print(np.mean(scalerad),np.std(scalerad))
	# R200 = ufloat(np.mean(scalerad),np.std(scalerad))*ufloat(np.mean(cvals),np.std(cvals))
	# print(R200)
	# M200 = (4/3)*np.pi*200*critdens*(R200**3)

ax4.set_xlabel(r'r [kpc]',fontsize=15,fontweight='bold')
ax4.set_ylabel(r'$\Sigma$ [M$_{\odot}$ kpc$^{-2}$]',fontsize=15,fontweight='bold')

ax4.set_xscale('log')
ax4.set_yscale('log')

ax4.legend(loc=3,title='',ncol=1,prop={'size':15})
ax4.minorticks_on()

# ax4.set_ylim(bottom=0.0, top=max(np.concatenate(circproftot))-100)
# ax4.set_xlim(left=-3.0, right=None)

def region(regionx,regiony,regionsize,avgmassdens):

	# regionx is x coordinate of center of circular region in arcsec
	# regiony is y coordinate of center of circular region in arcsec
	# regionsize is radius of circular region in arcesec

	# dist = np.sqrt((x-regionx)**2 + (y-regiony)**2)
	# densreginit = np.array([avgmassdens[i][j]*(stepx**2) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	# regpos = np.array([(i,j) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	# for i,item in enumerate(densreginit):
	# 	if item == max(densreginit):
	# 		xregpos,yregpos = regpos[i][0],regpos[i][1]
	# 		break
	# xclump = x[xregpos][yregpos] # Peak of Clump x
	# yclump = y[xregpos][yregpos] # Peak of Clump y

	xclump,yclump = regionx,regiony
	phi = np.linspace(0, 2*np.pi, 100)
	x1 = regionsize*np.cos(phi) + xclump
	x2 = regionsize*np.sin(phi) + yclump
	# ax1.errorbar(x1,x2,color='r',label='region')
	# # ax1.errorbar([regionx],[regiony],color='b',marker='*')
	# ax1.errorbar([xclump],[yclump],color='r',marker='*')

	dist = np.sqrt((x-xclump)**2 + (y-yclump)**2)
	densregion = np.array([(avgmassdens[i][j])*(stepx**2) for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	massregion = sum(densregion)
	stddensregion = np.array([((stdmassdens[i][j])*(stepx**2))**2 for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	stdmassregion = np.sqrt(sum(stddensregion))

	return xclump,yclump,massregion,densregion,stdmassregion
xclump,yclump,massclump,dens,stdmassclump = region(xbcgs,ybcgs,200/arcsec_kpc,avgmassdens)
print('BCG-S:', massclump, '+/-', stdmassclump)
xclump,yclump,massclump,dens,stdmassclump = region(xbcgn,ybcgn,200/arcsec_kpc,avgmassdens)
print('BCG-N:', massclump, '+/-', stdmassclump)

plt.show()

# CREATE PLOT
fig1,ax1=subplots(1,figsize=(24,24),sharex=False,sharey=False,facecolor='w', edgecolor='k')
# Bounds of Window
xlow,ylow,xhigh,yhigh = -70,-70,71,71
stepx=(xhigh-xlow)/(avgmassdens.shape[0])
stepy=(yhigh-ylow)/(avgmassdens.shape[1])
x,y = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))

avgmassdens1 = avgmassdens/sigcrit
path2 = '/Users/derek/Desktop/UMN/Research/MACSJ0416/dir21'
avgmassdens2 = np.genfromtxt('%s/AVGmassdens.txt'%(path2))/sigcrit
avgmassdenscompa = avgmassdens1-avgmassdens2
im = ax1.imshow(avgmassdenscompa,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='PuOr',origin='lower',norm=colours.CenteredNorm())
contourmass=ax1.contour(x,y,avgmassdenscompa,levels=[0],origin='lower',alpha=0.5,zorder=2)
divider = make_axes_locatable(ax1)
ax1.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax1.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax1=divider.append_axes("right", size="5%",pad=0.25)
cax1.yaxis.set_label_position("right")
cax1.yaxis.tick_right()
fig1.colorbar(im,label=r'$\Sigma / \Sigma_{crit}$',cax=cax1)
ax1.minorticks_on()
ax1.set_aspect('equal')
ax1.set_anchor('C')
ax1.grid(False)

ax1.tick_params(axis='x',labelsize='10')
ax1.tick_params(axis='y',labelsize='10')

ax1.scatter(xarc,yarc,color='b',alpha=1,marker='.',label='Observed',zorder=1.5,s=15)
ax1.errorbar(xgal,ygal,color='mediumspringgreen',marker='d',linestyle='None',label='Cluster Galaxy',alpha=0.5)
ax1.text(-12,-6,'M1',color='lime',alpha=0.9,size='x-small')
ax1.text(-20,-32,'M2',color='lime',alpha=0.9,size='x-small')
ax1.invert_xaxis()

ax1.legend(loc=1,title='',ncol=2,prop={'size':9})
plt.show()