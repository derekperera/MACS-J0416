from grale.constants import *
import grale.inversion as inversion
import grale.renderers as renderers
import grale.plotutil as plotutil
from grale.cosmology import Cosmology
import grale.lenses as lenses
import grale.images as images
import numpy as np
import sys # system commands
from scipy.stats import norm

# Write the RNG state, in case we want to reproduce the run exactly
# (note that the GRALE_DEBUG_SEED environment variable will need
# to be restored as well)
import random
print("RNG State:")
print(random.getstate())

renderers.setDefaultLensPlaneRenderer("threads") # threads, mpi, opencl, None or a Renderer object
renderers.setDefaultMassRenderer("threads") # threads, mpi, None, or a Renderer object
inversion.setDefaultInverter("threads") # threads, mpi, or an Inverter object
plotutil.setDefaultAngularUnit(ANGLE_ARCSEC)

V = lambda x, y: np.array([x,y], dtype=np.double)
z_lens = 0.396
cosm=Cosmology(0.7, 0.27, 0, 0.73)
iws = inversion.InversionWorkSpace(z_lens, 175*ANGLE_ARCSEC,regionCenter=[0.000*ANGLE_ARCSEC,0.000*ANGLE_ARCSEC], cosmology=cosm)
# NOTE: We are centered at the Zero Point, defined as the mean of the images

imgList = images.readInputImagesFile("MACSJ0416points_CANUCS24.txt", True) # Read in Data points

NULLSIZE=350
NULLSIZE2=NULLSIZE/2
NULLSUB=48
null = images.createGridTriangles(-V(NULLSIZE2, NULLSIZE2)*ANGLE_ARCSEC, V(NULLSIZE2, NULLSIZE2)*ANGLE_ARCSEC, NULLSUB, NULLSUB)

for i in imgList:
    # iws.addImageDataToList(i["imgdata"], i["z"], "pointgroupimages")
    iws.addImageDataToList(i["imgdata"], i["z"], "pointimages")
    iws.addImageDataToList(null, i["z"], "pointnullgrid")
# imgList[i]['imgdata'].getAllImagePoints() # reads in image locations in Arcsec!!!!

# #### Hybrid Model Parameters for the 4 galaxies
# Dd = cosm.getAngularDiameterDistance(z_lens) 
# Ddkpc = Dd/DIST_KPC # in kpc
# fac8 = 8*7*6*5*4*3*2*1
# n_sers = 4
# qell = 1.0
# b = 2*n_sers - (1/3)
# clusternames = ['BCG-N','spock-N','spock-S','BCG-S']
# massgalstel = np.array([2.02645608e+10, 1.43317051e+10, 4.02127470e+08, 1.86138355e+11]) # Stellar mass in Solar Masses
# reffgal = np.array([ 8.81,  1.88,  1.08, 10.83]) # Effective Radius in kpc
# galposx = np.array([  2.60591268,   7.04331105,   4.84097566, -17.54226198])
# galposy = np.array([ 12.53334684,   5.90934684,   0.90534684, -23.25065316])

# #### Sersic Parameters ####
# theta_s_kpc = (b**(-4))*reffgal # angular scale in kpc
# theta_s_rad = theta_s_kpc/Ddkpc # angular scale in radians
# sigcent_mkpc = (massgalstel*(b**8))/(fac8*np.pi*(reffgal**2)) # Solar mass per square kpc
# # sigcent_kgm2 = sigcent_mkpc*(MASS_SUN/(DIST_KPC**2)) # kg per square meter

# #### NFW parameters ####
# theta_s_rad = np.array([4.30989972e-05, 7.35764946e-06, 4.22673480e-06, 5.29809466e-05])
# rho_s_mkpc = np.array([1.92071128e+07, 2.60441266e+08, 2.96741294e+08, 9.58863947e+06]) # Solar mass per square kpc
# rho_s_kgm3 = rho_s_mkpc*(MASS_SUN/(DIST_KPC**3))

def setBasisFunctions(lens, minSub, maxSub):
    # iws.clearBasisFunctions() 
    iws.setUniformGrid(15) if not lens else iws.setSubdivisionGrid(lens, minSub, maxSub)
    # iws.addBasisFunctionsBasedOnCurrentGrid()

    # # Add Sersic basis lens (weight will also be optimized by GA)
    # for i in range(len(clusternames)):
    #     iws.addBasisFunctions( [{
    #         "lens": lenses.EllipticSersicLens(Dd, { "centraldensity": sigcent_kgm2[i] , "scale": theta_s_rad[i], "index":n_sers, "q":qell }),
    #         "center": [ galposx[i]*ANGLE_ARCSEC, galposy[i]*ANGLE_ARCSEC ]
    #         }])
    #     iws.addBasisFunctions( [{
    #         "lens": lenses.NFWLens(Dd, { "rho_s": rho_s_kgm3[i] , "theta_s": theta_s_rad[i]}),
    #         "center": [ galposx[i]*ANGLE_ARCSEC, galposy[i]*ANGLE_ARCSEC ]
    #         }])

beststepvals = []
for j in range(1,2): # Run Number
    bestStep, bestLens, bestFitness = None, None, None
    prevLens = None
    subDiv = 100
    subdivtot = []
    fitvals = []
    for i in range(1,16): # Subdivision Step
        if prevLens is None:
            setBasisFunctions(None, None, None)
        else:
            setBasisFunctions(prevLens, subDiv, subDiv+100)

        subdivtot.append((subDiv,subDiv+100))
        subDiv += 400

        lens, fitness, fitdesc = iws.invert(512) # Change to invertBasisFunctions() if using Sersic
        # lens.save(f"inv{j}_{i}.lensdata")
        prevLens = lens

        fitvals.append(fitness)

        if bestFitness is None or fitness < bestFitness:
            bestFitness = fitness
            bestLens = lens
            bestStep = i    

    bestLens.save(f"best_step{j}_{bestStep}.lensdata")

    # file = open(f'RunSummary{j}_20Gen.txt','w')
    # for i in range(len(subdivtot)):
    #     file.writelines('%f  %f \n'%(subdivtot[i][0],subdivtot[i][1]))
    # file.close()

#     np.savetxt(f'fitness{j}_20Gen.txt',fitvals)
#     beststepvals.append(bestStep)
# np.savetxt(f'beststeps{j}_20Gen.txt',beststepvals)