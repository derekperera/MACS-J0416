READ ME!!!!!!!

MACSJ0416points.txt is just the spectroscopic confirmed points (Bergamini et. al. 23)
MACSJ0416points_trans1.txt is the points including Yan23 transients and S1/S2 and F1/F2 counterimaged (dir11) FF12+D
MACSJ0416points_trans2.txt is the points including Yan23 transients and S1/F1 and S2/F2 counterimaged (dir12) FF11+D
MACSJ0416points_trans21.txt is the points including S1/S2 and F1/F2 counterimaged (dir21) FF12
MACSJ0416points_trans22.txt is the points including S1/F1 and S2/F2 counterimaged (dir22) FF11
MACSJ0416points_subset15.txt is the random subset of 15% fewer sources

Hybrid Models
Sersic
MACSJ0416points.txt is just the spectroscopic confirmed points (Bergamini et. al. 23) (dir30) H-Ser
MACSJ0416points_trans31.txt is the points including S1/S2 and F1/F2 counterimaged (dir31)
MACSJ0416points_trans32.txt is the points including S1/F1 and S2/F2 counterimaged (dir32)

NFW 
MACSJ0416points.txt is just the spectroscopic confirmed points (Bergamini et. al. 23) (dir40) H-NFW
MACSJ0416points_trans41.txt is the points including S1/S2 and F1/F2 counterimaged (dir41)

Pipeline:
1. Run inversion on MSI
2. Change directory for different runs, then run dataprocess.py
	This will gather the lensing potentials and average surface mass density map
3. Run backproject.py to get mean source positions for all sources
	Optionally: Run imagerecon.py to get mean reconstructed image positions, saved as .npy
4. Run src_optim.py to get optimized source positions, saved as optimsourcepos.txt
5. Now run imagerecon.py (again) to get optimized source positions for all sources. Note: Need to change file names
6. Use plotsurfdens.py and plottdsurf.py to see the results!