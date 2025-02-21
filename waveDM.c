#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <float.h>

float random(float min, float max);
float RMS(double arr[], int n, float avg);
double* linspace(double x1, double x2, int n);
int flip();
double GaussLens(double xprime, double yprime, double x, double y, double kappapeak, double width);

int main() {
  // Seed the random number generator with the current time
  srand(time(NULL));
  char filename[100],output[50],fullmap[100];
  sprintf(filename, "/Users/derek/Desktop/UMN/Research/MACSJ0416/dir00/kappamothra.txt"); // Zoomed in Kappa map (1000x1000)
  sprintf(fullmap, "/Users/derek/Desktop/UMN/Research/MACSJ0416/dir00/kappamothfullmap.txt"); // Full Kappa map (500x500)
  sprintf(output,"waveDMfield.txt");
  // sprintf(output,"wavetest.txt");

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// Get Kappa values for zooomed in map ///////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  FILE *file;

  float linecount = 0; // Line counter (result)
  char c; // To store a character read from file

  file = fopen(filename, "r");

  for (c = getc(file); c != EOF; c = getc(file))
    if (c == '\n') // Increment count if this character is newline
      linecount = linecount + 1;

  fclose(file);
  // printf("%f",linecount);

  int i,j,k,m,n,num,size=linecount;
  double *x = (double*)malloc(size * sizeof(double));
  double *y = (double*)malloc(size * sizeof(double));
  double *avgkap = (double*)malloc(size * sizeof(double));

  // Read in Data to get x,y and background kappa for Mothra
  file = fopen(filename, "r");
  for(i=0;i<size;i++){
    fscanf(file,"%lf %lf %lf\n",&x[i],&y[i],&avgkap[i]);
  }  
  fclose(file);

  int sizegrid = 1000; // Zoomed in map size
  double ** kappa;
  kappa = (double**)malloc(sizeof(double*)*sizegrid);
  for (i = 0; i <= sizegrid; i++){
    kappa[i] = (double*)malloc(sizeof(double)*sizegrid);
  }
  double ** xk;
  xk = (double**)malloc(sizeof(double*)*sizegrid);
  for (i = 0; i <= sizegrid; i++){
    xk[i] = (double*)malloc(sizeof(double)*sizegrid);
  }
  double ** yk;
  yk = (double**)malloc(sizeof(double*)*sizegrid);
  for (i = 0; i <= sizegrid; i++){
    yk[i] = (double*)malloc(sizeof(double)*sizegrid);
  }
  for (int i = 0; i < sizegrid; i++) {
    for (int j = 0; j < sizegrid; j++) {
      kappa[i][j] = avgkap[i * sizegrid + j];
      xk[i][j] = x[i * sizegrid + j];
      yk[i][j] = y[i * sizegrid + j];
      // printf("%lf %lf %lf \n",xk[i][j],yk[i][j],kappa[i][j]);
    }
  }
  // double *xrang = (double*)malloc(sizegrid * sizeof(double));
  // double *yrang = (double*)malloc(sizegrid * sizeof(double));
  // float nx=-70;
  // for (int iterx=0;iterx<501;iterx++){
  //   xrang[iterx] = nx;
  //   yrang[iterx] = nx;
  //   // printf("%d %lf %lf \n",iterx,xrang[iterx],yrang[iterx]);
  //   nx+=0.282;
  // }
  // CHECK
  // for (i=0;i<sizegrid;i++) {
  //   printf("%lf %lf %lf %lf\n",x[i],y[i],avgkap[i],kappa[0][i]);
  // }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// Get Kappa values for full map ///////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  FILE *fullfile;

  fullfile = fopen(fullmap, "r");

  float linecount2=0;
  for (c = getc(fullfile); c != EOF; c = getc(fullfile))
    if (c == '\n') // Increment count if this character is newline
      linecount2 = linecount2 + 1;

  fclose(fullfile);
  // printf("%f",linecount);

  int sizefull=linecount2;
  double *xf = (double*)malloc(sizefull * sizeof(double));
  double *yf = (double*)malloc(sizefull * sizeof(double));
  double *fullkap = (double*)malloc(sizefull * sizeof(double));

  // Read in Data to get x,y and background kappa for Mothra
  fullfile = fopen(fullmap, "r");
  for(i=0;i<sizefull;i++){
    fscanf(fullfile,"%lf %lf %lf\n",&xf[i],&yf[i],&fullkap[i]);
  }  
  fclose(fullfile);

  int fullmapsize=500; // Full map size
  double ** kappafull;
  kappafull = (double**)malloc(sizeof(double*)*fullmapsize);
  for (i = 0; i <= fullmapsize; i++){
    kappafull[i] = (double*)malloc(sizeof(double)*fullmapsize);
  }
  double ** xfull;
  xfull = (double**)malloc(sizeof(double*)*fullmapsize);
  for (i = 0; i <= fullmapsize; i++){
    xfull[i] = (double*)malloc(sizeof(double)*fullmapsize);
  }
  double ** yfull;
  yfull = (double**)malloc(sizeof(double*)*fullmapsize);
  for (i = 0; i <= fullmapsize; i++){
    yfull[i] = (double*)malloc(sizeof(double)*fullmapsize);
  }
  for (int i = 0; i < fullmapsize; i++) {
    for (int j = 0; j < fullmapsize; j++) {
      kappafull[i][j] = fullkap[i * fullmapsize + j];
      xfull[i][j] = xf[i * fullmapsize + j];
      yfull[i][j] = yf[i * fullmapsize + j];
      // printf("%lf %lf %lf \n",xfull[i][j],yfull[i][j],kappafull[i][j]);
    }
  }


  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////

  // Initialize Mothra
  int a,b;
  double arcsec_pc = 5385.993031091043; // pc per arcsec
  double arcsec_rad = 206264.80624709636; // arcsec per radian
  double xmoth=-1.798733738507787,ymoth=16.997346835444205; // Position of Mothra in arcsec
  double xlow=xmoth-0.7,ylow=ymoth-0.7,xhigh=xmoth+0.7,yhigh=ymoth+0.7; // Modelling window
  // Zoomed in Map dx,dy
  double dxi=(((xhigh-xlow)/1000.0)/arcsec_rad)*1.0e9,dyi=(((yhigh-ylow)/1000.0)/arcsec_rad)*1.0e9; // Resolution in arcsec per pixel (scaled for memory purposes)
  // Full Map dx,dy
  double dxm=((141/500.0)/arcsec_rad)*1.0e9,dym=((141/500.0)/arcsec_rad)*1.0e9; // Resolution in arcsec per pixel (scaled for memory purposes)
  double xgran,ygran; // Granule Position
  int perturb; // Either a Positive or Negative Wave DM perturbation

  // Individual Sublens (change fullmapsize to sizegrid if calculating on zoomed map)
  double ** sublens;
  sublens = (double**)malloc(sizeof(double*)*fullmapsize);
  for (i = 0; i <= fullmapsize; i++){
    sublens[i] = (double*)malloc(sizeof(double)*fullmapsize);
  }
  // Zoomed Individual Sublens (change fullmapsize to sizegrid if calculating on zoomed map)
  double ** sublensz;
  sublensz = (double**)malloc(sizeof(double*)*sizegrid);
  for (i = 0; i <= sizegrid; i++){
    sublensz[i] = (double*)malloc(sizeof(double)*sizegrid);
  }
  // Total Granule Map (Kappa) (change fullmapsize to sizegrid if calculating on zoomed map)
  double ** granule;
  granule = (double**)malloc(sizeof(double*)*fullmapsize);
  for (i = 0; i <= fullmapsize; i++){
    granule[i] = (double*)malloc(sizeof(double)*fullmapsize);
  }
  // Zoomed Granule Map (Kappa) (change fullmapsize to sizegrid if calculating on zoomed map)
  double ** granulez;
  granulez = (double**)malloc(sizeof(double*)*sizegrid);
  for (i = 0; i <= sizegrid; i++){
    granulez[i] = (double*)malloc(sizeof(double)*sizegrid);
  }
  double kappadens = 0.5, lamdb = 10.0; // Granule parameters: K~0.9, de Broglie wavelength ~ 10 pc
  double sublenswidth = (lamdb/arcsec_pc)/2.35482; // de Broglie wavelenght is the FWHM of Gaussian subhalo, so this is width in arcsec

  double test;
  // (change fullmapsize to sizegrid if calculating on zoomed map)
  for(num=0;num<0;num++){
    xgran = random(xlow,xhigh);
    ygran = random(ylow,yhigh);
    perturb = flip();
    test=0;
    for (int i = 0; i < fullmapsize; i++) {
      for (int j = 0; j < fullmapsize; j++) {
        sublens[i][j] = GaussLens(xgran,ygran,xfull[i][j],yfull[i][j],kappadens,sublenswidth);
        test+=granule[i][j];
      }
    }
    printf("%lf\n",test);
    test=0;
    for (int i = 0; i < sizegrid; i++) {
      for (int j = 0; j < sizegrid; j++) {
        sublensz[i][j] = GaussLens(xgran,ygran,xk[i][j],yk[i][j],kappadens,sublenswidth);
        test+=granulez[i][j];
      }
    }
    printf("%lf\n",test);
    if (perturb == 1)
      for (m=0;m<fullmapsize; m++) {
        for (n=0;n<fullmapsize;n++) {
          granule[m][n]+=sublens[m][n];
        }
      }
      for (a=0;a<sizegrid; a++) {
        for (b=0;b<sizegrid;b++) {
          granulez[a][b]+=sublensz[a][b];
        }
      }
      printf("%d %lf %lf %d\n",num,xgran,ygran,perturb);
    if (perturb == 0)
      for (m=0;m<fullmapsize; m++) {
        for (n=0;n<fullmapsize;n++) {
          granule[m][n]+=sublens[m][n];
        }
      }
      for (a=0;a<sizegrid; a++) {
        for (b=0;b<sizegrid;b++) {
          granulez[a][b]+=sublensz[a][b];
        }
      }
      printf("%d %lf %lf %d\n",num,xgran,ygran,perturb);
  }

  double gransum,distance;
  double ** totlpot;
  totlpot = (double**)malloc(sizeof(double*)*sizegrid);
  for (i = 0; i <= sizegrid; i++){
    totlpot[i] = (double*)malloc(sizeof(double)*sizegrid);
  }

  //////////////////////////////////////////////////////////
  ////////////////////// MAIN LOOP /////////////////////////
  //////////////////////////////////////////////////////////

  // Here i,j is the lens potential map; m,n is the full map
  #pragma omp parallel for num_threads(4) private(j,m,n,a,b,gransum,distance)
  for (int i = 0; i < sizegrid; i++) {
    for (int j = 0; j < sizegrid; j++) {
      double gransum = 0.0;
      for (int a = 0; a < sizegrid; a++) {
        for (int b = 0; b < sizegrid; b++) {
          double distance = sqrt(pow(xk[i][j] - xk[a][b], 2.0) + pow(yk[i][j] - yk[a][b], 2.0)) + DBL_EPSILON;
          gransum += (granulez[a][b]+kappa[a][b]) * log(distance) * dxi * dyi;
        }
      }
      for (int m = 0; m < fullmapsize; m++) {
        for (int n = 0; n < fullmapsize; n++) {
          if (xfull[m][n]<xlow || xfull[m][n]>xhigh || yfull[m][n]<ylow || yfull[m][n]>yhigh){
            double distance = sqrt(pow(xk[i][j] - xfull[m][n], 2.0) + pow(yk[i][j] - yfull[m][n], 2.0)) + DBL_EPSILON;
            gransum += (granule[m][n]+kappafull[m][n]) * log(distance) * dxm * dym;
          }
        }
      }
      totlpot[i][j] = gransum/M_PI;
      printf("%d %d %lf %lf %lf\n",i,j,xk[i][j],yk[i][j],totlpot[i][j]);
    }
  }

  file = fopen(output,"w");
  for(i=0;i<sizegrid;i++){
    for(j=0;j<sizegrid;j++){
      // fprintf(file,"%lf %lf %lf %lf\n",xk[i][j],yk[i][j],granule[i][j]+kappa[i][j],totlpot[i][j]);
      fprintf(file,"%lf %lf %lf\n",xk[i][j],yk[i][j],totlpot[i][j]);
    }
  }  
  fclose(file);

  return 0;
}

int flip( ){
    int i = rand() % 2;
        if (i == 0)
             return 0;
        else
             return 1;  
       }

float random(float min, float max) {
    // Generate a random number between 0 and 1
    float random = (float)rand() / RAND_MAX;

    // Scale and shift the random number to the desired range
    return random * (max - min) + min;
}

float RMS(double arr[], int n, float avg){
  int i;
  float mean,root,square;

  square=0;
  for (i=0;i<n;i++){
    square+=(arr[i] - avg)*(arr[i] - avg);
  }

  mean = square/n;
  root = sqrt(mean);
  return root;
}

double GaussLens(double xprime, double yprime, double x, double y, double kappapeak, double width){
  double kappa;
  kappa = kappapeak*exp(-( (x-xprime)*(x-xprime) + (y-yprime)*(y-yprime) )/(2*width*width));
  return kappa;
}

double* linspace(double x1, double x2, int n) {

 double *x = calloc(n, sizeof(double));

 double step = (x2 - x1) / (double)(n - 1);

 for (int i = 0; i < n; i++) {
     x[i] = x1 + ((double)i * step);
 }
 
return x;
}