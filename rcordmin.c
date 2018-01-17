#include "mex.h"
#include "matrix.h"
#include "math.h"
#include <time.h>
#include <stdlib.h>

/* compile with -largeArrayDims flag */

/* to use "mwSize", need matrix.h, but doesn't work for R2006a */
/* So, use the following definitions instead: */
#ifndef mwSize
    #define mwSize size_t
#endif
#ifndef mwIndex
    #define mwIndex size_t  /* should make it compatible w/ 64-bit systems */
#endif

#define x_out plhs[0]
#define Ax_out plhs[1]

#define x_in  prhs[0]
#define A_in  prhs[1]
#define b_in  prhs[2]
#define Ax_in prhs[3]
#define Rp prhs[4]


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    /* Inputs:
     * 
     * x_in: initial point
     * A_in: data matrix (must be in sparse representation)
     * b_in: the vector of labels
     * Ax_in: the vector of A*x_in
     * Rp: regularization parameter of the problem
     *
     * Outputs:
     * x_out: the output of a cycle of coordinate minimization
     * Ax_out: the vector of A*x_out
     */

    mwIndex *ir, *jc, i,ii,j,jj=0;
    mwSize ncol, mrow;

    double *x, *A, *b, *Ax, rp, *xo, *Axo, *Axt, F=0, nsmv=0, LHS, RHS, diff=0;
    int ls;
    
    mwSize k,stop;
    double gj, hj, yj, xp, td, xjt, Ft, d, ss, sig=0.1, beta=0.1;
    
    
    
    
    /* Check for proper number of arguments */
    if (nlhs!=2) {
        mexErrMsgTxt("rcordmin.c : requires 2 outputs");
    }
    if (nrhs!=5) {
        mexErrMsgTxt("rcordmin.c : requires 5 inputs");
    }
    
    /* Check for proper format and dimension of inputs */
    
    if (!mxIsSparse(A_in)) {
        mexErrMsgTxt("rcordmin.c : the input data matrix must be sparse");
    }
    mrow = mxGetM(A_in);
    ncol = mxGetN(A_in);
    
    /* Allocate output */
    x_out = mxCreateDoubleMatrix(ncol,1,mxREAL);
    Ax_out = mxCreateDoubleMatrix(mrow,1,mxREAL);
    
    /* I/O pointers */
	x = mxGetPr(x_in);
    A = mxGetPr(A_in);
    b = mxGetPr(b_in);
    Ax = mxGetPr(Ax_in);
    ir = mxGetIr(A_in);      /* Row indexing      */
    jc = mxGetJc(A_in);      /* Column count      */
	rp = mxGetScalar(Rp);
    
    xo = mxGetPr(x_out);
    Axo = mxGetPr(Ax_out);
    
    

    

    
    /* Initialization of outputs */
    for(i=0;i<ncol;i++){
        xo[i] = x[i];
        nsmv += rp*fabs(xo[i]);
    }
    
    Axt = malloc(mrow*sizeof(double));
    
    for(i=0;i<mrow;i++){
        Axo[i] = Ax[i];
        Axt[i] = Axo[i];
        F += (log(1+exp(Axo[i])) - b[i]*Axo[i])/mrow;
    }
    F += nsmv;
    
    /* A cycle of randomized coordimate minimization */
    for(i=0; i<ncol; i++){
        
        /* First calculate gj, hj */

        j = i; // j = rand() % ncol;
        gj = 0;
        hj = 0.000001;
        stop = jc[j+1];
        for(k=jc[j]; k<stop; k++) {
            gj += (exp(Axo[ir[k]])/(1 + exp(Axo[ir[k]])) - b[ir[k]])*A[k]/mrow;
            hj += exp(Axo[ir[k]])/((1 + exp(Axo[ir[k]]))*(1 + exp(Axo[ir[k]])))*A[k]*A[k]/mrow;
        }

        /* Second calculate yj */
        yj = xo[j] - gj/hj;


        td = rp/hj;

        if(yj>td){
            xp = yj - td;
        }
        else if(yj<-td){
            xp = yj + td;
        }
        else {
            xp = 0;
        }
        
        /* Use d = xp - xo[j] as the descent direction and perform line search */
        d = xp - xo[j];        
        
        ss = 1;
        xjt = xo[j] + ss*d;
        for(k=jc[j]; k<stop; k++) {
            Axt[ir[k]] = Axo[ir[k]] + (xjt - xo[j])*A[k];
        }
        
        
        Ft = nsmv + rp*(fabs(xjt) - fabs(xo[j]));
        for(ii=0;ii<mrow;ii++){
            Ft += (log(1+exp(Axt[ii])) - b[ii]*Axt[ii])/mrow;
        }
        
//         if(Ft - F > sig*ss*(gj*d + rp*fabs(xjt) - rp*fabs(xo[j]))){
//             ls = 1;
//         }
        
//         if(ls==1){
//             ls = 0;
//             ss = ss*beta;
//             xjt = xo[j] + ss*d;
//             for(k=jc[j]; k<stop; k++) {
//                 Axt[ir[k]] = Axo[ir[k]] + (xjt - xo[j])*A[k];
//             }     
//             Ft = nsmv + rp*(fabs(xjt) - fabs(xo[j]));
//             for(ii=0;ii<mrow;ii++){
//                 Ft += (log(1+exp(Axt[ii])) - b[ii]*Axt[ii])/mrow;
//             }
//         }

        



        
        /* Update xo and Axo */
        for(k=jc[j]; k<stop; k++) {
            Axo[ir[k]] += (xjt - xo[j])*A[k];
        }
                
        
        F = Ft;
        
        nsmv = nsmv + rp*(fabs(xjt) - fabs(xo[j]));
        
        xo[j] = xjt;
    }
    
    return;
}
