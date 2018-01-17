#include "mex.h"
#include "matrix.h"
#include "math.h"

/* compile with -largeArrayDims flag */

/* to use "mwSize", need matrix.h, but doesn't work for R2006a */
/* So, use the following definitions instead: */
#ifndef mwSize
    #define mwSize size_t
#endif
#ifndef mwIndex
    #define mwIndex size_t  /* should make it compatible w/ 64-bit systems */
#endif

#define s_out plhs[0]
#define Ax_out plhs[1]

#define s_in  prhs[0]
#define A_in  prhs[1]
#define Dx  prhs[2]
#define Ax_in prhs[3]
#define q_in  prhs[4]
#define Rp prhs[5]
#define Mu prhs[6]


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    /* Inputs:
     * 
     * s: initial point
     * A: data matrix (must be in sparse representation)
     * Dx: the VECTOR of diagonal component of Hessian, H = A'*Diag(Dx)*A;
     * Ax_in: the vector of A*s
     * q_in: the vector of gradient and other information
     * Rp: regularization parameter of the problem
     * Mu: Hessian regularization parameter
     
     * Outputs:
     * s_out: the output of a cyclic coordinate minimization
     * Ax_out: the vector of A*x
     */

    mwIndex *ir, *jc, i;
    mwSize ncol, mrow;

    double *s, *q, *D, *A, *Ax, *so, *Axo;
    double mu, rp;
    
    mwSize k,stop;
    double H_ii, adai, tmp, td;
    
    
    
    
    /* Check for proper number of arguments */
    if (nlhs!=2) {
        mexErrMsgTxt("ripn_cordmin.c : requires 2 outputs");
    }
    if (nrhs!=7) {
        mexErrMsgTxt("ripn_cordmin.c : requires 7 inputs");
    }
    
    /* Check for proper format and dimension of inputs */
    
    if (!mxIsSparse(A_in)) {
        mexErrMsgTxt("ripn_cordmin.c : the input data matrix must be sparse");
    }
    mrow = mxGetM(A_in);
    ncol = mxGetN(A_in);
//     if ((mxGetM(Dx)!=mrow)||(mxGetM(q)!=ncol)||(mxGetM(s)!=ncol)) {
//         mexErrMsgTxt("rcordmin.c : the input data has incoherent dimensions");
//     }
    
    /* Allocate output */
    s_out = mxCreateDoubleMatrix(ncol,1,mxREAL);
    Ax_out = mxCreateDoubleMatrix(mrow,1,mxREAL);
    
    /* I/O pointers */
	s = mxGetPr(s_in);
	q = mxGetPr(q_in);
	D = mxGetPr(Dx);
    A = mxGetPr(A_in);
    Ax = mxGetPr(Ax_in);
    ir = mxGetIr(A_in);      /* Row indexing      */
    jc = mxGetJc(A_in);      /* Column count      */
    mu = mxGetScalar(Mu);
	rp = mxGetScalar(Rp);
    
    so = mxGetPr(s_out);
    Axo = mxGetPr(Ax_out);
    
    

    

    
    /* Initialization */
    for(i=0;i<mrow;i++){
        Axo[i] = Ax[i];
    }
    for(i=0;i<ncol;i++){
        so[i] = s[i];
    }
    
       
    /* A cycle of Coordimate Minimization */
    for(i=0; i<ncol; i++){
        /* First calculate H_jj and a'*D*Axh */
        H_ii = 0;
        adai = 0;
        stop = jc[i+1];
        for(k=jc[i]; k<stop; k++) {
            H_ii += D[ir[k]]*A[k]*A[k];
            adai += D[ir[k]]*A[k]*Axo[ir[k]];
        }


        /* Second calculate temp */
        tmp = (q[i] - adai + H_ii*s[i])/(H_ii + mu);


        td = rp/(H_ii + mu);

        if(tmp>td){
            so[i] = tmp - td;
        }
        else if(tmp<-td){
            so[i] = tmp + td;
        }
        else {
            so[i] = 0;
        }

        /* Update Axo */
        for(k=jc[i]; k<stop; k++) {
            Axo[ir[k]] += (so[i] - s[i])*A[k];
        }

    }

    return;
}
