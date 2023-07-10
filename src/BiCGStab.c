#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <hb_io.h>

#include "reloj.h"
#include "ScalarVectors.h"
#include "SparseProduct.h"
#include "ToolsMPI.h"
#include "matrix.h"
#include "common.h"

#include <cstddef>
#include <mpfr.h>

#define DIRECT_ERROR 0
#define PRECOND 1
#define VECTOR_OUTPUT 0

double dot_mpfr(int *N, double *a, int *inca, double *b, int *incb) {
    mpfr_t sum, dot, op1, op2;
    mpfr_init2(op1, 64);
    mpfr_init2(op2, 64);
    mpfr_init2(dot, 192);
    mpfr_init2(sum, 2048);

    mpfr_set_zero(sum, 0.0);

    for (int i = 0; i < *N; i++) {
        mpfr_set_d(op1, a[i], MPFR_RNDN);
        mpfr_set_d(op2, b[i], MPFR_RNDN);

        mpfr_set_zero(dot, 0.0);
        mpfr_mul(dot, op1, op2, MPFR_RNDN);

        mpfr_add(sum, sum, dot, MPFR_RNDN);
    }
    double dacc = mpfr_get_d(sum, MPFR_RNDN);

    mpfr_clear(op1);
    mpfr_clear(op2);
    mpfr_clear(dot);
    mpfr_clear(sum);
    mpfr_free_cache();

    return dacc;
}

// ================================================================================

void BiCGStab (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, int myId) {
    int size = mat.dim2, sizeR = mat.dim1; 
    int IONE = 1; 
    double DONE = 1.0, DMONE = -1.0, DZERO = 0.0;
    int n, n_dist, iter, maxiter, nProcs;
    double beta, tol, tol0, alpha, umbral, rho, omega, tmp;
    double *s = NULL, *q = NULL, *r = NULL, *p = NULL, *r0 = NULL, *y = NULL, *p_hat = NULL, *q_hat = NULL;
    double *aux = NULL;
    double t1, t2, t3, t4;
    double reduce[2];
#if PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    n = size; n_dist = sizeR; maxiter = 16 * size; umbral = 1.0e-6;
    CreateDoubles (&s, n_dist);
    CreateDoubles (&q, n_dist);
    CreateDoubles (&r, n_dist);
    CreateDoubles (&r0, n_dist);
    CreateDoubles (&p, n_dist);
    CreateDoubles (&y, n_dist);
#if DIRECT_ERROR
    // init exact solution
    double *res_err = NULL, *x_exact = NULL;
    CreateDoubles (&x_exact, n_dist);
    CreateDoubles (&res_err, n_dist);
    InitDoubles (x_exact, n_dist, DONE, DZERO);
#endif // DIRECT_ERROR 

#if PRECOND
    CreateInts (&posd, n_dist);
    CreateDoubles (&p_hat, n_dist);
    CreateDoubles (&q_hat, n_dist);
    CreateDoubles (&diags, n_dist);
    GetDiagonalSparseMatrix2 (mat, dspls[myId], diags, posd);
#pragma omp parallel for
    for (i=0; i<n_dist; i++) 
        diags[i] = DONE / diags[i];
#endif
    CreateDoubles (&aux, n); 

#if VECTOR_OUTPUT
    // write to file for testing purpose
    FILE *fp;
    if (myId == 0) {
        char name[50] = "mpfr-1.txt";
        fp = fopen(name,"w");
    }
#endif

    iter = 0;
    MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (s, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, s);            			// s = A * x

    // r = b - s
    for (int jj = 0; jj < n_dist; jj++) {
        r[jj] = b[jj] - s[jj];    
        r0[jj] = r[jj];                                                   // ro = r
        p[jj] = r[jj];                                                    // p = r
    }    

    beta = 0.0;
    omega = 0.0;

    // compute tolerance and <r0,r0>
    rho = dot_mpfr (&n_dist, r, &IONE, r, &IONE);                           // tol = r' * r
    if (myId == 0)
        printf("rho0 = %20.10e\n", rho);
    tol0 = sqrt (rho);
    tol = tol0; 

#if DIRECT_ERROR
    // compute direct error
    double direct_err;
    for (int jj = 0; jj < n_dist; jj++)
        res_err[jj] = x_exact[jj] - x[jj];

    // compute inf norm
    direct_err = norm_inf(n_dist, res_err);

    //    // compute euclidean norm
    //    direct_err = ddot (&n_dist, res_err, &IONE, res_err, &IONE);            // direct_err = res_err' * res_err
    //    direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) 
        reloj (&t1, &t2);

    while ((iter < maxiter) && (tol > umbral)) {

#if PRECOND
        VvecDoubles (DONE, diags, p, DZERO, p_hat, n_dist);              // p_hat = D^-1 * p
#else
        p_hat = p;
#endif
        MPI_Allgatherv (p_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (s, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, s);            	    // s = A * p

        if (myId == 0) 
#if DIRECT_ERROR
            printf ("%d \t %a \t %a \n", iter, tol, direct_err);
#else        
        printf ("%d \t %a \n", iter, tol);
#endif // DIRECT_ERROR

        alpha = dot_mpfr (&n_dist, r0, &IONE, s, &IONE);                // alpha = <r_0, r_iter> / <r_0, s>
        alpha = rho / alpha;

		// q = r - alpha * s
        tmp = -alpha;
        for (int jj = 0; jj < n_dist; jj++) 
			q[jj] = fma(tmp, s[jj], r[jj]);

        // second spmv
#if PRECOND
        VvecDoubles (DONE, diags, q, DZERO, q_hat, n_dist);             // q_hat = D^-1 * q
#else
        q_hat = q;
#endif
        MPI_Allgatherv (q_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (y, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, y);            		// y = A * q

        // omega = <q, y> / <y, y>
        reduce[0] = dot_mpfr (&n_dist, q, &IONE, y, &IONE);
        reduce[1] = dot_mpfr (&n_dist, y, &IONE, y, &IONE);
        omega = reduce[0] / reduce[1];

        // x+1 = x + alpha * p + omega * q
        for (int jj = 0; jj < n_dist; jj++) { 
			x[jj] = fma(alpha, p_hat[jj], x[jj]);
			x[jj] = fma(omega, q_hat[jj], x[jj]);
		}

        // r+1 = q - omega * y
        tmp = -omega;
        for (int jj = 0; jj < n_dist; jj++)
            r[jj] = fma(tmp, y[jj], q[jj]);
        
        // rho = <r0, r+1> and tolerance
        // cannot just use <r0, r> as the stopping criteria since it slows the convergence compared to <r, r>
        reduce[0] = dot_mpfr (&n_dist, r0, &IONE, r, &IONE);
        tmp = reduce[0];
        tol = sqrt (fabs(tmp)) / tol0;

        // beta = (alpha / omega) * <r0, r+1> / <r0, r>
        beta = (alpha / omega) * (tmp / rho);
        rho = tmp;
       
        // p+1 = r+1 + beta * (p - omega * s)
        tmp = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            p[jj] = fma(tmp, s[jj], p[jj]);
            p[jj] = fma(beta, p[jj], r[jj]);
        }

#if DIRECT_ERROR
        // compute direct error
        for (int jj = 0; jj < n_dist; jj++)
            res_err[jj] = x_exact[jj] - x[jj];

        // compute inf norm
        direct_err = norm_inf(n_dist, res_err);

        //        // compute euclidean norm
        //        direct_err = ddot (&n_dist, res_err, &IONE, res_err, &IONE);
        //        direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) 
        reloj (&t3, &t4);

#if VECTOR_OUTPUT
    // print aux
    MPI_Allgatherv (x, n_dist, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    if (myId == 0) {
        fprintf(fp, "%d\n", iter);
        for (int ip = 0; ip < n; ip++)
            fprintf(fp, "%a\n", aux[ip]);
        fprintf(fp, "\n");
        fclose(fp);
    }
#endif

    if (myId == 0) {
        printf ("Size: %d \n", n);
        printf ("Iter: %d \n", iter);
        printf ("Tol: %a \n", tol);
        printf ("Time_loop: %20.10e\n", (t3-t1));
        printf ("Time_iter: %20.10e\n", (t3-t1)/iter);
    }

    RemoveDoubles (&aux); RemoveDoubles (&s); RemoveDoubles (&q); 
    RemoveDoubles (&r); RemoveDoubles (&p); RemoveDoubles (&r0); RemoveDoubles (&y);
#if PRECOND
    RemoveDoubles (&diags); RemoveInts (&posd);
    RemoveDoubles(&p_hat); RemoveDoubles (&q_hat); 
#endif
}

/*********************************************************************************/

int main (int argc, char **argv) {
    int dim; 
    double *sol1 = NULL, *sol2 = NULL;
    int index = 0, indexL = 0;
    SparseMatrix mat  = {0, 0, NULL, NULL, NULL}, sym = {0, 0, NULL, NULL, NULL};

    int root = 0, myId, nProcs;
    int dimL, dspL, *vdimL = NULL, *vdspL = NULL;
    SparseMatrix matL = {0, 0, NULL, NULL, NULL};
    double *sol1L = NULL, *sol2L = NULL;

    int mat_from_file, nodes, size_param, stencil_points;

    if (argc == 3) {
        mat_from_file = atoi(argv[2]);
    } else {
        mat_from_file = atoi(argv[2]);
        nodes = atoi(argv[3]);
        size_param = atoi(argv[4]);
        stencil_points = atoi(argv[5]);
    }

    /***************************************/

    MPI_Init (&argc, &argv);

    // Definition of the variables nProcs and myId
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    root = nProcs-1;
    root = 0;

    /***************************************/

    CreateInts (&vdimL, nProcs); CreateInts (&vdspL, nProcs); 
    if(mat_from_file) {
        if (myId == root) {
            // Creating the matrix
            ReadMatrixHB (argv[1], &sym);
            TransposeSparseMatrices (sym, 0, &mat, 0);
            dim = mat.dim1;
        }

        // Distributing the matrix
        dim = DistributeMatrix (mat, index, &matL, indexL, vdimL, vdspL, root, MPI_COMM_WORLD);
        dimL = vdimL[myId]; dspL = vdspL[myId];
    }
    else {
        dim = size_param * size_param * size_param;
        int divL, rstL, i;
        divL = (dim / nProcs); rstL = (dim % nProcs);
        for (i=0; i<nProcs; i++) vdimL[i] = divL + (i < rstL);
        vdspL[0] = 0; for (i=1; i<nProcs; i++) vdspL[i] = vdspL[i-1] + vdimL[i-1];
        dimL = vdimL[myId]; dspL = vdspL[myId];
        int band_width = size_param * (size_param + 1) + 1;
        band_width = 100 * nodes;
        long nnz_here = ((long) (stencil_points + 2 * band_width)) * dimL;
        printf ("dimL: %d, nodes: %d, size_param: %d, band_width: %d, stencil_points: %d, nnz_here: %ld\n",
                dimL, nodes, size_param, band_width, stencil_points, nnz_here);
        allocate_matrix(dimL, dim, nnz_here, &matL);
        generate_Poisson3D_filled(&matL, size_param, stencil_points, band_width, dspL, dimL, dim);

        // To generate ill-conditioned matrices
        double factor = 1.0e6;
        ScaleFirstRowCol(matL, dspL, dimL, myId, root, factor);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Creating the vectors
    CreateDoubles (&sol1, dim);
    CreateDoubles (&sol2, dim);
    CreateDoubles (&sol1L, dimL);
    CreateDoubles (&sol2L, dimL);
    InitDoubles (sol1, dim, 1.0, 0.0);
    InitDoubles (sol2, dim, 0.0, 0.0);
    InitDoubles (sol1L, dimL, 0.0, 0.0);
    InitDoubles (sol2L, dimL, 0.0, 0.0);

    /***************************************/

    int IONE = 1;
//    for (int i=0; i < matL.dim1; i++) {
//        for(int j=vptrM[i]; j<vptrM[i+1]; j++) {
//            sol1L[k] += matL.vval[j];
//        }
//    }

    // compute b = A * x_c, x_c = 1/sqrt(nbrows)
    ProdSparseMatrixVectorByRows (matL, 0, sol1, sol1L);            			// s = A * x
    double beta = 1.0 / sqrt(dim);
    for (int jj = 0; jj < dimL; jj++)                                           // s = beta * s
    	sol1L[jj] = beta * sol1L[jj];

    MPI_Scatterv (sol2, vdimL, vdspL, MPI_DOUBLE, sol2L, dimL, MPI_DOUBLE, root, MPI_COMM_WORLD);

    BiCGStab (matL, sol2L, sol1L, vdimL, vdspL, myId);

    // Error computation ||b-Ax||
    // case with x_exact = {1.0}
//    for (i=0; i<dimL; i++)
//        sol2L[i] -= 1.0;
//    beta = ddot (&dimL, sol2L, &IONE, sol2L, &IONE);            
    InitDoubles (sol2, dimL, 0, 0);
    ProdSparseMatrixVectorByRows (matL, 0, sol2L, sol2);            			// s = A * x
    for (int jj = 0; jj < dimL; jj++)                                           // r -= s
        sol1L[jj] = sol1L[jj] - sol2L[jj];
    beta = dot_mpfr (&dimL, sol1L, &IONE, sol1L, &IONE);            beta = sqrt(beta);
    if (myId == 0) 
        printf ("Error: %20.10e\n", beta);

    /***************************************/
    // Freeing memory
    RemoveDoubles (&sol1); 
    RemoveDoubles (&sol2); 
    RemoveDoubles (&sol1L); 
    RemoveDoubles (&sol2L);
    RemoveInts (&vdspL); RemoveInts (&vdimL); 
    if (myId == root) {
        RemoveSparseMatrix (&mat);
        RemoveSparseMatrix (&sym);
    } 

    MPI_Finalize ();

    return 0;
}

