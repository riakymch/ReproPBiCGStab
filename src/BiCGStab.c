#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mkl_blas.h>
#include <mpi.h>
#include <hb_io.h>
#include <vector>

#include "reloj.h"
#include "ScalarVectors.h"
#include "SparseProduct.h"
#include "ToolsMPI.h"
#include "matrix.h"
#include "common.h"

#include "exblas/exdot.h"

// ================================================================================

#define DIRECT_ERROR 0
#define PRECOND 1
#define VECTOR_OUTPUT 0
#define SPMV_OPTIMIZED 1
#ifdef SPMV_OPTIMIZED
	#define COLL_P2P_SPMV 0
#endif

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

#if VECTOR_OUTPUT
    // write to file for testing purpose
    FILE *fp;
    if (myId == 0) {
        char name[50];
        sprintf(name, "exblas-%d.txt", nProcs);
        fp = fopen(name,"w");
    }
#endif

#ifdef SPMV_OPTIMIZED
    int *permP = NULL, *ipermP = NULL;
    int *vdspP = NULL, *vdimP = NULL, *vdspR = NULL, *vdimR = NULL;
    double *vecP = NULL;
    MPI_Datatype *vectDatatypeP = NULL, *vectDatatypeR = NULL;

    CreateInts (&ipermP, size);
    CreateInts (&vdimP, nProcs); CreateInts (&vdspP, nProcs + 1);
    CreateInts (&vdimR, nProcs); CreateInts (&vdspR, nProcs + 1);
    vectDatatypeP = (MPI_Datatype *) malloc (nProcs * sizeof(MPI_Datatype));
    vectDatatypeR = (MPI_Datatype *) malloc (nProcs * sizeof(MPI_Datatype));
    createAlltoallwStruct (COLL_P2P_SPMV, MPI_COMM_WORLD, mat, sizes, dspls, vdimP, vdspP, &vecP, &permP, ipermP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);

    // Code required before the loop  
    PermuteInts (mat.vpos, ipermP, mat.vptr[mat.dim1]);
#else
    CreateDoubles (&aux, n); 
#endif

#ifdef SPMV_OPTIMIZED
    joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, x, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
    InitDoubles (s, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, vecP, s);            			// s = A * x
#else
    MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (s, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, s);            			// s = A * x
#endif

    dcopy (&n_dist, b, &IONE, r, &IONE);                                // r = b
    daxpy (&n_dist, &DMONE, s, &IONE, r, &IONE);                        // r -= s

    dcopy (&n_dist, r, &IONE, p, &IONE);                                // p = r
    dcopy (&n_dist, r, &IONE, r0, &IONE);                               // r0 = r

    // compute tolerance and <r0,r0>
    std::vector<int64_t> h_superacc(2 * exblas::BIN_COUNT);
    std::vector<int64_t> h_superacc_tol(exblas::BIN_COUNT);
    int imin=exblas::IMIN, imax=exblas::IMAX;
    exblas::cpu::exdot (n_dist, r, r, &h_superacc[0]);
    // ReproAllReduce -- Begin
    exblas::cpu::Normalize(&h_superacc[0], imin, imax);
    MPI_Allreduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    rho = exblas::cpu::Round( &h_superacc[0] );
    // ReproAllReduce -- End
    tol0 = sqrt (rho);
    tol = tol0;

#if DIRECT_ERROR
    // compute direct error
    double direct_err;
    dcopy (&n_dist, x_exact, &IONE, res_err, &IONE);                    // res_err = x_exact
    daxpy (&n_dist, &DMONE, x, &IONE, res_err, &IONE);                  // res_err -= x

    // compute inf norm
    direct_err = norm_inf(n_dist, res_err);
    MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    //    // compute euclidean norm
    //    direct_err = ddot (&n_dist, res_err, &IONE, res_err, &IONE);            // direct_err = res_err' * res_err
    //    MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //    direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) 
        reloj (&t1, &t2);

    iter = 0;
    while ((iter < maxiter) && (tol > umbral)) {

#if PRECOND
        VvecDoubles (DONE, diags, p, DZERO, p_hat, n_dist);              // p_hat = D^-1 * p
#else
        p_hat = p;
#endif
#ifdef SPMV_OPTIMIZED
        joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, p_hat, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
        InitDoubles (s, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, vecP, s);            	     // s = A * p
#else
        MPI_Allgatherv (p_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (s, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, s);            	     // s = A * p
#endif

        if (myId == 0) 
#if DIRECT_ERROR
            printf ("%d \t %a \t %a \n", iter, tol, direct_err);
#else        
        printf ("%d \t %a \n", iter, tol);
#endif // DIRECT_ERROR

        exblas::cpu::exdot (n_dist, r0, s, &h_superacc[0]);             // alpha = <r_0, r_iter> / <r_0, s>
        // ReproAllReduce -- Begin
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
        MPI_Allreduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        alpha = exblas::cpu::Round( &h_superacc[0] );
        // ReproAllReduce -- End
        alpha = rho / alpha;

        dcopy (&n_dist, r, &IONE, q, &IONE);                            // q = r
        tmp = -alpha;
        daxpy (&n_dist, &tmp, s, &IONE, q, &IONE);                      // q = r - alpha * s;

        // second spmv
#if PRECOND
        VvecDoubles (DONE, diags, q, DZERO, q_hat, n_dist);             // q_hat = D^-1 * q
#else
        q_hat = q;
#endif
#ifdef SPMV_OPTIMIZED
        joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, q_hat, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
        InitDoubles (y, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, vecP, y);            		// y = A * q
#else
        MPI_Allgatherv (q_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (y, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, y);            		// y = A * q
#endif

        // omega = <q, y> / <y, y>
        exblas::cpu::exdot (n_dist, q, y, &h_superacc[0]);
        exblas::cpu::exdot (n_dist, y, y, &h_superacc_tol[0]);
        // ReproAllReduce -- Begin
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
        exblas::cpu::Normalize(&h_superacc_tol[0], imin, imax);
        // merge two superaccs into one for reduction
        for (int i = 0; i < exblas::BIN_COUNT; i++) {
            h_superacc[exblas::BIN_COUNT + i] = h_superacc_tol[i]; 
        } 
        MPI_Allreduce (MPI_IN_PLACE, &h_superacc[0], 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        // split them back
        for (int i = 0; i < exblas::BIN_COUNT; i++) {
            h_superacc_tol[i] = h_superacc[exblas::BIN_COUNT + i]; 
        } 
        reduce[0] = exblas::cpu::Round( &h_superacc[0] );
        reduce[1] = exblas::cpu::Round( &h_superacc_tol[0] );
        // ReproAllReduce -- End
        omega = reduce[0] / reduce[1];

        // x+1 = x + alpha * p + omega * q
        daxpy (&n_dist, &alpha, p_hat, &IONE, x, &IONE); 
        daxpy (&n_dist, &omega, q_hat, &IONE, x, &IONE); 

        // r+1 = q - omega * y
        dcopy (&n_dist, q, &IONE, r, &IONE);                            // r = q
        tmp = -omega;
        daxpy (&n_dist, &tmp, y, &IONE, r, &IONE);                      // r = q - omega * y;
        
        // rho = <r0, r+1> and tolerance
        exblas::cpu::exdot (n_dist, r0, r, &h_superacc[0]);
        // ReproAllReduce -- Begin
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
        MPI_Allreduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        tmp = exblas::cpu::Round( &h_superacc[0] );
        // ReproAllReduce -- End
        tol = sqrt(fabs(tmp)) / tol0;

        // beta = (alpha / omega) * <r0, r+1> / <r0, r>
        beta = (alpha / omega) * (tmp / rho);
        rho = tmp;
       
        // p+1 = r+1 + beta * (p - omega * s)
        tmp = -omega; 
        daxpy (&n_dist, &tmp, s, &IONE, p, &IONE);                     // p -= omega * s
        dscal (&n_dist, &beta, p, &IONE);                              // p = beta * p
        daxpy (&n_dist, &DONE, r, &IONE, p, &IONE);                    // p += r

#if DIRECT_ERROR
        // compute direct error
        dcopy (&n_dist, x_exact, &IONE, res_err, &IONE);               // res_err = x_exact
        daxpy (&n_dist, &DMONE, x, &IONE, res_err, &IONE);             // res_err -= x

        // compute inf norm
        direct_err = norm_inf(n_dist, res_err);
        MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //        // compute euclidean norm
        //        direct_err = ddot (&n_dist, res_err, &IONE, res_err, &IONE);
        //        MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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

#ifdef SPMV_OPTIMIZED
  // Code required after the loop 
  PermuteInts (mat.vpos, permP, mat.vptr[mat.dim1]);

  // Freeing memory for Permutation
  free (vectDatatypeR); vectDatatypeR = NULL; free (vectDatatypeP); vectDatatypeP = NULL;
  RemoveDoubles (&vecP); RemoveInts (&permP);
  RemoveInts (&vdspR); RemoveInts (&vdimR); RemoveInts (&vdspP); RemoveInts (&vdimP);
  RemoveInts (&ipermP);
#else
  RemoveDoubles (&aux); 
#endif

    RemoveDoubles (&s); RemoveDoubles (&q); 
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
//        double factor = 1.0e6;
//        ScaleFirstRowCol(matL, dspL, dimL, myId, root, factor);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Creating the vectors
    CreateDoubles (&sol1, dim);
    CreateDoubles (&sol2, dim);
    CreateDoubles (&sol1L, dimL);
    CreateDoubles (&sol2L, dimL);

    InitDoubles (sol2, dim, 0.0, 0.0);
    InitDoubles (sol1L, dimL, 0.0, 0.0);
    InitDoubles (sol2L, dimL, 0.0, 0.0);

    /***************************************/

    int IONE = 1;
    double beta = 1.0 / sqrt(dim);
    if(mat_from_file) {
        // compute b = A * x_c, x_c = 1/sqrt(nbrows)
        InitDoubles (sol1, dim, 1.0, 0.0);
        ProdSparseMatrixVectorByRows (matL, 0, sol1, sol1L);            			// s = A * x
        dscal (&dimL, &beta, sol1L, &IONE);                                         // s = beta * s
    } else {
        InitDoubles (sol1, dim, 0.0, 0.0);

        int k=0;
        int *vptrM = matL.vptr;
        for (int i=0; i < matL.dim1; i++) {
            for(int j=vptrM[i]; j<vptrM[i+1]; j++) {
                sol1L[k] += matL.vval[j];
            }
        }
    }

    MPI_Scatterv (sol2, vdimL, vdspL, MPI_DOUBLE, sol2L, dimL, MPI_DOUBLE, root, MPI_COMM_WORLD);

    BiCGStab (matL, sol2L, sol1L, vdimL, vdspL, myId);

    // Error computation ||b-Ax||
//    if(mat_from_file) {
        MPI_Allgatherv (sol2L, dimL, MPI_DOUBLE, sol2, vdimL, vdspL, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (sol2L, dimL, 0, 0);
        ProdSparseMatrixVectorByRows (matL, 0, sol2, sol2L);
        double DMONE = -1.0;
        daxpy (&dimL, &DMONE, sol2L, &IONE, sol1L, &IONE);          

        // ReproAllReduce -- Begin
        std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
        exblas::cpu::exdot (dimL, sol1L, sol1L, &h_superacc[0]);
        int imin=exblas::IMIN, imax=exblas::IMAX;
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
        MPI_Allreduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        beta = exblas::cpu::Round( &h_superacc[0] );
        // ReproAllReduce -- End
        
//    } else {
//        // case with x_exact = {1.0}
//        for (int i=0; i<dimL; i++)
//            sol2L[i] -= 1.0;
//        beta = ddot (&dimL, sol2L, &IONE, sol2L, &IONE);            
//    } 

    beta = sqrt(beta);
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

