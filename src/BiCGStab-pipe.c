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

void BiCGStab (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, int myId) {
    int size = mat.dim2, sizeR = mat.dim1; 
    int IONE = 1; 
    double DONE = 1.0, DMONE = -1.0, DZERO = 0.0;
    int n, n_dist, iter, maxiter, nProcs;
    double beta, tol, tol0, alpha, umbral, rho, omega, tmp;
    double *s = NULL, *q = NULL, *r = NULL, *p = NULL, *r0 = NULL, *y = NULL, *p_hat = NULL, *q_hat = NULL;
    double *r_hat = NULL, *z = NULL, *t = NULL, *z_hat = NULL, *w = NULL, *w_hat = NULL, *s_hat = NULL, *v = NULL;
    double *aux = NULL;
    double t1, t2, t3, t4;
    double reduce[5];
#if PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    n = size; n_dist = sizeR; maxiter = 16 * size; umbral = 1.0e-8;
    CreateDoubles (&s, n_dist);
    CreateDoubles (&q, n_dist);
    CreateDoubles (&r, n_dist);
    CreateDoubles (&r0, n_dist);
    CreateDoubles (&p, n_dist);
    CreateDoubles (&y, n_dist);
    CreateDoubles (&z, n_dist);
    CreateDoubles (&w, n_dist);
    CreateDoubles (&t, n_dist);
    CreateDoubles (&v, n_dist);
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
    CreateDoubles (&r_hat, n_dist);
    CreateDoubles (&w_hat, n_dist);
    CreateDoubles (&s_hat, n_dist);
    CreateDoubles (&z_hat, n_dist);
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
        char name[50];
        sprintf(name, "exblas-%d.txt", nProcs);
        fp = fopen(name,"w");
    }
#endif

    // r0 = b - Ax0
    MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (s, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, s);            			// s = A * x
    dcopy (&n_dist, b, &IONE, r, &IONE);                                // r = b
    daxpy (&n_dist, &DMONE, s, &IONE, r, &IONE);                        // r -= s

    // w0 = A * r0 
#if PRECOND
    VvecDoubles (DONE, diags, r, DZERO, r_hat, n_dist);                 // r_hat = D^-1 * r
#else
    r_hat = r;
#endif
    MPI_Allgatherv (r_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (w, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, w);            			// w = A * r_hat

    // t0 = A * w0 
#if PRECOND
    VvecDoubles (DONE, diags, w, DZERO, w_hat, n_dist);                 // w_hat = D^-1 * w
#else
    w_hat = w;
#endif
    MPI_Allgatherv (w_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (t, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, t);            			// t = A * w_hat

    dcopy (&n_dist, r, &IONE, r0, &IONE);                               // r0 = r
    dcopy (&n_dist, r, &IONE, p, &IONE);                                // p = r
    dcopy (&n_dist, w, &IONE, s, &IONE);                                // s = w
    dcopy (&n_dist, t, &IONE, z, &IONE);                                // z = t
    dcopy (&n_dist, r, &IONE, q, &IONE);                                // q = r
    dcopy (&n_dist, w, &IONE, y, &IONE);                                // y = w

    // superaccs
    std::vector<int64_t> h_superacc(5 * exblas::BIN_COUNT);
    std::vector<int64_t> h_superacc_1(exblas::BIN_COUNT);
    std::vector<int64_t> h_superacc_2(exblas::BIN_COUNT);
    std::vector<int64_t> h_superacc_3(exblas::BIN_COUNT);
    std::vector<int64_t> h_superacc_4(exblas::BIN_COUNT);

    // compute tolerance, <r0,r0>, and <r0, w0>
    // alpha = (r0, r0) / (r0, w0)
    int imin=exblas::IMIN, imax=exblas::IMAX;
    exblas::cpu::exdot (n_dist, r, r, &h_superacc[0]);
    exblas::cpu::exdot (n_dist, r, w, &h_superacc_1[0]);
    // ReproAllReduce -- Begin
    exblas::cpu::Normalize(&h_superacc[0], imin, imax);
    exblas::cpu::Normalize(&h_superacc_1[0], imin, imax);
    // merge two superaccs into one for reduction
    for (int i = 0; i < exblas::BIN_COUNT; i++) {
        h_superacc[exblas::BIN_COUNT + i] = h_superacc_1[i]; 
    } 
    if (myId == 0) {
        MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce (&h_superacc[0], NULL, 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (myId == 0) {
        // split them back
        for (int i = 0; i < exblas::BIN_COUNT; i++) {
            h_superacc_1[i] = h_superacc[exblas::BIN_COUNT + i]; 
        } 
        reduce[0] = exblas::cpu::Round( &h_superacc[0] );
        reduce[1] = exblas::cpu::Round( &h_superacc_1[0] );
    }
    MPI_Bcast(reduce, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ReproAllReduce -- End
    rho = reduce[0];
    alpha = rho / reduce[1];
    tol0 = sqrt(rho);
    tol = tol0;

    // beta = 0
    beta = 0.0;
    // omeg = 0
    omega = 0.0;

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

        if (myId == 0) 
#if DIRECT_ERROR
            printf ("%d \t %a \t %a \n", iter, tol, direct_err);
#else        
        printf ("%d \t %a \n", iter, tol);
#endif // DIRECT_ERROR

        // p = r + beta-1 * (p-1 - omega-1 * s-1)
        tmp = -omega; 
        daxpy (&n_dist, &tmp, s, &IONE, p, &IONE);                     // p -= omega * s
        dscal (&n_dist, &beta, p, &IONE);                              // p = beta * p
        daxpy (&n_dist, &DONE, r, &IONE, p, &IONE);                    // p += r

        // s = w + beta-1 * (s-1 - omega-1 * z-1)
        tmp = -omega; 
        daxpy (&n_dist, &tmp, z, &IONE, s, &IONE);                     // s -= omega * z
        dscal (&n_dist, &beta, s, &IONE);                              // s = beta * s
        daxpy (&n_dist, &DONE, w, &IONE, s, &IONE);                    // s += w

        // z = t + beta-1 * (z-1 - omega-1 * v-1)
        tmp = -omega; 
        daxpy (&n_dist, &tmp, v, &IONE, z, &IONE);                     // z -= omega * v
        dscal (&n_dist, &beta, z, &IONE);                              // z = beta * z
        daxpy (&n_dist, &DONE, t, &IONE, z, &IONE);                    // z += t

        // q = r - alpha * s 
        dcopy (&n_dist, r, &IONE, q, &IONE);                            // q = r
        tmp = -alpha;
        daxpy (&n_dist, &tmp, s, &IONE, q, &IONE);                      // q = r - alpha * s;

        // y = w - alpha * z 
        dcopy (&n_dist, w, &IONE, y, &IONE);                            // y = w
        tmp = -alpha;
        daxpy (&n_dist, &tmp, z, &IONE, y, &IONE);                      // y = w - alpha * z;

        // omega = <q, y> / <y, y>
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, q, y, &fpe[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, y, y, &fpe1[0]);
        // ReproAllReduce -- Begin
        // merge two fpes
        for (int i = 0; i < NBFPE; i++) { 
            fpe[NBFPE + i] = fpe1[i];
        }
        if (myId == 0) {
            MPI_Reduce (MPI_IN_PLACE, &fpe[0], 2*NBFPE, MPI_DOUBLE, Op5, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce (&fpe[0], NULL, 2*NBFPE, MPI_DOUBLE, Op5, 0, MPI_COMM_WORLD);
        }
        if (myId == 0) {
            // split two fpes
            for (int i = 0; i < NBFPE; i++) { 
                fpe1[i] = fpe[NBFPE + i];
            }
            reduce[0] = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
            reduce[1] = exblas::cpu::Round<double, NBFPE> (&fpe1[0]);
        }
        MPI_Bcast(reduce, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ReproAllReduce -- End
        omega = reduce[0] / reduce[1];

#if PRECOND
        VvecDoubles (DONE, diags, z, DZERO, z_hat, n_dist);              // z_hat = D^-1 * z

        // p_hat = r_hat + beta-1 * (p_hat-1 - omega-1 * s_hat-1)
        tmp = -omega; 
        daxpy (&n_dist, &tmp, s_hat, &IONE, p_hat, &IONE);               // p_hat -= omega * s_hat
        dscal (&n_dist, &beta, p_hat, &IONE);                            // p_hat = beta * p_hat
        daxpy (&n_dist, &DONE, r_hat, &IONE, p_hat, &IONE);              // p_hat += r_hat

        // s_hat = w_hat + beta-1 * (s_hat-1 - omega-1 * z_hat-1)
        tmp = -omega; 
        daxpy (&n_dist, &tmp, z_hat, &IONE, s_hat, &IONE);               // s_hat -= omega * z_hat
        dscal (&n_dist, &beta, s_hat, &IONE);                            // s_hat = beta * s_hat
        daxpy (&n_dist, &DONE, w_hat, &IONE, s_hat, &IONE);              // s_hat += w_hat

        // q_hat = r_hat - alpha * s_hat 
        dcopy (&n_dist, r_hat, &IONE, q_hat, &IONE);                     // q_hat = r_hat
        tmp = -alpha;
        daxpy (&n_dist, &tmp, s_hat, &IONE, q_hat, &IONE);               // q_hat = q_hat - alpha * s_hat;
#else
        z_hat = z;
#endif
        MPI_Allgatherv (z_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (v, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, v);            	    // v = A * z_hat

        // x+1 = x + alpha * p + omega * q
        daxpy (&n_dist, &alpha, p, &IONE, x, &IONE); 
        daxpy (&n_dist, &omega, q, &IONE, x, &IONE); 

        // r+1 = q - omega * y
        dcopy (&n_dist, q, &IONE, r, &IONE);                            // r = q
        tmp = -omega;
        daxpy (&n_dist, &tmp, y, &IONE, r, &IONE);                      // r = q - omega * y;
       
        // w+1 = y - omega * (t - alpha * v)
        dcopy (&n_dist, t, &IONE, w, &IONE);                            // w = t
        tmp = -alpha; 
        daxpy (&n_dist, &tmp, v, &IONE, w, &IONE);                      // w -= alpha * v
        tmp = -omega; 
        dscal (&n_dist, &tmp, w, &IONE);                               // w = -omega * w
        daxpy (&n_dist, &DONE, y, &IONE, w, &IONE);                     // w += y

        // t = A w
#if PRECOND
        VvecDoubles (DONE, diags, w, DZERO, w_hat, n_dist);             // w_hat = D^-1 * w
       
        // r_hat+1 = q_hat - omega * (w_hat - alpha * z_hat)
        dcopy (&n_dist, w_hat, &IONE, r_hat, &IONE);                    // r_hat = w_hat
        tmp = -alpha; 
        daxpy (&n_dist, &tmp, z, &IONE, r_hat, &IONE);                  // r_hat -= alpha * z
        tmp = -omega; 
        dscal (&n_dist, &tmp, r_hat, &IONE);                            // r_hat = -omega * r_hat
        daxpy (&n_dist, &DONE, q, &IONE, r_hat, &IONE);                 // r_hat += y
#else
        w_hat = w;
#endif
        MPI_Allgatherv (w_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (t, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, t);            	    // t = A * w

        // beta = (alpha / omega) * <r0, r+1> / <r0, r>
        // rho = <r0, r+1> and tolerance
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, r, &fpe[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, w, &fpe1[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, s, &fpe2[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, z, &fpe3[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r, r, &fpe4[0]);
        // ReproAllReduce -- Begin
        // merge two fpes
        for (int i = 0; i < NBFPE; i++) { 
            fpe[NBFPE + i] = fpe1[i];
            fpe[2*NBFPE + i] = fpe2[i];
            fpe[3*NBFPE + i] = fpe3[i];
            fpe[4*NBFPE + i] = fpe4[i];
        }
        if (myId == 0) {
            MPI_Reduce (MPI_IN_PLACE, &fpe[0], 5*NBFPE, MPI_DOUBLE, Op5, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce (&fpe[0], NULL, 5*NBFPE, MPI_DOUBLE, Op5, 0, MPI_COMM_WORLD);
        }
        if (myId == 0) {
            // split two fpes
            for (int i = 0; i < NBFPE; i++) { 
                fpe1[i] = fpe[NBFPE + i];
                fpe2[i] = fpe[2*NBFPE + i];
                fpe3[i] = fpe[3*NBFPE + i];
                fpe4[i] = fpe[4*NBFPE + i];
            }
            reduce[0] = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
            reduce[1] = exblas::cpu::Round<double, NBFPE> (&fpe1[0]);
            reduce[2] = exblas::cpu::Round<double, NBFPE> (&fpe2[0]);
            reduce[3] = exblas::cpu::Round<double, NBFPE> (&fpe3[0]);
            reduce[4] = exblas::cpu::Round<double, NBFPE> (&fpe4[0]);
        }
        MPI_Bcast(reduce, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ReproAllReduce -- End
        tmp = reduce[0];
        tol = sqrt(reduce[4]) / tol0;
        beta = (alpha / omega) * (tmp / rho);
        rho = tmp;

        // alpha = <r0, r+1> / (<r0, w+1> + beta <r0, s> - beta omega <r0, z>)
        alpha = rho / (reduce[1] + beta * (reduce[2] - omega * reduce[3]));

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

    MPI_Op_free( &Op );
    MPI_Op_free( &Op5 );

    RemoveDoubles (&aux); RemoveDoubles (&s); RemoveDoubles (&q); 
    RemoveDoubles (&r); RemoveDoubles (&p); RemoveDoubles (&r0); RemoveDoubles (&y);
    RemoveDoubles (&z); RemoveDoubles (&w); RemoveDoubles (&t); RemoveDoubles (&v);
#if PRECOND
    RemoveDoubles (&diags); RemoveInts (&posd);
    RemoveDoubles(&p_hat); RemoveDoubles (&q_hat); 
    RemoveDoubles(&r_hat); RemoveDoubles (&w_hat); 
    RemoveDoubles(&s_hat); RemoveDoubles (&z_hat); 
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
        ProdSparseMatrixVectorByRows (matL, 0, sol2, sol2L);            			// s = A * x
        double DMONE = -1.0;
        daxpy (&dimL, &DMONE, sol2L, &IONE, sol1L, &IONE);                          // r -= s

        std::vector<double> fpe(NBFPE);
        exblas::cpu::exdot<double*, double*, NBFPE> (dimL, sol1L, sol1L, &fpe[0]);

        // ReproAllReduce -- Begin
        // user-defined reduction operations
        MPI_Op Op;
        MPI_Op_create( (MPI_User_function *) fpeSum, 1, &Op ); 
        if (myId == 0) {
            MPI_Reduce (MPI_IN_PLACE, &fpe[0], NBFPE, MPI_DOUBLE, Op, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce (&fpe[0], NULL, NBFPE, MPI_DOUBLE, Op, 0, MPI_COMM_WORLD);
        }
        MPI_Op_free( &Op );
        if (myId == 0) {
            beta = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
        }
        MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

