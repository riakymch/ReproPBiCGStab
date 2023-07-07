#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
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
#include "exblas/fpexpansionvect.hpp"

// ================================================================================

#define DIRECT_ERROR 0
#define PRECOND 1
#define VECTOR_OUTPUT 0
#define NBFPE 8
#define SPMV_OPTIMIZED 1
#ifdef SPMV_OPTIMIZED
	#define COLL_P2P_SPMV 0
#endif

/* 
 * operation to reduce fpes 
 */ 
void fpeSum( double *in, double *inout, int *len, MPI_Datatype *dptr ) { 

    double s;
    for (int j = 0; j < *len; ++j) { 
        if (in[j] == 0.0)
            return;

        for (int i = 0; i < *len; ++i) { 
            inout[i] = exblas::cpu::FMA2Sum(inout[i], in[j], s);
            in[j] = s;
            if(true && !(in[j] != 0))
                break;
        }
    }
}

void fpeSum5( double *in, double *inout, int *len, MPI_Datatype *dptr ) { 

    double s;
    int laps = *len / NBFPE;
    for (int k = 0; k < laps; k++) {
        int start = k * NBFPE;
        int end = (k+1) * NBFPE;
        for (int j = start; j < end; ++j) { 
            if (in[j] == 0.0)
                break;

            for (int i = start; i < end; ++i) { 
                inout[i] = exblas::cpu::FMA2Sum(inout[i], in[j], s);
                in[j] = s;
                if(true && !(in[j] != 0))
                    break;
            }
        }
    }
}

void BiCGStab (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, int myId) {
    int size = mat.dim2, sizeR = mat.dim1; 
    double DONE = 1.0, DZERO = 0.0;
    int n, n_dist, iter, maxiter, nProcs;
    double beta, tol, tol0, alpha, umbral, rho, omega, tmp, tmpo;
    double *s = NULL, *q = NULL, *r = NULL, *p = NULL, *r0 = NULL, *y = NULL, *p_hat = NULL, *q_hat = NULL;
    double *r_hat = NULL, *z = NULL, *t = NULL, *z_hat = NULL, *w = NULL, *w_hat = NULL, *s_hat = NULL, *v = NULL, *tmpv = NULL;
    double t1, t2, t3, t4;
    double *aux = NULL;
    double reduce[5];
#if PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif
    MPI_Request request;

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    n = size; n_dist = sizeR; maxiter = 16 * size; umbral = 1.0e-6;
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
    CreateDoubles (&tmpv, n_dist);
#if DIRECT_ERROR
    // init exact solution
    double *res_err = NULL, *x_exact = NULL;
    CreateDoubles (&x_exact, n_dist);
    CreateDoubles (&res_err, n_dist);
    InitDoubles (x_exact, n_dist, DONE, DZERO);
#endif // DIRECT_ERROR 

#if PRECOND
    CreateInts (&posd, n_dist);
    CreateDoubles (&r_hat, n_dist);
    CreateDoubles (&w_hat, n_dist);
    CreateDoubles (&z_hat, n_dist);
    CreateDoubles (&diags, n_dist);
    GetDiagonalSparseMatrix2 (mat, dspls[myId], diags, posd);
#pragma omp parallel for
    for (i=0; i<n_dist; i++) 
        diags[i] = DONE / diags[i];
#endif
    CreateDoubles (&p_hat, n_dist);
    CreateDoubles (&q_hat, n_dist);
    CreateDoubles (&s_hat, n_dist);

#if VECTOR_OUTPUT
    // write to file for testing purpose
    FILE *fp;
    if (myId == 0) {
        char name[50];
        sprintf(name, "fpe-%d.txt", nProcs);
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

    // r0 = b - Ax0
#ifdef SPMV_OPTIMIZED
    joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, x, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
    InitDoubles (s, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, vecP, s);            			// s = A * x
#else
    MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (s, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, s);            			// s = A * x
#endif

    // r = b - s
    for (int jj = 0; jj < n_dist; jj++)
        r[jj] = b[jj] - s[jj];    

    // w0 = A * r0 
#if PRECOND
    VvecDoubles (DONE, diags, r, DZERO, r_hat, n_dist);                 // r_hat = D^-1 * r
#else
    r_hat = r;
#endif
#ifdef SPMV_OPTIMIZED
    joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, r_hat, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
    InitDoubles (w, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, vecP, w);            			// s = A * x
#else
    MPI_Allgatherv (r_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (w, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, w);            			// w = A * r_hat
#endif

    // t0 = A * w0 
#if PRECOND
    VvecDoubles (DONE, diags, w, DZERO, w_hat, n_dist);                 // w_hat = D^-1 * w
#else
    w_hat = w;
    z_hat = z;
#endif
#ifdef SPMV_OPTIMIZED
    joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, w_hat, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
    InitDoubles (t, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, vecP, t);            			// s = A * x
#else
    MPI_Allgatherv (w_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (t, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, t);            			// t = A * w_hat
#endif

    // user-defined reduction operations
    MPI_Op Op, Op5;
    MPI_Op_create( (MPI_User_function *) fpeSum, 1, &Op ); 
    MPI_Op_create( (MPI_User_function *) fpeSum5, 1, &Op5 ); 
    // fpes
    std::vector<double> fpe(4*NBFPE);
    std::vector<double> fpe1(NBFPE);
    std::vector<double> fpe2(NBFPE);
    std::vector<double> fpe3(NBFPE);

    // compute tolerance, <r0,r0>, and <r0, w0>
    // alpha = (r0, r0) / (r0, w0)
    exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r, r, &fpe[0]);
    exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r, w, &fpe1[0]);
    // ReproAllReduce -- Begin
    // merge two fpes
    for (int i = 0; i < NBFPE; i++) { 
        fpe[NBFPE + i] = fpe1[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &fpe[0], 2*NBFPE, MPI_DOUBLE, Op5, MPI_COMM_WORLD);
    // split two fpes
    for (int i = 0; i < NBFPE; i++) { 
        fpe1[i] = fpe[NBFPE + i];
    }
    reduce[0] = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
    reduce[1] = exblas::cpu::Round<double, NBFPE> (&fpe1[0]);
    // ReproAllReduce -- End
    rho = reduce[0];
    alpha = rho / reduce[1];
    tol0 = sqrt(rho);
    tol = tol0;

    for (int jj = 0; jj < n_dist; jj++) {
        r0[jj] = r[jj];                                                   // ro = r
        p[jj] = r[jj];                                                    // p = r
        s[jj] = w[jj];                                                    // s = w  
        z[jj] = t[jj];                                                    // z = t  
    }    

    beta = 0.0;
    omega = 0.0;

#if DIRECT_ERROR
    // compute direct error
    double direct_err;
    for (int jj = 0; jj < n_dist; jj++)
        res_err[jj] = x_exact[jj] - x[jj];

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

        // p_hat = r_hat + beta-1 * (p_hat-1 - omega-1 * s_hat-1)
        tmp = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            p_hat[jj] = fma(tmp, s_hat[jj], p_hat[jj]);
            p_hat[jj] = fma(beta, p_hat[jj], r_hat[jj]);
        }   

        // s = w + beta-1 * (s-1 - omega-1 * z-1)
        tmp = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            s[jj] = fma(tmp, z[jj], s[jj]);
            s[jj] = fma(beta, s[jj], w[jj]);
        }

        // s_hat = w_hat + beta-1 * (s_hat-1 - omega-1 * z_hat-1)
        tmp = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            s_hat[jj] = fma(tmp, z_hat[jj], s_hat[jj]);
            s_hat[jj] = fma(beta, s_hat[jj], w_hat[jj]);
        }

        // z = t + beta-1 * (z-1 - omega-1 * v-1)
        tmp = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            z[jj] = fma(tmp, v[jj], z[jj]);
            z[jj] = fma(beta, z[jj], t[jj]);
        }

        // q = r - alpha * s 
        tmp = -alpha;
        for (int jj = 0; jj < n_dist; jj++)
            q[jj] = fma(tmp, s[jj], r[jj]);

        // q_hat = r_hat - alpha * s_hat 
        tmp = -alpha;
        for (int jj = 0; jj < n_dist; jj++)
            q_hat[jj] = fma(tmp, s_hat[jj], r_hat[jj]);

        // y = w - alpha * z 
        tmp = -alpha;
        for (int jj = 0; jj < n_dist; jj++)
            y[jj] = fma(tmp, z[jj], w[jj]);

        // omega = <q, y> / <y, y>
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, q, y, &fpe[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, y, y, &fpe1[0]);
        // ReproAllReduce -- Begin
        // merge two fpes
        for (int i = 0; i < NBFPE; i++) { 
            fpe[NBFPE + i] = fpe1[i];
        }
        MPI_Iallreduce(MPI_IN_PLACE, &fpe[0], 2*NBFPE, MPI_DOUBLE, Op5, MPI_COMM_WORLD, &request);

#if PRECOND
        VvecDoubles (DONE, diags, z, DZERO, z_hat, n_dist);              // z_hat = D^-1 * z
#else
        z_hat = z;
#endif
#ifdef SPMV_OPTIMIZED
        joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, z_hat, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
        InitDoubles (v, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, vecP, v);            	     // v = A * z_hat
#else
        MPI_Allgatherv (z_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (v, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, v);            	    // v = A * z_hat
#endif

        // wait for MPI_Iallreduce to complete
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        // ReproAllReduce -- End
        // split two fpes
        for (int i = 0; i < NBFPE; i++) { 
            fpe1[i] = fpe[NBFPE + i];
        }
        reduce[0] = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
        reduce[1] = exblas::cpu::Round<double, NBFPE> (&fpe1[0]);
        omega = reduce[0] / reduce[1];

        // x+1 = x + alpha * p_hat + omega * q_hat
        for (int jj = 0; jj < n_dist; jj++) {
            x[jj] = fma(alpha, p_hat[jj], x[jj]);
            x[jj] = fma(omega, q_hat[jj], x[jj]);
        }

        // r+1 = q - omega * y
        tmp = -omega;
        for (int jj = 0; jj < n_dist; jj++)
            r[jj] = fma(tmp, y[jj], q[jj]);
       
        // r_hat+1 = q_hat - omega * (w_hat - alpha * z_hat)
        tmp = -alpha; 
        tmpo = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            r_hat[jj] = fma(tmp, z_hat[jj], w_hat[jj]);
            r_hat[jj] = fma(tmpo, r_hat[jj], q_hat[jj]);
        }
       
        // w+1 = y - omega * (t - alpha * v)
        tmp = -alpha; 
        tmpo = -omega; 
        for (int jj = 0; jj < n_dist; jj++) {
            w[jj] = fma(tmp, v[jj], t[jj]);
            w[jj] = fma(tmpo, w[jj], y[jj]);
        }

        // beta = (alpha / omega) * <r0, r+1> / <r0, r>
        // rho = <r0, r+1> and tolerance
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, r, &fpe[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, w, &fpe1[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, s, &fpe2[0]);
        exblas::cpu::exdot<double*, double*, NBFPE> (n_dist, r0, z, &fpe3[0]);
        // ReproAllReduce -- Begin
        // merge two fpes
        for (int i = 0; i < NBFPE; i++) { 
            fpe[NBFPE + i] = fpe1[i];
            fpe[2*NBFPE + i] = fpe2[i];
            fpe[3*NBFPE + i] = fpe3[i];
        }
        MPI_Iallreduce(MPI_IN_PLACE, &fpe[0], 4*NBFPE, MPI_DOUBLE, Op5, MPI_COMM_WORLD, &request);

        // t = A w
#if PRECOND
        VvecDoubles (DONE, diags, w, DZERO, w_hat, n_dist);             // w_hat = D^-1 * w
#else
        w_hat = w;
#endif
#ifdef SPMV_OPTIMIZED
        joinDistributeVectorSPMV (COLL_P2P_SPMV, MPI_COMM_WORLD, w_hat, vecP, vdimP, vdspP, vdimR, vdspR, vectDatatypeP, vectDatatypeR);
        InitDoubles (t, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, vecP, t);            	     // t = A * w
#else
        MPI_Allgatherv (w_hat, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (t, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, t);            	    // t = A * w
#endif

        // wait for MPI_Iallreduce to complete
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        // ReproAllReduce -- End
        // split two fpes
        for (int i = 0; i < NBFPE; i++) { 
            fpe1[i] = fpe[NBFPE + i];
            fpe2[i] = fpe[2*NBFPE + i];
            fpe3[i] = fpe[3*NBFPE + i];
        }
        reduce[0] = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
        reduce[1] = exblas::cpu::Round<double, NBFPE> (&fpe1[0]);
        reduce[2] = exblas::cpu::Round<double, NBFPE> (&fpe2[0]);
        reduce[3] = exblas::cpu::Round<double, NBFPE> (&fpe3[0]);
        tmp = reduce[0];
        tol = sqrt(fabs(tmp)) / tol0;
        beta = (alpha / omega) * (tmp / rho);
        rho = tmp;

        // alpha = <r0, r+1> / (<r0, w+1> + beta <r0, s> - beta omega <r0, z>)
	// alpha = rho / (reduce[1] + beta * (reduce[2] - omega * reduce[3]));
	tmp = -omega;
	tmp = fma(tmp, reduce[3],reduce[2]);
	tmp = fma(tmp, beta, reduce[1]);
	alpha = rho / tmp;

#if DIRECT_ERROR
        // compute direct error
        for (int jj = 0; jj < n_dist; jj++)
            res_err[jj] = x_exact[jj] - x[jj];

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

    RemoveDoubles(&p_hat); RemoveDoubles (&q_hat); RemoveDoubles(&s_hat); 
    RemoveDoubles (&s); RemoveDoubles (&q); 
    RemoveDoubles (&r); RemoveDoubles (&p); RemoveDoubles (&r0); RemoveDoubles (&y);
    RemoveDoubles (&z); RemoveDoubles (&w); RemoveDoubles (&t); RemoveDoubles (&v); RemoveDoubles (&tmpv);
#if PRECOND
    RemoveDoubles (&diags); RemoveInts (&posd);
    RemoveDoubles(&r_hat); RemoveDoubles (&w_hat); RemoveDoubles (&z_hat); 
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

    double beta = 1.0 / sqrt(dim);
    if(mat_from_file) {
        // compute b = A * x_c, x_c = 1/sqrt(nbrows)
        InitDoubles (sol1, dim, 1.0, 0.0);
        ProdSparseMatrixVectorByRows (matL, 0, sol1, sol1L);            			// s = A * x
        for (int jj = 0; jj < dimL; jj++)                                           // s = beta * s
            sol1L[jj] = beta * sol1L[jj];
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
        for (int jj = 0; jj < dimL; jj++)                                           // r -= s
            sol1L[jj] = sol1L[jj] - sol2L[jj];

        std::vector<double> fpe(NBFPE);
        exblas::cpu::exdot<double*, double*, NBFPE> (dimL, sol1L, sol1L, &fpe[0]);

        // ReproAllReduce -- Begin
        // user-defined reduction operations
        MPI_Op Op;
        MPI_Op_create( (MPI_User_function *) fpeSum, 1, &Op ); 
        MPI_Allreduce(MPI_IN_PLACE, &fpe[0], NBFPE, MPI_DOUBLE, Op, MPI_COMM_WORLD);
        beta = exblas::cpu::Round<double, NBFPE> (&fpe[0]);
        MPI_Op_free( &Op );
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

