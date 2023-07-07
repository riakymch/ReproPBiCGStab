#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <ScalarVectors.h>
#include "ToolsMPI.h"

// #define PRINT_SEND_RESOTRNF_VECTORS 1

/*********************************************************************************/

void Synchronization (MPI_Comm Synch_Comm, char *message) {
	int my_id, i ; 

	MPI_Comm_rank(Synch_Comm, &my_id); 
	MPI_Barrier (Synch_Comm);
	printf ("(%d) %s\n", my_id, message);
	if (my_id == 0) printf ("Waiting ... \n");
	if (my_id == 0) scanf ("%d", &i);
	if (my_id == 0) printf (" ... done\n");
	MPI_Barrier (Synch_Comm);
}

/*********************************************************************************/

// Return true if the corresponding asynchonous communication,
// defined by data, has been finalized
int TestSimple (void *data) {
	int flag = 0;
	ptr_SimpleNode smpnode = (ptr_SimpleNode) data;
	
	// Verify if the communication has finalized
	MPI_Test (&(smpnode->req), &flag, &(smpnode->sta));
	if (flag) {
		// Remove the data included in the simple node
		MPI_Wait (&(smpnode->req), &(smpnode->sta));
		free (smpnode);
	}

	// Returns the result
	return flag;
}

// Return true if the corresponding asynchonous communication,
// defined by data, has been finalized
int TestPacket (void *data) {
	int flag = 0;
	ptr_PacketNode pcknode = (ptr_PacketNode) data;
	
	// Verify if the communication has finalized
	MPI_Test (&(pcknode->req), &flag, &(pcknode->sta));
	if (flag) {
		// Remove the data included in the pack
		MPI_Wait (&(pcknode->req), &(pcknode->sta));
		MPI_Type_free (&(pcknode->pack));
		free (pcknode);
	}

	// Returns the result
	return flag;
}

// Detect the lost messages whose destination is one process
// into the processes of communicator Err_Comm
void DetectErrorsMPI (MPI_Comm Err_Comm) {
	int my_id, flag= 0;
	MPI_Status st;

	// Definition of the variable my_id
	MPI_Comm_rank(Err_Comm, &my_id); 
	// Test if some message exists
	MPI_Iprobe (MPI_ANY_SOURCE, MPI_ANY_TAG, Err_Comm, &flag, &st);
	if (flag) {
		printf ("%d --> (%d,%d)\n", my_id, st.MPI_SOURCE, st.MPI_TAG);
	}
}

/*********************************************************************************/

// Prepare the structures required to send/receive a SparseMatrix structure
// * spr refers to the SparseMatrix from where the data is obtained
// * size is the number of rows to be sent
// * weight is the number of nonzeros to be sent
// * pcknode, where the resulted packet appears
void MakeSprMatrixPacket (SparseMatrix spr, int size, int weight, ptr_PacketNode pcknode) {
	int k;
	int *lblq = pcknode->lblq;
	MPI_Aint *dspl = pcknode->dspl;
	MPI_Datatype *type = pcknode->type;
		
	// Definition of reference pointer
	pcknode->ptr = (unsigned char *) spr.vptr;
	// Definition of the required vectors to create the packet
	type[0] = MPI_INT   ; lblq[0] = size+1; dspl[0] = (MPI_Aint) spr.vptr;
	type[1] = MPI_INT   ; lblq[1] = weight; dspl[1] = (MPI_Aint) spr.vpos;
	type[2] = MPI_DOUBLE; lblq[2] = weight; dspl[2] = (MPI_Aint) spr.vval;
	type[3] = MPI_UB    ; lblq[3] = 1     ; dspl[3] = (MPI_Aint) (spr.vptr+size+1);
	for (k=3; k>=0; k--) dspl[k] -= dspl[0]; 
	// Creation of the packet
	MPI_Type_create_struct (4, lblq, dspl, type, &(pcknode->pack));
	MPI_Type_commit(&(pcknode->pack));
}

void MakeSprMatrixSendPacket (SparseMatrix spr, int *vlen, int dimL, int dspL, 
															ptr_PacketNode pcknode) {
	int k, weight, dspZ;
	int *lblq = pcknode->lblq;
	MPI_Aint *dspl = pcknode->dspl;
	MPI_Datatype *type = pcknode->type;
		
//	printf ("dimL = %d , dspL = %d\n", dimL, dspL);
//	PrintInts (vlen, spr.dim1);
//	PrintInts (spr.vptr, spr.dim1+1);
	// Definition of reference pointer
	pcknode->ptr = (unsigned char *) (vlen+dspL);
	// Definition of the required vectors to create the packet
	dspZ = spr.vptr[dspL]; weight = spr.vptr[dspL+dimL] - dspZ;
//	printf ("dspZ = %d , weight = %d\n", dspZ, weight);
	type[0] = MPI_INT   ; lblq[0] = dimL  ; dspl[0] = (MPI_Aint) (vlen+dspL     );
	type[1] = MPI_INT   ; lblq[1] = weight; dspl[1] = (MPI_Aint) (spr.vpos+dspZ );
	type[2] = MPI_DOUBLE; lblq[2] = weight; dspl[2] = (MPI_Aint) (spr.vval+dspZ );
	type[3] = MPI_UB    ; lblq[3] = 1     ; dspl[3] = (MPI_Aint) (vlen+dimL+dspL);
	for (k=3; k>=0; k--) dspl[k] -= dspl[0]; 
	// Creation of the packet
	MPI_Type_create_struct (4, lblq, dspl, type, &(pcknode->pack));
	MPI_Type_commit(&(pcknode->pack));
}

void MakeSprMatrixRecvPacket (SparseMatrix sprL, int nnzL, ptr_PacketNode pcknode) {
	int k, dimL = sprL.dim1;
	int *lblq = pcknode->lblq;
	MPI_Aint *dspl = pcknode->dspl;
	MPI_Datatype *type = pcknode->type;
		
//	printf ("nnzL = %d\n", nnzL);
	// Definition of reference pointer
	pcknode->ptr = (unsigned char *) (sprL.vptr+1);
	// Definition of the required vectors to create the packet
	type[0] = MPI_INT   ; lblq[0] = dimL; dspl[0] = (MPI_Aint) (sprL.vptr+1);
	type[1] = MPI_INT   ; lblq[1] = nnzL; dspl[1] = (MPI_Aint) sprL.vpos;
	type[2] = MPI_DOUBLE; lblq[2] = nnzL; dspl[2] = (MPI_Aint) sprL.vval;
	type[3] = MPI_UB    ; lblq[3] = 1   ; dspl[3] = (MPI_Aint) (sprL.vptr+1+dimL);
	for (k=3; k>=0; k--) dspl[k] -= dspl[0]; 
	// Creation of the packet
	MPI_Type_create_struct (4, lblq, dspl, type, &(pcknode->pack));
	MPI_Type_commit(&(pcknode->pack));
}

// Compute the number of nonzeros elements of a PermSprMatrixRecvPacket packet
// * prc_src is the processor from which the messages is sent
// * dimL is the number of rows to be received
// * comm is the communicator in which the messages is sent
int ComputeSprMatrixRecvWeights (int prc_src, int dimL, MPI_Comm comm) {
	int tam, tam_int, tam_double, tam_ub;
	MPI_Status sta;

	// Definition of sizes
	MPI_Type_size(MPI_INT, &tam_int);
	MPI_Type_size(MPI_DOUBLE, &tam_double);
	MPI_Type_size(MPI_UB, &tam_ub);
	MPI_Probe (prc_src, Tag_Send_Packet_Matrix_To_Leaf, comm, &sta);
	MPI_Get_count (&sta, MPI_BYTE, &tam);

	// Return the number of nonzeros included in a packet
	return (tam - (dimL*tam_int + tam_ub)) / (tam_int + tam_double);
}


int DistributeMatrix (SparseMatrix spr, int index, ptr_SparseMatrix sprL, int indexL,
												int *vdimL, int *vdspL, int root, MPI_Comm comm) {
	int myId, nProcs;
	int i, dim = spr.dim1, divL, rstL, dimL, dspL, nnzL;
	ptr_PacketNode pcknode;

	// Getiing the parameter of the communicator
	MPI_Comm_rank(comm, &myId); MPI_Comm_size(comm, &nProcs); 
	// Broadcasting the matrix dimension
	MPI_Bcast (&dim, 1, MPI_INT, root, MPI_COMM_WORLD); 

	// Calculating the vectors of sizes (vdimL) and displacements (vdspl)
	divL = (dim / nProcs); rstL = (dim % nProcs);
	for (i=0; i<nProcs; i++) vdimL[i] = divL + (i < rstL);
	vdspL[0] = 0; for (i=0; i<nProcs; i++) vdspL[i+1] = vdspL[i] + vdimL[i];
	dimL = vdimL[myId];	dspL = vdspL[myId];	
	
	// Distribution of the matrix, by blocks
	if (root == myId) {
		int *vlen = NULL;

		CreateInts (&vlen, dim); ComputeLengthfromHeader (spr.vptr, vlen, dim);
		for (i=0; i<nProcs; i++) {
			if (i != myId) {
				// Creating the message for each destination
				pcknode = (ptr_PacketNode) malloc (sizeof(PacketNode));
				MakeSprMatrixSendPacket (spr, vlen, vdimL[i], vdspL[i], pcknode);
				MPI_Send (pcknode->ptr, 1, pcknode->pack, i, Tag_Send_Packet_Matrix_To_Leaf, comm);
				MPI_Type_free (&(pcknode->pack));
				free (pcknode);
			}
		}
		nnzL = spr.vptr[dspL+dimL] - spr.vptr[dspL]; 
		CreateSparseMatrix (sprL, indexL, dimL, dim, nnzL, 0);
		CopyInts    (vlen+dspL, sprL->vptr+1, dimL);
		CopyInts    (spr.vpos+spr.vptr[dspL], sprL->vpos, nnzL);
		CopyDoubles (spr.vval+spr.vptr[dspL], sprL->vval, nnzL);

		RemoveInts (&vlen);
	} else {
		MPI_Status sta;

		// Compute the number of nonzeroes and creating the local matrix
		nnzL = ComputeSprMatrixRecvWeights (root, dimL, comm);
		CreateSparseMatrix (sprL, indexL, dimL, dim, nnzL, 0);
		// Receiving the data on the local matrix
		pcknode = (ptr_PacketNode) malloc (sizeof(PacketNode));
		MakeSprMatrixRecvPacket (*sprL, nnzL, pcknode);
		MPI_Recv (pcknode->ptr, 1, pcknode->pack, root, Tag_Send_Packet_Matrix_To_Leaf,
               comm, &sta);
		MPI_Type_free (&(pcknode->pack));
		free (pcknode);
	}
	*(sprL->vptr) = indexL; TransformLengthtoHeader (sprL->vptr, dimL);

	return dim;
}

/*********************************************************************************/

// vcols is a vector with dimPos elements, including integer values from 0 to dim-1
// The routine creates a bitmap determining which col index exists in vcols.
// The bitmao is stored in colsJoin whose size is colsJoin_dim
void joinColumns (int dim, int *vcols, int dimPos, unsigned char **colsJoin, 
										int *colsJoin_dim) {
	int i, div, rem;
	int vec_dim = (dim + (sizeof(unsigned char) * 8) - 1) / (sizeof(unsigned char) * 8);
	unsigned char *vec = (unsigned char *) malloc(sizeof(unsigned char) * vec_dim);
	
	for (i=0; i<vec_dim; i++) {
		vec[i] = 0x0;
	}
	
	for (i=0; i<dimPos; i++) {
		div = vcols[i] / (sizeof(unsigned char) * 8);
		rem = vcols[i] % (sizeof(unsigned char) * 8);
		vec[div] |= (1 << rem);
	}
	*colsJoin = vec;
	*colsJoin_dim = vec_dim;
}


// From colsJoin, this routine creates vector perm including the contents of the bitmap
// Knowing the partition defined by nProcs and vdspL, this routine extends this
//     partition on perm, using vdimP and vdspP vectors
int createPerm (unsigned char *colsJoin, int colsJoin_dim, int *perm, int dim,
									int *vdspL, int *vdimP, int *vdspP, int nProcs) {
	int i, j, prc = 1, k = 0, col = 0;

	vdspP[0] = 0;
	for (i=0; i<colsJoin_dim; i++) {
		if (colsJoin[i] != 0x0) {
			unsigned char car = 0x1;
			for (j=0; j<(int)8*sizeof(unsigned char); j++) {
				if (col == vdspL[prc]) {
					vdimP[prc-1] = k - vdspP[prc-1];
					vdspP[prc] = k;
					prc++;
				}
				if (colsJoin[i] & car) {
					perm[k] = col;
					k++; 
				}
				car <<= 1;
				col++;
			}
		} else {
			col += 8*sizeof(unsigned char);
			while ((prc <= nProcs) && (col >= vdspL[prc])) {
					vdimP[prc-1] = k - vdspP[prc-1];
					vdspP[prc] = k;
					prc++;
			}
		}
	}
	while ((prc <= nProcs) && (col >= vdspL[prc])) {
		vdimP[prc-1] = k - vdspP[prc-1];
		vdspP[prc] = k;
		prc++;
  }
	return k;
}


// Creation of the MPI_Datatype vectors to perform a reduction of the communications
// vectDatatypeP includes MPI_DOUBLE for all processes.
// vectDatatypeR includes the permutation required for each process.
void createVectMPITypes (int myId, int nProcs, int *vdimL, int *vdimP, 
													int *permR, int *vdimR, int *vdspR,
													MPI_Datatype *vectDatatypeP, MPI_Datatype *vectDatatypeR) {
	int i;

	for (i=0; i<nProcs; i++) {
		vectDatatypeP[i] = MPI_DOUBLE;
		if (i == myId) {
			MPI_Type_contiguous(vdimR[i], MPI_DOUBLE, vectDatatypeR+i);
		} else {
			MPI_Type_create_indexed_block(vdimR[i], 1, permR+vdspR[i], MPI_DOUBLE, 
																		vectDatatypeR+i);
		}
		MPI_Type_commit(vectDatatypeR+i);
	}
}


// Creation of all structures to perform a reduction of the communications
// coll_p2p adapts the contents of the structure for collective or p2p operations
// vecP is created to store the required elements to complete the product in the process
// permP is created and included the permutation to be applied on vcols.
void createAlltoallwStruct (int coll_p2p, MPI_Comm comm, SparseMatrix matL, 
														int *vdimL, int *vdspL, int *vdimP, int *vdspP,
														double **vecP, int **permP, int *ipermP, 
														int *vdimR, int *vdspR,
														MPI_Datatype *vectDatatypeP, MPI_Datatype *vectDatatypeR) {
	
	// Definition of the variables nProcs and myId
	int myId, nProcs;
  MPI_Comm_size(comm, &nProcs); MPI_Comm_rank(comm, &myId);

	// Creation of column bitmap related to myId.
	int colsJoin_dim = 0, dim = matL.dim2;
	unsigned char *colsJoin = NULL;
	joinColumns (dim, matL.vpos, matL.vptr[matL.dim1], &colsJoin, &colsJoin_dim);

	// Creation of permutations, getting informaction from column bitmap
	int permP_dim = 0;
	permP_dim = createPerm (colsJoin, colsJoin_dim, ipermP, dim, vdspL, 
														vdimP, vdspP, nProcs);
	free (colsJoin); colsJoin = NULL;
	CreateDoubles (vecP, permP_dim);
	CreateInts (permP, permP_dim); CopyInts (ipermP, *permP, permP_dim); 
	InitInts (ipermP, dim, -1, 0);
	ComputeInvPermutation (*permP, ipermP, permP_dim);

	// Definition of sizes of the sending pattern from myId
	MPI_Alltoall (vdimP, 1, MPI_INT, vdimR, 1, MPI_INT, MPI_COMM_WORLD);
	vdspR[0] = 0; ComputeHeaderfromLength (vdimR, vdspR, nProcs);

	// Creation of the sending pattern from myId
	int *permR = NULL, permR_dim = 0;;
	permR_dim = vdspR[nProcs];
	CreateInts (&permR, permR_dim);
	MPI_Alltoallv (*permP, vdimP, vdspP, MPI_INT, 
								 permR , vdimR, vdspR, MPI_INT, MPI_COMM_WORLD);
	CopyShiftInts (permR, permR, permR_dim, -vdspL[myId]);

	// Definition of the MPI_Datatype vectors required for communication
	createVectMPITypes (myId, nProcs, vdimL, vdimP, permR, vdimR, vdspR, 
												vectDatatypeP, vectDatatypeR);

	// Computation of percentage of communication is done
  int saving = vdspR[nProcs] - vdimL[myId];
	MPI_Allreduce (MPI_IN_PLACE, &saving, 1, MPI_INT, MPI_SUM, comm);
	if (myId == 0) {
		printf ("%d nnzs of %d = %f %% \n", saving, dim * (nProcs - 1), 
									100.0 * saving / (dim * (nProcs - 1)));
	}

	// Adaptation of sizes to complete the comunication
	InitInts (vdimR, nProcs, 1, 0);
	InitInts (vdspR, nProcs+1, 0, 0);
	// This step is only required for MPI_Alltoallw
	if (coll_p2p) {	
		ScaleInts (vdspP, sizeof(double), nProcs+1);
	}

	// Freeing unuseful structures
	RemoveInts (&permR);
}


// Communications to complete a MPI_Alltoallv reducing the communications
// All elements required to compute SpMV is stored on vecP from vecL
// coll_p2p marks if collective or p2p operations are used
void joinDistributeVectorSPMV (int coll_p2p, MPI_Comm comm, double *vecL, 
																double *vecP, int *vdimP, int *vdspP, int *vdimR, 
																int *vdspR, MPI_Datatype *vectDatatypeP, 
																MPI_Datatype *vectDatatypeR) {
	if (coll_p2p) {
		// Comunication using a collective operation
  	MPI_Alltoallw (vecL, vdimR, vdspR, vectDatatypeR,
                   vecP, vdimP, vdspP, vectDatatypeP, MPI_COMM_WORLD);
	} else {
  	// Definition of the variables nProcs, myId and other variables
  	int i, k = 0, myId, nProcs;
  	MPI_Comm_size (comm, &nProcs); MPI_Comm_rank(comm, &myId);

  	// Definition of the vectors for implementing non-blocking communications
		MPI_Status vectSta[2*nProcs-2];
		MPI_Request vectReq[2*nProcs-2];
		
  	// Non-blocking send communications
		for (i=0; i<nProcs; i++) {
			if (i != myId) {
				MPI_Isend (vecL+vdspR[i], vdimR[i], vectDatatypeR[i], i, Tag_NonBlocking_SpMV,
              			comm, vectReq+k); 
				k++;
			}
		}
  	// Non-blocking receive communications
		for (i=0; i<nProcs; i++) {
			if (i != myId) {
				MPI_Irecv (vecP+vdspP[i], vdimP[i], vectDatatypeP[i], i, Tag_NonBlocking_SpMV,
              			comm, vectReq+k); 
				k++;
			}
		}
  	// Local copy
		memcpy (vecP+vdspP[myId], vecL, vdimP[myId] * sizeof(double));

  	// Waiting until all communications are complete
		MPI_Waitall (2*nProcs-2, vectReq, vectSta);
	}
}

/*********************************************************************************/
