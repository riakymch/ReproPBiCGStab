#ifndef ToolsMPI

#define ToolsMPI 1

// #include <SparseMatricesNew.h>
#include <SparseProduct.h>

/*********************************************************************************/

#define Tag_Demand_Matrix_From_Root       1001
#define Tag_Send_Task_To_Leaf             1002
#define Tag_Receive_Dims_Factor_From_Leaf 1003
#define Tag_End_Distribution_To_Leaf      1004
#define Tag_Send_Dims_Matrix_To_Leaf      1006
#define Tag_Send_Data_Matrix_To_Leaf      1007

#define Tag_Demand_Vector_From_Root       1011
#define Tag_Send_Dims_Vector_To_Father    1015
#define Tag_Send_Data_Vector_To_Father    1016

#define Tag_Send_Task_To_Root             1021
#define Tag_Send_Solution_To_Root         1022
#define Tag_Send_Dims_Vector_To_Children  1025
#define Tag_Send_Data_Vector_To_Children  1026

#define Tag_End_Resolution_To_Leaf        1031

#define Tag_Send_Vector_Up_1              1041
#define Tag_Send_Vector_Up_2              1042

#define Tag_Send_Packet_Matrix_To_Leaf     210
#define Tag_Receive_Data_Factor_From_Leaf  220
#define Tag_Send_Vector_To_Leaf            230
// #define Tag_FactorVector           240

#define Tag_NonBlocking_SpMV               1051

/*********************************************************************************/

// typedef struct SimpleNode {
typedef struct {
	MPI_Status sta;
	MPI_Request req;
} SimpleNode, *ptr_SimpleNode;

// Return true if the corresponding asynchonous communication,
// defined by data, has been finalized
extern int TestSimple (void *data);

/*********************************************************************************/

// #define MaxPacketSize                    10000
#define MaxPacketSize                    5000

// typedef struct PacketNode {
typedef struct {
	unsigned char *ptr;
	int lblq[2*MaxPacketSize+3], vlen[MaxPacketSize];
	MPI_Aint dspl[2*MaxPacketSize+3];
	MPI_Datatype type[2*MaxPacketSize+3];
	MPI_Datatype pack;
	MPI_Status sta;
	MPI_Request req;
} PacketNode, *ptr_PacketNode;


/*********************************************************************************/

extern void Synchronization (MPI_Comm Synch_Comm, char *message);

/*********************************************************************************/

// Return true if the corresponding asynchonous communication,
// defined by data, has been finalized
extern int TestSimple (void *data);

// Return true if the corresponding asynchonous communication,
// defined by data, has been finalized
extern int TestPacket (void *data);

// Detect the lost messages whose destination is one process
// into the processes of communicator Err_Comm
extern void DetectErrorsMPI (MPI_Comm Err_Comm);

/*********************************************************************************/

// Prepare the structures required to send/receive a SparseMatrix structure
// * spr refers to the SparseMatrix from where the data is obtained
// * size is the number of rows to be sent
// * weight is the number of nonzeros to be sent
// * pcknode, where the resulted packet appears
extern void MakeSprMatrixPacket (SparseMatrix spr, int size, int weight, ptr_PacketNode pcknode);

extern void MakeSprMatrixSendPacket (SparseMatrix spr, int *len, int dimL, int dspL, 
																			ptr_PacketNode pcknode);

extern void MakeSprMatrixRecvPacket (SparseMatrix sprL, int nnzL, ptr_PacketNode pcknode);

// Compute the number of nonzeros elements of a PermSprMatrixRecvPacket packet
// * prc_src is the processor from which the messages is sent
// * sizes is the number of rows to be received
// * comm is the communicator in which the messages is sent
extern int ComputeSprMatrixRecvWeights (int prc_src, int sizes, MPI_Comm comm);

extern int DistributeMatrix (SparseMatrix spr, int index, ptr_SparseMatrix sprL, int indexL,
															int *vdimL, int *vdspL, int root, MPI_Comm comm);

/*********************************************************************************/

// vcols is a vector with dimPos elements, including integer values from 0 to dim-1
// The routine creates a bitmap determining which col index exists in vcols.
// The bitmao is stored in colsJoin whose size is colsJoin_dim
extern void joinColumns (int dim, int *vcols, int dimPos, unsigned char **colsJoin, 
										int *colsJoin_dim);

// From colsJoin, this routine creates vector perm including the contents of the bitmap
// Knowing the partition defined by nProcs and vdspL, this routine extends this
//     partition on perm, using vdimP and vdspP vectors
extern int createPerm (unsigned char *colsJoin, int colsJoin_dim, int *perm, int dim,
									int *vdspL, int *vdimP, int *vdspP, int nProcs);

// Creation of the MPI_Datatype vectors to perform a reduction of the communications
// vectDatatypeP includes MPI_DOUBLE for all processes.
// vectDatatypeR includes the permutation required for each process.
extern void createVectMPITypes (int myId, int nProcs, int *vdimL, int *vdimP, 
													int *permR, int *vdimR, int *vdspR,
													MPI_Datatype *vectDatatypeP, MPI_Datatype *vectDatatypeR);

// Creation of all structures to perform a reduction of the communications
// coll_p2p adapts the contents of the structure for collective or p2p operations
// vecP is created to store the required elements to complete the product in the process
// permP is created and included the permutation to be applied on vcols.
extern void createAlltoallwStruct (int coll_p2p, MPI_Comm comm, SparseMatrix matL, 
														int *vdimL, int *vdspL, int *vdimP, int *vdspP,
														double **vecP, int **permP, int *ipermP, 
														int *vdimR, int *vdspR,
														MPI_Datatype *vectDatatypeP, MPI_Datatype *vectDatatypeR);

// Communications to complete a MPI_Alltoallv reducing the communications
// All elements required to compute SpMV is stored on vecP from vecL
// coll_p2p marks if collective or p2p operations are used
extern void joinDistributeVectorSPMV (int coll_p2p, MPI_Comm comm, double *vecL, 
																double *vecP, int *vdimP, int *vdspP, int *vdimR, 
																int *vdspR, MPI_Datatype *vectDatatypeP, 
																MPI_Datatype *vectDatatypeR);

/*********************************************************************************/

#endif
