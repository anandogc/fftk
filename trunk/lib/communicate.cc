#include "fftk.h"


/** @file fftk.cc
 * 
 * @sa fftk.h
 * 
 * @author  A. G. Chatterjee
 * @version 2.0
 * @date    01/03/2018
 * @bug     No known bugs
 */ 


/** @brief Transpose a 2d Fourier space array into 2d real array
 *  @param FA - The Fourier space array
 *  @param RA - The real space array
 *
 */
void FFTK::Transpose(Array<Complex,2> FA, Array<Real,2> RA)
{
    // cout << "num_p_cols = " << num_p_cols << endl;
    if (num_p_cols>1) {
        Real *__restrict__ RA_data=reinterpret_cast<Real*>(RA.data());
        Real *__restrict__ FA_data=reinterpret_cast<Real*>(FA.data());

        RA = 0;

        MPI_Alltoall(FA_data, maxfz*Fz_unit*maxrx, MPI_Real, buffer_Z, maxfz*Fz_unit*maxrx, MPI_Real, MPI_COMM_ROW);

        for (size_t c=0; c<num_p_cols; c++)
            for (size_t rx=0; rx<maxrx; rx++)
                std::copy(buffer_Z + c*maxrx*maxfz*Fz_unit + rx*maxfz*Fz_unit, buffer_Z + c*maxrx*maxfz*Fz_unit + (rx+1)*maxfz*Fz_unit, RA_data + rx*Rz + c*maxfz*Fz_unit);

    }
    else
        RA=Array<Real,2>(reinterpret_cast<Real*>(FA.data()), RA.shape(), neverDeleteData);
}

/** @brief Transpose a 2d Fourier space array into 2d real space array
 *  @param RA - The real space array
 *  @param FA - The Fourier space array
 *
 */
void FFTK::Transpose(Array<Real,2> RA, Array<Complex,2> FA)
{
    if (num_p_cols>1) {
        Real *__restrict__ RA_data=reinterpret_cast<Real*>(RA.data());
        Real *__restrict__ FA_data=reinterpret_cast<Real*>(FA.data());

        for (size_t c=0; c<num_p_cols; c++)
            for (size_t rx=0; rx<maxrx; rx++)
                for (int i=0; i<maxfz*Fz_unit; i++)
                    *(buffer_Z + c*maxrx*maxfz*Fz_unit + rx*maxfz*Fz_unit + i) = *(RA_data + rx*Rz + c*maxfz*Fz_unit+i);
        //      std::copy(RA_data + rx*(Nz+2) + c*maxfz*Fz_unit, RA_data + rx*(Nz+2) + (c+1)*maxfx*Fz_unit, RA_data + c*maxrx*maxfz*Fz_unit + rx*maxfz*Fz_unit);

        MPI_Alltoall(buffer_Z, maxfz*Fz_unit*maxrx, MPI_Real, FA_data, maxfz*Fz_unit*maxrx, MPI_Real, MPI_COMM_ROW);
    }
    else
        FA=Array<Complex,2>(reinterpret_cast<Complex*>(RA.data()), FA.shape(), neverDeleteData);
}

/** @brief Transpose the processor division of a 3d real space array into 3d Fourier space array
 *  @param RA - The real space array
 *  @param FA - The Fourier space array
 *
 */
void FFTK::Transpose(Array<Real,3> Ar, Array<Complex,3> A)
{
    Transpose_ZY(Ar,IA);
    Transpose_YX(IA,A);
}

/** @brief Transpose the processor division of a 3d Fourier space array into 3d real space array
 *  @param RA - The real space array
 *  @param FA - The Fourier space array
 *
 */
void FFTK::Transpose(Array<Complex,3> A, Array<Real,3> Ar)
{
    Transpose_XY(A,IA);
    Transpose_YZ(IA,Ar);
}

/** @brief Transpose the processor division of a 3d Fourier space array into 3d real space array
 *  @param RA - The real space array
 *  @param FA - The Fourier space array
 */
void FFTK::Transpose_XY(Array<Complex,3> FA, Array<Complex,3> IA)
{

    if (num_p_cols>1) {
        Real *__restrict__ FA_data=reinterpret_cast<Real*>(FA.data());
        Real *__restrict__ IA_data=reinterpret_cast<Real*>(IA.data());

        timer[11] -= MPI_Wtime();
        MPI_Alltoall(FA_data, block_size_YX, MPI_Real, buffer_Y, block_size_YX, MPI_Real, MPI_COMM_ROW);
        timer[11] += MPI_Wtime();

        timer[12] -= MPI_Wtime();
        for (int c=0; c<num_p_cols; c++)
            for (int ix=0; ix<maxix; ix++)
                std::copy(buffer_Y + c*block_size_YX + ix*copy_size_YX, buffer_Y + c*block_size_YX + (ix+1)*copy_size_YX, IA_data + ix*plane_size_YX + c*copy_size_YX);
        timer[12] += MPI_Wtime();
    }
    else
        IA=Array<Complex,3>(reinterpret_cast<Complex*>(FA.data()), IA_shape, neverDeleteData);

}

/** @brief Transpose the processor division of pencils along Y to pencils along X
 *  @param IA - The intermideate space array
 *  @param FA - The Fourier space array
 */
void FFTK::Transpose_YX(Array<Complex,3> IA, Array<Complex,3> FA)
{
    if (num_p_cols>1) {
        Real *__restrict__ IA_data=reinterpret_cast<Real*>(IA.data());
        Real *__restrict__ FA_data=reinterpret_cast<Real*>(FA.data());


        timer[13] -= MPI_Wtime();
        for (size_t c=0; c<num_p_cols; c++)
            for (size_t ix=0; ix<maxix; ix++)
                std::copy(IA_data + ix*plane_size_YX + c*copy_size_YX, IA_data + ix*plane_size_YX + (c+1)*copy_size_YX, buffer_Y + c*block_size_YX + ix*copy_size_YX);
        timer[13] += MPI_Wtime();

        timer[14] -= MPI_Wtime();
        MPI_Alltoall(buffer_Y, block_size_YX, MPI_Real, FA_data, block_size_YX, MPI_Real, MPI_COMM_ROW);
        timer[14] += MPI_Wtime();
    }
    else
        FA=Array<Complex,3>(reinterpret_cast<Complex*>(IA.data()), FA_shape, neverDeleteData);

}

/** @brief Transpose the processor division of pencils along Y to pencils along Z
 * 
 *  @param IA - The intermideate space array
 *  @param RA - The real space array
 */
void FFTK::Transpose_YZ(Array<Complex,3> IA, Array<Real,3> RA)
{
    if (num_p_rows>1) {
        Real *__restrict__ IA_data=reinterpret_cast<Real*>(IA.data());
        Real *__restrict__ Ar_data=reinterpret_cast<Real*>(RA.data());

        timer[15] -= MPI_Wtime();
        for (size_t r=0; r<num_p_rows; r++)
            for (size_t ix=0; ix<maxix; ix++)
                std::copy(IA_data + ix*plane_size_YZ + r*copy_size_YZ, IA_data + ix*plane_size_YZ + (r+1)*copy_size_YZ, buffer_Y + r*block_size_YZ + ix*copy_size_YZ);
        timer[15] += MPI_Wtime();


        timer[16] -= MPI_Wtime();
        MPI_Alltoall(buffer_Y, block_size_YZ, MPI_Real, buffer_Z, block_size_YZ, MPI_Real, MPI_COMM_COL);
        timer[16] += MPI_Wtime();


        timer[17] -= MPI_Wtime();
        for (size_t r=0; r<num_p_rows; r++)
            for (size_t j=0; j<maxry*maxrx; j++)
                std::copy(buffer_Z + r*block_size_YZ + j*Iz_unit*maxiz,buffer_Z + r*block_size_YZ + (j+1)*Iz_unit*maxiz, Ar_data + j*Rz + r*Iz_unit*maxiz);
        timer[17] += MPI_Wtime();
    }
    else 
        RA=Array<Real,3>(reinterpret_cast<Real*>(IA.data()), RA_shape, neverDeleteData);


    if (Iz_unit*maxiz*num_p_rows < Rz) {
        timer[18] -= MPI_Wtime();
        Zero_pad_last_plane(RA);
        timer[18] += MPI_Wtime();
    }

}

/** @brief Transpose the processor division of pencils along Y to pencils along Z
 *
 *  @param RA - The real space array
 *  @param IA - The intermideate space array
 */
void FFTK::Transpose_ZY(Array<Real,3> RA, Array<Complex,3> IA)
{
    if (num_p_rows>1) {
        Real *__restrict__ RA_data=reinterpret_cast<Real*>(RA.data());
        Real *__restrict__ IA_data=reinterpret_cast<Real*>(IA.data());

        timer[19] -= MPI_Wtime();
        for (size_t r=0; r<num_p_rows; r++)
            for (size_t j=0; j<maxry*maxrx; j++)
                std::copy(RA_data + j*Rz + r*Iz_unit*maxiz, RA_data + j*Rz + (r+1)*Iz_unit*maxiz, buffer_Z + r*block_size_YZ + j*Iz_unit*maxiz);
        timer[19] += MPI_Wtime();

        timer[20] -= MPI_Wtime();
        MPI_Alltoall(buffer_Z, block_size_YZ, MPI_Real, buffer_Y, block_size_YZ, MPI_Real, MPI_COMM_COL);
        timer[20] += MPI_Wtime();

        timer[21] -= MPI_Wtime();
        for (size_t r=0; r<num_p_rows; r++)
            for (size_t j=0; j<maxix; j++)
                std::copy(buffer_Y + r*block_size_YZ + j*copy_size_YZ, buffer_Y + r*block_size_YZ + (j+1)*copy_size_YZ, IA_data + j*plane_size_YZ + r*copy_size_YZ);
        timer[21] += MPI_Wtime();
    }
    else
        IA=Array<Complex,3>(reinterpret_cast<Complex*>(RA.data()), FA_shape, neverDeleteData);

}
