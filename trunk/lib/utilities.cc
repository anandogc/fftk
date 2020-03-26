/* FFTK
 *
 * Copyright (C) 2020, Mahendra K. Verma, Anando Gopal Chatterjee
 *
 * Mahendra K. Verma
 * Indian Institute of Technology, Kanpur-208016
 * UP, India
 *
 * mkv@iitk.ac.in
 *
 * This file is part of FFTK.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * \file  utilities.cc
 * @author  A. G. Chatterjee, M. K. Verma
 * @date Jan 2017
 * @bug  No known bugs
 */

#include "fftk.h"

void FFTK::Zero_pad_last_plane(Array<Real,2> Ar) {
    Ar(Range::all(),Range(Nz,Nz+1))=0;
}

void FFTK::Zero_pad_last_plane(Array<Real,3> Ar) {
    Ar(Range::all(),Range::all(),Range(Nz,Nz+1))=0;
}

void  FFTK::Normalize(Array<Complex,2> A) {
    A*=normalizing_factor;
}

void  FFTK::Normalize(Array<Complex,3> A) {
    A*=normalizing_factor;
}

/** @brief After SinCos transform, shift right
 * 
 *  @param data - The real space array
 *  @param IA - The intermideate space array
 */
void FFTK::ArrayShiftRight_basic(void *data, TinyVector<int,3> shape, int axis)
{   
    Array<Real,3> Ar((Real*)(data), shape, neverDeleteData);

    if (axis == 'X') {
        Ar(Range(Ar.extent(0)-1,1,-1),Range::all(),Range::all()) = Ar(Range(Ar.extent(0)-2,0,-1),Range::all(),Range::all());
        Ar(0,Range::all(),Range::all()) = 0.0;
    }
    else if (axis == 'Y') {
        Ar(Range::all(),Range(Ar.extent(1)-1,1,-1),Range::all()) = Ar(Range::all(),Range(Ar.extent(1)-2,0,-1),Range::all());
        Ar(Range::all(),0,Range::all()) = 0.0;  
    }
    else if (axis == 'Z') {
        Ar(Range::all(),Range::all(),Range(Ar.extent(2)-1,1,-1)) = Ar(Range::all(),Range::all(),Range(Ar.extent(2)-2,0,-1));
        Ar(Range::all(),Range::all(),0) = 0.0;  
    }
}

/** @brief After SinCos transform, shift right
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftRight(Array<Real,3> Ar, int axis)
{   
    ArrayShiftRight_basic(Ar.data(), Ar.shape(), axis);
}



/** @brief After SinCos transform, shift right
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftRight(Array<Complex,3> A, int axis)
{   
    ArrayShiftRight_basic(A.data(), A.shape()*shape(1,1,2), axis);
}


/** @brief After SinCos transform, shift right
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftRight(Array<Real,2> Ar, int axis)
{   
    ArrayShiftRight_basic(Ar.data(), shape(Ar.extent(0), 1, Ar.extent(1)), axis);
}


/** @brief After SinCos transform, shift right
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftRight(Array<Complex,2> A, int axis)
{   
    ArrayShiftRight_basic(A.data(), shape(A.extent(0), 1, A.extent(1)*2), axis);
}


/** @brief Before inverse SinCos transform, shift left
 * 
 *  @param data - The sequene of real numbers
 *  @param shape - shape of the array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftLeft_basic(void *data, TinyVector<int,3> shape, int axis)
{
    Array<Real,3> Ar((Real*)(data), shape, neverDeleteData);

    if (axis == 'X') {
        Ar(Range(0,Ar.extent(0)-2),Range::all(),Range::all()) = Ar(Range(1,Ar.extent(0)-1),Range::all(),Range::all());
        Ar(Ar.extent(0)-1,Range::all(),Range::all()) = 0.0; 
    }
    else if (axis == 'Y') {
        Ar(Range::all(),Range(0,Ar.extent(1)-2),Range::all()) = Ar(Range::all(),Range(1,Ar.extent(1)-1),Range::all());
        Ar(Range::all(),Ar.extent(1)-1,Range::all()) = 0.0;

    }
    else if (axis == 'Z') {
        Ar(Range::all(),Range::all(),Range(0,Ar.extent(2)-2)) = Ar(Range::all(),Range::all(),Range(1,Ar.extent(2)-1));
        Ar(Range::all(),Range::all(),Ar.extent(2)-1) = 0.0; 
    }
}

/** @brief Before inverse SinCos transform, shift left
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftLeft(Array<Real,3> Ar, int axis)
{   
    ArrayShiftLeft_basic(Ar.data(), Ar.shape(), axis);
}

/** @brief Before inverse SinCos transform, shift left
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftLeft(Array<Complex,3> A, int axis)
{   
    ArrayShiftLeft_basic(A.data(), A.shape()*shape(1,1,2), axis);
}

/** @brief Before inverse SinCos transform, shift left
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftLeft(Array<Real,2> Ar, int axis)
{   
    ArrayShiftLeft_basic(Ar.data(), shape(Ar.extent(0), 1, Ar.extent(1)), axis);
}

/** @brief Before inverse SinCos transform, shift left
 * 
 *  @param Ar - The real space array
 *  @param axis - The axis along which to shift
 */
void FFTK::ArrayShiftLeft(Array<Complex,2> A, int axis)
{   
    ArrayShiftLeft_basic(A.data(), shape(A.extent(0), 1, A.extent(1)*2), axis);
}


/** @brief Convert pencil into slab
 * 
 *  @param ArPencil - The real space pencil divided array
 *  @param ArSlab - The real space slab divided array
 */
void FFTK::To_slab(Array<Real,3> ArPencil, Array<Real,3> ArSlab)
{
    /*MPI_Gather(ArPencil.data(), ArPencil.size(), MPI_Real,
           ArSlab.data(), 1, MPI_Vector_resized_ZY_planes, 0,
           MPI_COMM_COL);
    */

    Array<Real,3> ArSlab_temp(num_p_rows*maxrx,Ry/num_p_rows,Rz);

    MPI_Gather(ArPencil.data(), ArPencil.size(), MPI_Real,
           ArSlab_temp.data(), ArPencil.size(), MPI_Real, 0,
           MPI_COMM_COL);


    if (my_row_id==0) {
        for (int p=0; p<num_p_rows; p++) {
            ArSlab(Range::all(), Range(p*Ry/num_p_rows,(p+1)*Ry/num_p_rows-1), Range::all())
                = ArSlab_temp(Range(p*maxrx,(p+1)*maxrx-1),Range::all(),Range::all());
        }
    }
}

/** @brief Convert slab into pencil
 * 
 *  @param ArSlab - The real space slab divided array
 *  @param ArPencil - The real space pencil divided array
 */
void FFTK::To_pencil(Array<Real,3> ArSlab, Array<Real,3> ArPencil)
{
    Array<Real,3> ArSlab_temp(num_p_rows*maxrx,Ry/num_p_rows,Rz);

    if (my_row_id==0) {
        for (int p=0; p<num_p_rows; p++) {
            ArSlab_temp(Range(p*maxrx,(p+1)*maxrx-1),Range::all(),Range::all())
                = ArSlab(Range::all(), Range(p*Ry/num_p_rows,(p+1)*Ry/num_p_rows-1), Range::all());
        }
    }

    MPI_Scatter(ArSlab_temp.data(), ArPencil.size(), MPI_Real,
        ArPencil.data(), ArPencil.size(), MPI_Real, 0,
        MPI_COMM_COL);
}
MPI_Comm FFTK::Get_communicator(string which) {
    if (which=="ROW")
        return MPI_COMM_ROW;
    else if (which=="COL")
        return MPI_COMM_COL;
    else {
        if (my_id==0)
            cerr << "Invalid communicator: " << which << endl;
        MPI_Finalize();
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
}


TinyVector<int,3> FFTK::Get_FA_shape()
{
    return FA_shape;
}
TinyVector<int,3> FFTK::Get_IA_shape()
{
    return IA_shape;
}
TinyVector<int,3> FFTK::Get_RA_shape()
{
    return RA_shape;
}

TinyVector<int,3> FFTK::Get_FA_start()
{
    return TinyVector<int, 3>(fx_start,fy_start,fz_start);
}
TinyVector<int,3> FFTK::Get_IA_start()
{
    return TinyVector<int, 3>(ix_start,iy_start,iz_start);
}
TinyVector<int,3> FFTK::Get_RA_start()
{
    return TinyVector<int, 3>(rx_start,ry_start,rz_start);
}

int FFTK::Get_row_id()
{
    return my_row_id;
}
int FFTK::Get_col_id()
{
    return my_col_id;
}

int FFTK::Get_num_p_rows()
{
    return num_p_rows;
}
int FFTK::Get_num_p_cols()
{
    return num_p_cols;
}
