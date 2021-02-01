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
#include "utilities.h"

namespace FFTK {

    Utilities::Utilities() {}
    Utilities::Utilities(shared_ptr<Global> g): global(g) {}

    void Utilities::Zero_pad_last_plane(Array<Real,2> Ar) {
        Ar(Range::all(),Range(global->field.Nz, global->field.Nz+ 1))=0;
    }

    void Utilities::Zero_pad_last_plane(Array<Real,3> Ar) {
        Ar(Range::all(),Range::all(),Range(global->field.Nz, global->field.Nz+ 1))=0;
    }

    /** @brief After SinCos transform, shift right
     * 
     *  @param data - The real space array
     *  @param IA - The intermideate space array
     */
    void Utilities::ArrayShiftRight_basic(void *data, TinyVector<int,3> shape, int axis)
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
    void Utilities::ArrayShiftRight(Array<Real,3> Ar, int axis)
    {   
        ArrayShiftRight_basic(Ar.data(), Ar.shape(), axis);
    }



    /** @brief After SinCos transform, shift right
     * 
     *  @param Ar - The real space array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftRight(Array<Complex,3> A, int axis)
    {   
        ArrayShiftRight_basic(A.data(), A.shape()*shape(1,1,2), axis);
    }


    /** @brief After SinCos transform, shift right
     * 
     *  @param Ar - The real space array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftRight(Array<Real,2> Ar, int axis)
    {   
        ArrayShiftRight_basic(Ar.data(), shape(Ar.extent(0), 1, Ar.extent(1)), axis);
    }


    /** @brief After SinCos transform, shift right
     * 
     *  @param Ar - The real space array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftRight(Array<Complex,2> A, int axis)
    {   
        ArrayShiftRight_basic(A.data(), shape(A.extent(0), 1, A.extent(1)*2), axis);
    }


    /** @brief Before inverse SinCos transform, shift left
     * 
     *  @param data - The sequene of real numbers
     *  @param shape - shape of the array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftLeft_basic(void *data, TinyVector<int,3> shape, int axis)
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
    void Utilities::ArrayShiftLeft(Array<Real,3> Ar, int axis)
    {   
        ArrayShiftLeft_basic(Ar.data(), Ar.shape(), axis);
    }

    /** @brief Before inverse SinCos transform, shift left
     * 
     *  @param Ar - The real space array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftLeft(Array<Complex,3> A, int axis)
    {   
        ArrayShiftLeft_basic(A.data(), A.shape()*shape(1,1,2), axis);
    }

    /** @brief Before inverse SinCos transform, shift left
     * 
     *  @param Ar - The real space array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftLeft(Array<Real,2> Ar, int axis)
    {   
        ArrayShiftLeft_basic(Ar.data(), shape(Ar.extent(0), 1, Ar.extent(1)), axis);
    }

    /** @brief Before inverse SinCos transform, shift left
     * 
     *  @param Ar - The real space array
     *  @param axis - The axis along which to shift
     */
    void Utilities::ArrayShiftLeft(Array<Complex,2> A, int axis)
    {   
        ArrayShiftLeft_basic(A.data(), shape(A.extent(0), 1, A.extent(1)*2), axis);
    }

    int Utilities::Get_Rz_comm() {
        int Rz_comm;

        if (global->field.Ny > 1) {
            if (global->field.basis_name[2] == 'F')
                Rz_comm = (global->mpi.num_p_rows == 1 ? global->field.Rz : global->field.Rz-2);
            else
                Rz_comm = global->field.Rz;
        }
        else {
            if (global->field.basis_name[2] == 'F')
                Rz_comm = (global->mpi.num_p_cols == 1 ? global->field.Rz : global->field.Rz-2);
            else {
                Rz_comm = global->field.Rz;
            }
        }

        return Rz_comm;

    }

    void Utilities::Init_timers(){
        for (int i=0; i<22; i++)
            global->misc.timer[i]=0;
    }
    
};
