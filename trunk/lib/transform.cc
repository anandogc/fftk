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
 * \file  transform.cc
 * @author  A. G. Chatterjee, M. K. Verma
 * @date Jan 2017
 * @bug  No known bugs
 */

#include "fftk.h"

/** @brief Perform the Forward transform
 *
 *  @param basis_option - basis_option can be "FFF", "SFF", "SSF", "SSS"
 *                      - F stands for Fourier
 *                      - S stands for Sine/Cosine
 *  @param Ar - The real space array
 *  @param A  - The Fourier space array
 *
 */
void FFTK::Forward_transform(string basis_option, Array<Real,2> Ar, Array<Complex,2> A)
{
    timer[0]-=MPI_Wtime();
    if ( (basis_option[2] == 'F') && (fft_plan_Fourier_forward_z != NULL) ) {
        FFTW_EXECUTE_DFT_R2C(fft_plan_Fourier_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<FFTW_Complex*>(Ar.data()));
    }
    else if ( (basis_option[2] == 'S') && (fft_plan_Sine_forward_z != NULL) ) {
        FFTW_EXECUTE_R2R(fft_plan_Sine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
        ArrayShiftRight(Ar, 'Z');
    }
    else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_forward_z != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "' or plan z not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    timer[0]+=MPI_Wtime();

    timer[1]-=MPI_Wtime();
    Transpose(Ar, A);
    timer[1]+=MPI_Wtime();


    timer[4]-=MPI_Wtime();
    if ( (basis_option[0] == 'F') && (fft_plan_Fourier_forward_x != NULL) )
        FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    else if ( (basis_option[0] == 'S') && (fft_plan_Sine_forward_x != NULL) ) {
        FFTW_EXECUTE_R2R(fft_plan_Sine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
        ArrayShiftRight(A, 'X');
    }
    else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_forward_x != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "' or plan x not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[4]+=MPI_Wtime();

    timer[5]-=MPI_Wtime();
    Normalize(A);
    timer[5]+=MPI_Wtime();
}


/** @brief Perform the Inverse transform
 *
 *  @param basis_option - basis_option can be "FFF", "SFF", "SSF", "SSS"
 *                      - F stands for Fourier
 *                      - S stands for Sine/Cosine
 *  @param A  - The Fourier space array
 *  @param Ar - The real space array
 *
 */
void FFTK::Inverse_transform(string basis_option, Array<Complex,2> A, Array<Real,2> Ar)
{
    timer[6]-=MPI_Wtime();
    if ( (basis_option[0] == 'F') && (fft_plan_Fourier_inverse_x != NULL) )
        FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));

    else if ( (basis_option[0] == 'S') && (fft_plan_Sine_inverse_x != NULL) ) {
        ArrayShiftLeft(A, 'X');
        FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
    }
    else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_inverse_x != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
    else {
        if (my_id==0) cerr << "FFTK::inverse_transform - Invalid basis_option '" << basis_option << "' or inverse plan x not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[6]+=MPI_Wtime();


    timer[7]-=MPI_Wtime();
    Transpose(A, Ar);
    timer[7]+=MPI_Wtime();


    timer[10]-=MPI_Wtime();
    if ( (basis_option[2] == 'F') && (fft_plan_Fourier_inverse_z != NULL) )
        FFTW_EXECUTE_DFT_C2R(fft_plan_Fourier_inverse_z, reinterpret_cast<FFTW_Complex*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    else if ( (basis_option[2] == 'S') && (fft_plan_Sine_inverse_z != NULL) ){
        ArrayShiftLeft(Ar, 'Z');
        FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    }
    else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_inverse_z != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "' or inverse plan z not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[10]+=MPI_Wtime();
}


/** @brief Perform the Forward transform
 *
 *  @param basis_option - basis_option can be "FFF", "SFF", "SSF", "SSS"
 *                      - F stands for Fourier
 *                      - S stands for Sine/Cosine
 *  @param Ar - The real space array
 *  @param A  - The Fourier space array
 *
 */
void FFTK::Forward_transform(string basis_option, Array<Real,3> Ar, Array<Complex,3> A)
{
    timer[0]-=MPI_Wtime();
    if ( (basis_option[2] == 'F') && (fft_plan_Fourier_forward_z != NULL) )
        FFTW_EXECUTE_DFT_R2C(fft_plan_Fourier_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<FFTW_Complex*>(Ar.data()));
    else if ( (basis_option[2] == 'S') && (fft_plan_Sine_forward_z != NULL) ) {
        FFTW_EXECUTE_R2R(fft_plan_Sine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
        ArrayShiftRight(Ar, 'Z');
    }
    else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_forward_z != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "' or plan z not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[0]+=MPI_Wtime();


    timer[1]-=MPI_Wtime();
    Transpose_ZY(Ar, IA);
    timer[1]+=MPI_Wtime();


    timer[2]-=MPI_Wtime();
    if ( (basis_option[1] == 'F') && (fft_plan_Fourier_forward_y != NULL) )
        for (int ix = 0; ix < maxix; ix++)
            FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_y, reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()));
    else if ( (basis_option[1] == 'S') && (fft_plan_Sine_forward_y != NULL) ) {
        for (int ix = 0; ix < maxix; ix++)
            FFTW_EXECUTE_R2R(fft_plan_Sine_forward_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
        ArrayShiftRight(IA, 'Y');
    }
    else if ( (basis_option[1] == 'C') && (fft_plan_Cosine_forward_y != NULL) )
        for (int ix = 0; ix < maxix; ix++)
            FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "' or plan y not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[2]+=MPI_Wtime();


    timer[3]-=MPI_Wtime();
    Transpose_YX(IA, A);
    timer[3]+=MPI_Wtime();


    timer[4]-=MPI_Wtime();
    if ( (basis_option[0] == 'F') && (fft_plan_Fourier_forward_x != NULL) )
        FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    else if ( (basis_option[0] == 'S') && (fft_plan_Sine_forward_x != NULL) ) {
        FFTW_EXECUTE_R2R(fft_plan_Sine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
        ArrayShiftRight(A, 'X');
    }
    else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_forward_x != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "' or plan x not initialized" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[4]+=MPI_Wtime();

    timer[5]-=MPI_Wtime();
    Normalize(A);
    timer[5]+=MPI_Wtime();
}


/** @brief Perform the Inverse transform
 *
 *  @param basis_option - basis_option can be "FFF", "SFF", "SSF", "SSS"
 *                      - F stands for Fourier
 *                      - S stands for Sine/Cosine
 *  @param A  - The Fourier space array
 *  @param Ar - The real space array
 *
 */
void FFTK::Inverse_transform(string basis_option, Array<Complex,3> A, Array<Real,3> Ar) 
{
    timer[6]-=MPI_Wtime();
    if ( (basis_option[0] == 'F') && (fft_plan_Fourier_inverse_x != NULL) )
        FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    else if ( (basis_option[0] == 'S') && (fft_plan_Sine_inverse_x != NULL) ) {
        ArrayShiftLeft(A, 'X');
        FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
    }
    else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_inverse_x != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
    else {
        if (my_id==0) cerr << "FFTK::inverse_transform - Invalid basis_option '" << basis_option << "'" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[6]+=MPI_Wtime();


    timer[7]-=MPI_Wtime();
    Transpose_XY(A, IA);
    timer[7]+=MPI_Wtime();


    timer[8]-=MPI_Wtime();
    if ( (basis_option[1] == 'F') && (fft_plan_Fourier_inverse_y != NULL) )
        for (int ix = 0; ix < maxix; ix++)
            FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_y, reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()));
    else if ( (basis_option[1] == 'S') && (fft_plan_Sine_inverse_y != NULL) ){
        ArrayShiftLeft(IA, 'Y');
        for (int ix = 0; ix < maxix; ix++)
            FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
    }
    else if ( (basis_option[1] == 'C') && (fft_plan_Cosine_inverse_y != NULL) )
        for (int ix = 0; ix < maxix; ix++)
            FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
    else {
        if (my_id==0) cerr << "FFTK::inverse_transform - Invalid basis_option '" << basis_option << "'" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[8]+=MPI_Wtime();


    timer[9]-=MPI_Wtime();
    Transpose_YZ(IA, Ar);
    timer[9]+=MPI_Wtime();


    timer[10]-=MPI_Wtime();
    if ( (basis_option[2] == 'F') && (fft_plan_Fourier_inverse_z != NULL) )
        FFTW_EXECUTE_DFT_C2R(fft_plan_Fourier_inverse_z, reinterpret_cast<FFTW_Complex*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    else if ( (basis_option[2] == 'S') && (fft_plan_Sine_inverse_z != NULL) ){
        ArrayShiftLeft(Ar, 'Z');
        FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    }
    else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_inverse_z != NULL) )
        FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    else {
        if (my_id==0) cerr << "FFTK::forward_transform - Invalid basis_option '" << basis_option << "'" << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    timer[10]+=MPI_Wtime();
}
