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
 * \file  fftk.h
 * @author  A. G. Chatterjee, M. K. Verma
 * @date Jan 2017
 * @bug  No known bugs
 */

#include "fftk_def_vars.h"

#ifndef _FFTK_H
#define _FFTK_H

class FFTK {
private:
	string basis;

	int my_id;
	int numprocs;

	MPI_Comm MPI_COMM_ROW;
	MPI_Comm MPI_COMM_COL;

	int my_row_id;
	int my_col_id;

	int Nx,Ny,Nz;

	int Fx,Fy,Fz;
	int Ix,Iy,Iz;
	int Rx,Ry,Rz;

	int Fz_unit, Iz_unit, Rz_unit;

	int num_p_rows;
	int num_p_cols;

	//Set Array size
	TinyVector<int,3> FA_shape, IA_shape, RA_shape;

	int maxfx, maxfy, maxfz;
	int maxix, maxiy, maxiz;
	int maxrx, maxry, maxrz;

	int fx_start, fy_start, fz_start;
	int ix_start, iy_start, iz_start;
	int rx_start, ry_start, rz_start;

	FFTW_PLAN fft_plan_Fourier_forward_x;
	FFTW_PLAN fft_plan_Fourier_inverse_x;
	FFTW_PLAN fft_plan_Fourier_forward_y;
	FFTW_PLAN fft_plan_Fourier_inverse_y;
	FFTW_PLAN fft_plan_Fourier_forward_z;
	FFTW_PLAN fft_plan_Fourier_inverse_z;

	FFTW_PLAN fft_plan_Sine_forward_x;
	FFTW_PLAN fft_plan_Sine_inverse_x;
	FFTW_PLAN fft_plan_Sine_forward_y;
	FFTW_PLAN fft_plan_Sine_inverse_y;
	FFTW_PLAN fft_plan_Sine_forward_z;
	FFTW_PLAN fft_plan_Sine_inverse_z;

	FFTW_PLAN fft_plan_Cosine_forward_x;
	FFTW_PLAN fft_plan_Cosine_inverse_x;
	FFTW_PLAN fft_plan_Cosine_forward_y;
	FFTW_PLAN fft_plan_Cosine_inverse_y;
	FFTW_PLAN fft_plan_Cosine_forward_z;
	FFTW_PLAN fft_plan_Cosine_inverse_z;

	Real *buffer_Y;
	Real *buffer_Z;

	size_t plane_size_YX;
	size_t copy_size_YX;
	size_t block_size_YX;

	size_t plane_size_YZ;
	size_t block_size_YZ;
	size_t copy_size_YZ;
	//


	Array<Complex,3> FA;
	Array<Complex,3> IA;
	Array<Real,3> RA;

	Real normalizing_factor;

	void set_vars();
	void init_transpose();
	void init_transform();
	void init_timers();
	void Transpose_XY(Array<Complex,3> A1, Array<Complex,3> A2);
	void Transpose_YX(Array<Complex,3> A1, Array<Complex,3> A2);
	void Transpose_YZ(Array<Complex,3> A, Array<Real,3> Ar);
	void Transpose_ZY(Array<Real,3> Ar, Array<Complex,3> A);

	void ArrayShiftRight(Array<Real,2> Ar, int axis);
	void ArrayShiftRight(Array<Complex,2> A, int axis);
	void ArrayShiftLeft(Array<Real,2> Ar, int axis);
	void ArrayShiftLeft(Array<Complex,2> A, int axis);

	void ArrayShiftRight_basic(void *data, TinyVector<int,3> shape, int axis);
	void ArrayShiftRight(Array<Real,3> Ar, int axis);
	void ArrayShiftRight(Array<Complex,3> A, int axis);
	void ArrayShiftLeft_basic(void *data, TinyVector<int,3> shape, int axis);
	void ArrayShiftLeft(Array<Real,3> Ar, int axis);
	void ArrayShiftLeft(Array<Complex,3> A, int axis);

public:
	double *timer;

	void Init(string basis, int Nx, int Ny, int Nz, int num_p_rows);
	void Finalize();
	
	void Forward_transform(string basis_option, Array<Real,2> RA, Array<Complex,2> FA);
	void Inverse_transform(string basis_option, Array<Complex,2> FA, Array<Real,2> RA);

	void Forward_transform(string basis_option, Array<Real,3> RA, Array<Complex,3> FA);
	void Inverse_transform(string basis_option, Array<Complex,3> FA, Array<Real,3> RA);

	void Transpose(Array<Real,2> RA, Array<Complex,2> FA);
	void Transpose(Array<Complex,2> FA, Array<Real,2> RA);
	
	void Transpose(Array<Real,3> Ar, Array<Complex,3> A);
	void Transpose(Array<Complex,3> A, Array<Real,3> Ar);
	
	void Zero_pad_last_plane(Array<Real,2> Ar);
	void Zero_pad_last_plane(Array<Real,3> Ar);

	void Normalize(Array<Complex,2> A);
	void Normalize(Array<Complex,3> A);

	void To_slab(Array<Real,3> ArPencil, Array<Real,3> ArSlab);
	void To_pencil(Array<Real,3> ArSlab, Array<Real,3> ArPencil);

	MPI_Comm Get_communicator(string which);

	TinyVector<int,3> Get_FA_shape();
	TinyVector<int,3> Get_IA_shape();
	TinyVector<int,3> Get_RA_shape();

	TinyVector<int,3> Get_FA_start();
	TinyVector<int,3> Get_IA_start();
	TinyVector<int,3> Get_RA_start();

	int Get_row_id();
	int Get_col_id();

	int Get_num_p_rows();
	int Get_num_p_cols();

};

#endif
