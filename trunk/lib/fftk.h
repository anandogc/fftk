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
#include "adapter/FFTW_Adapter_def_vars.h"
#include <functional>

#ifndef _FFTK_H
#define _FFTK_H

#include "global.h"
#include "utilities.h"

#include "basis/FFZ.h"

namespace FFTK {

	class FFTK {

	    template<class Basis>
	    void Init_with_basis();

	    void Init_with_basis_with_FFZ();
	    


	public:
		Utilities utilities;

	    shared_ptr<Global> global;
		double *timer;

		void Init(string basis, int Nx, int Ny, int Nz, int num_p_rows, MPI_Comm MPI_COMMUNICATOR);


		function<void (string basis, int Nx, int Ny, int Nz, int num_p_rows)> Init_basis;
		function<void ()> Finalize;
		
		function<void (string basis_option, Array<Complex,2> RA, Array<Complex,2> FA)> Forward_transform_2d_C2C;
		function<void (string basis_option, Array<Complex,2> FA, Array<Complex,2> RA)> Inverse_transform_2d_C2C;

		function<void (string basis_option, Array<Real,2> RA, Array<Complex,2> FA)> Forward_transform_2d;
		function<void (string basis_option, Array<Complex,2> FA, Array<Real,2> RA)> Inverse_transform_2d;

		function<void (string basis_option, Array<Complex,3> RA, Array<Complex,3> FA)> Forward_transform_3d_C2C;
		function<void (string basis_option, Array<Complex,3> FA, Array<Complex,3> RA)> Inverse_transform_3d_C2C;

		function<void (string basis_option, Array<Real,3> RA, Array<Complex,3> FA)> Forward_transform_3d;
		function<void (string basis_option, Array<Complex,3> FA, Array<Real,3> RA)> Inverse_transform_3d;
		
		function<void (Array<Real,2> Ar)> Zero_pad_last_plane_2D;
		function<void (Array<Real,3> Ar)> Zero_pad_last_plane_3D;

		function<void (Array<Complex,2> A)> Normalize_2D;
		function<void (Array<Complex,3> A)> Normalize_3D;


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
};
#endif
