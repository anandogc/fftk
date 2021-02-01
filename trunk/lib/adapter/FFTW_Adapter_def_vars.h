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
 * \file  fftk_def_vars.h
 * @author  A. G. Chatterjee, M. K. Verma
 * @date Jan 2017
 * @bug  No known bugs
 */

#include <fftw3.h>
#include <blitz/array.h>

#ifdef FFTK_THREADS
	#include <omp.h>
#endif

#include <complex>
#include <string>
#include <functional>
#include <memory>

using namespace std;
using namespace blitz;

#ifndef _FFTW_ADAPTER_DEF_VARS_H
#define _FFTW_ADAPTER_DEF_VARS_H


// Defining REAL_DOUBLE: switch for setting double
#if defined(REAL_DOUBLE)


#define Real                            double



#define FFTW_Complex                    fftw_complex
#define FFTW_PLAN                       fftw_plan
// #define FFTW_MPI_INIT                   fftw_mpi_init

#define FFTW_PLAN_MANY_R2R              fftw_plan_many_r2r
#define FFTW_PLAN_MANY_DFT              fftw_plan_many_dft
#define FFTW_PLAN_MANY_DFT_R2C          fftw_plan_many_dft_r2c
#define FFTW_PLAN_MANY_DFT_C2R          fftw_plan_many_dft_c2r

#define FFTW_EXECUTE                    fftw_execute
#define FFTW_EXECUTE_R2R                fftw_execute_r2r
#define FFTW_EXECUTE_DFT                fftw_execute_dft
#define FFTW_EXECUTE_DFT_R2C            fftw_execute_dft_r2c
#define FFTW_EXECUTE_DFT_C2R            fftw_execute_dft_c2r
#define FFTW_DESTROY_PLAN               fftw_destroy_plan


// Define REAL_FLOAT: switch for setting float
#elif defined(REAL_FLOAT)

#define Real                            float


#define FFTW_Complex                    fftwf_complex
#define FFTW_PLAN                       fftwf_plan
// #define FFTW_MPI_INIT                   fftwf_mpi_init

#define FFTW_PLAN_MANY_R2R              fftwf_plan_many_r2r
#define FFTW_PLAN_MANY_DFT              fftwf_plan_many_dft
#define FFTW_PLAN_MANY_DFT_R2C          fftwf_plan_many_dft_r2c
#define FFTW_PLAN_MANY_DFT_C2R          fftwf_plan_many_dft_c2r

#define FFTW_EXECUTE                    fftwf_execute
#define FFTW_EXECUTE_R2R                fftwf_execute_r2r
#define FFTW_EXECUTE_DFT                fftwf_execute_dft
#define FFTW_EXECUTE_DFT_R2C            fftwf_execute_dft_r2c
#define FFTW_EXECUTE_DFT_C2R            fftwf_execute_dft_c2r
#define FFTW_DESTROY_PLAN               fftwf_destroy_plan

#endif


#ifndef Complex
#define Complex  complex<Real>
#endif

// Switches for FFTW plans 

#if defined(ESTIMATE)
#define FFTW_PLAN_FLAG FFTW_ESTIMATE

#elif defined(MEASURE)
#define FFTW_PLAN_FLAG FFTW_MEASURE

#elif defined(PATIENT)
#define FFTW_PLAN_FLAG FFTW_PATIENT

#elif defined(EXHAUSTIVE)
#define FFTW_PLAN_FLAG FFTW_EXHAUSTIVE
#endif


#endif
