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

#include <mpi.h>
#include <blitz/array.h>
#include "adapter/FFTW_Adapter_def_vars.h"
#include "global.h"
#include "utilities.h"


#include <functional>
#include <memory>


#ifndef _FFTK_DEF_VARS_H
#define _FFTK_DEF_VARS_H


// Defining REAL_DOUBLE: switch for setting double
#if defined(REAL_DOUBLE)
#define MPI_Real                        MPI_DOUBLE

#elif defined(REAL_FLOAT)
// Define REAL_FLOAT: switch for setting float
#define MPI_Real                        MPI_FLOAT


#endif
/*
#ifndef Complex
#define Complex  complex<Real>
#endif
*/
// Switches for FFTW plans 


#define IN_ALL_PROCS(l, expr) \
for (int l=0; l<numprocs; l++) { \
    if (my_id==l)\
        {expr;}\
    MPI_Barrier(MPI_COMM_WORLD);\
}


#ifdef TIMERS
    #define TIMER_START(n) global->misc.timer[n] -= MPI_Wtime()
    #define TIMER_END(n) global->misc.timer[n] += MPI_Wtime()
#else
    #define TIMER_START(n) 
    #define TIMER_END(n)
#endif

#endif
