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
 * \file  finalize.cc
 * @author  A. G. Chatterjee, M. K. Verma
 * @date Jan 2017
 * @bug  No known bugs
 */

#include "fftk.h"

void FFTK::Finalize() {
    delete[] buffer_Y;
    delete[] buffer_Z;

    FFTW_DESTROY_PLAN(fft_plan_Fourier_forward_x);
    FFTW_DESTROY_PLAN(fft_plan_Fourier_inverse_x);
    FFTW_DESTROY_PLAN(fft_plan_Sine_forward_x);
    FFTW_DESTROY_PLAN(fft_plan_Sine_inverse_x);
    FFTW_DESTROY_PLAN(fft_plan_Cosine_forward_x);
    FFTW_DESTROY_PLAN(fft_plan_Cosine_inverse_x);
    FFTW_DESTROY_PLAN(fft_plan_Fourier_forward_y);
    FFTW_DESTROY_PLAN(fft_plan_Fourier_inverse_y);
    FFTW_DESTROY_PLAN(fft_plan_Sine_forward_y);
    FFTW_DESTROY_PLAN(fft_plan_Sine_inverse_y);
    FFTW_DESTROY_PLAN(fft_plan_Cosine_forward_y);
    FFTW_DESTROY_PLAN(fft_plan_Cosine_inverse_y);
    FFTW_DESTROY_PLAN(fft_plan_Fourier_forward_z);
    FFTW_DESTROY_PLAN(fft_plan_Fourier_inverse_z);
    FFTW_DESTROY_PLAN(fft_plan_Sine_forward_z);
    FFTW_DESTROY_PLAN(fft_plan_Sine_inverse_z);
    FFTW_DESTROY_PLAN(fft_plan_Cosine_forward_z);
    FFTW_DESTROY_PLAN(fft_plan_Cosine_inverse_z);
}
