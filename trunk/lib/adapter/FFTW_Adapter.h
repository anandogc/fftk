#include "FFTW_Adapter_def_vars.h"
#include "global.h"
#include <fftw3.h>
#include "utilities.h"


#ifndef _FFTW_Adapter_H
#define _FFTW_Adapter_H


namespace FFTK {	
	class FFTW_Adapter {
	    shared_ptr<Global> global;
	    Utilities utilities;

	    //plans for FFF basis
		FFTW_PLAN fft_plan_Fourier_forward_x;
        FFTW_PLAN fft_plan_Fourier_inverse_x;
        FFTW_PLAN fft_plan_Fourier_forward_y;
        FFTW_PLAN fft_plan_Fourier_inverse_y;
        FFTW_PLAN fft_plan_Fourier_forward_z;
        FFTW_PLAN fft_plan_Fourier_inverse_z;

        //plans for sine/ cosine basis
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

        Array<Complex,3> FA;
        Array<Complex,3> IA;
        Array<Real,3> RA;

	public: 
		FFTW_Adapter();
		FFTW_Adapter(shared_ptr<Global> global);
		void init_transform(string basis);
		void DFT_forward_x(Array<Complex,2> A);
		void DFT_forward_R2C(Array<Real,2> Ar);
		void DFT_inverse_x(Array<Complex,2> A);
		void DFT_inverse_C2R(Array<Real,2> Ar);

		void DFT_forward_x(Array<Complex,3> A);
		void DFT_forward_R2C(Array<Real,3> Ar);
		void DFT_forward_y(Array<Complex,3> IA);
		void DFT_inverse_x(Array<Complex,3> A);
		void DFT_inverse_C2R(Array<Real,3> Ar);
		void DFT_inverse_y(Array<Complex,3> IA);

		void DFT_forward_R2R(string basis_option, Array<Real,2> Ar);
		void DFT_forward_R2R_x(string basis_option, Array<Complex,2> A);
		void DFT_inverse_R2R(string basis_option, Array<Real,2> Ar);
		void DFT_inverse_R2R_x(string basis_option, Array<Complex,2> A);

		void DFT_forward_R2R(string basis_option, Array<Real,3> Ar);
		void DFT_forward_R2R_x(string basis_option, Array<Complex,3> A);
		void DFT_inverse_R2R(string basis_option, Array<Real,3> Ar);
		void DFT_inverse_R2R_x(string basis_option, Array<Complex,3> A);
		void DFT_forward_R2R_y(string basis_option, Array<Complex,3> IA);
		void DFT_inverse_R2R_y(string basis_option, Array<Complex,3> IA);

		void DFT_forward_C2C(string basis_option, Array<Complex,2> A);
		void DFT_inverse_C2C(string basis_option, Array<Complex,2> A);
		void DFT_forward_C2C(string basis_option, Array<Complex,3> A);
		void DFT_inverse_C2C(string basis_option, Array<Complex,3> A);


		void Finalize();
	};
}

#endif
