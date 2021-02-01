
#include "fftk_def_vars.h"
#include "global.h"
#include "communication.h"
#include "../adapter/FFTW_Adapter.h"

#ifndef _SSS_H
#define _SSS_H

namespace FFTK {	
	class SSS {
	    shared_ptr<Global> global;
	    FFTW_Adapter fftw_adapter;

	    Communication communication;

	    
	    Array<Complex,3> IA;
	    

	    Real normalizing_factor;
	    
	public:
		SSS(shared_ptr<Global> global);

		void set_field_vars();
		void init_transform();
		void Init_timers();
		void Forward_transform_2d(string basis_option, Array<Real,2> Ar, Array<Complex,2> A);
		void Inverse_transform_2d(string basis_option, Array<Complex,2> A, Array<Real,2> Ar);
		void Forward_transform_3d(string basis_option, Array<Real,3> Ar, Array<Complex,3> A);
		void Inverse_transform_3d(string basis_option, Array<Complex,3> A, Array<Real,3> Ar);
		void Finalize();
	};
}

#endif
