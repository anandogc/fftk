
#include "fftk_def_vars.h"
#include "global.h"
#include "communication.h"
#include "../adapter/FFTW_Adapter.h"

#ifndef _FFZ_H
#define _FFZ_H

namespace FFTK {	
	class FFZ {
	    shared_ptr<Global> global;
	    FFTW_Adapter fftw_adapter;

	    Communication communication;

	    
	    Array<Complex,3> IA;
	    

	    Real normalizing_factor;
	    
	public:
		FFZ(shared_ptr<Global> global);
		
		void set_field_vars();
		void init_transform();
		void Forward_transform_2d(string basis_option, Array<Complex,2> A1, Array<Complex,2> A2);
		void Inverse_transform_2d(string basis_option, Array<Complex,2> A2, Array<Complex,2> A1);
		void Forward_transform_3d(string basis_option, Array<Complex,3> A1, Array<Complex,3> A2);
		void Inverse_transform_3d(string basis_option, Array<Complex,3> A2, Array<Complex,3> A1);
		void Finalize();
	};
}

#endif
