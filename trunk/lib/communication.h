#include "fftk_def_vars.h"
#include "global.h"
#include "utilities.h"


#ifndef _Communication_H
#define _Communication_H

namespace FFTK {

	class Communication {
		shared_ptr<Global> global;
	    Utilities utilities;
		
	private:

		MPI_Datatype X_axes_selector_vector;
		MPI_Datatype Y_axes_selector_for_XY_vector;
		MPI_Datatype Y_axes_selector_for_YZ_vector;
		MPI_Datatype Z_axes_selector_vector;

		MPI_Datatype X_axes_selector;
		MPI_Datatype Y_axes_selector_for_XY;
		MPI_Datatype Y_axes_selector_for_YZ;
		MPI_Datatype Z_axes_selector;

	    Array<Complex,3> IA;

	public:
		double timer[22];

		Communication(shared_ptr<Global> global);
		void Init();
		void Transpose(Array<Complex,2> FA, Array<Real,2> RA);
		void Transpose(Array<Real,2> RA, Array<Complex,2> FA);

		void Transpose_XZ_2d(Array<Complex,2> A1, Array<Complex,2> A2);
		void Transpose_ZX_2d(Array<Complex,2> A2, Array<Complex,2> A1);

		void Transpose(Array<Real,3> Ar, Array<Complex,3> A);
		void Transpose(Array<Complex,3> A, Array<Real,3> Ar);
		void Transpose_XY(Array<Complex,3> FA, Array<Complex,3> IA);
		void Transpose_YX(Array<Complex,3> IA, Array<Complex,3> FA);
		void Transpose_YZ(Array<Complex,3> IA, Array<Real,3> RA);
		void Transpose_ZY(Array<Real,3> RA, Array<Complex,3> IA);
		void Transpose_YZ(Array<Complex,3> IA, Array<Complex,3> RA);
		void Transpose_ZY(Array<Complex,3> RA, Array<Complex,3> IA);
		void Finalize();
	};
}
#endif