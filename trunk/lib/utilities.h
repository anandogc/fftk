#ifndef _Utilities_H
#define _Utilities_H

namespace FFTK {
	class Utilities {
	public:
		shared_ptr<Global> global;

		Utilities();
		Utilities(shared_ptr<Global> global);
		void Zero_pad_last_plane(Array<Real,2> Ar);
		void Zero_pad_last_plane(Array<Real,3> Ar);
		void ArrayShiftRight_basic(void *data, TinyVector<int,3> shape, int axis);
		void ArrayShiftRight(Array<Real,3> Ar, int axis);
		void ArrayShiftRight(Array<Complex,3> A, int axis);
		void ArrayShiftRight(Array<Real,2> Ar, int axis);
		void ArrayShiftRight(Array<Complex,2> A, int axis);
		void ArrayShiftLeft_basic(void *data, TinyVector<int,3> shape, int axis);
		void ArrayShiftLeft(Array<Real,3> Ar, int axis);
		void ArrayShiftLeft(Array<Complex,3> A, int axis);
		void ArrayShiftLeft(Array<Real,2> Ar, int axis);
		void ArrayShiftLeft(Array<Complex,2> A, int axis);
		int Get_Rz_comm();
		void Init_timers();
		void To_slab(Array<Real,3> ArPencil, Array<Real,3> ArSlab);
		void To_pencil(Array<Real,3> ArSlab, Array<Real,3> ArPencil);
	};
}
#endif