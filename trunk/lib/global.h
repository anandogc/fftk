#include <string>
#include <blitz/array.h>
#include <mpi.h>

#ifndef _Global_H
#define _Global_H

using namespace blitz;
using namespace std;

namespace FFTK {
	class Global {
	public:
		struct Field {
			string basis_name;
			
			
			int Nx, Ny, Nz;

			int Fx, Fy, Fz;
			int Ix, Iy, Iz;
			int Rx, Ry, Rz;

			int maxfx, maxfy, maxfz;
			int maxix, maxiy, maxiz;
			int maxrx, maxry, maxrz;

			int fx_start, fy_start, fz_start;
			int ix_start, iy_start, iz_start;
			int rx_start, ry_start, rz_start;

			int Fz_unit, Iz_unit, Rz_unit;

			TinyVector<int,3> FA_shape, IA_shape, RA_shape;

		} field;

		struct MPI {
				int my_id;
				int numprocs;

				MPI_Comm MPI_COMMUNICATOR;
				MPI_Comm MPI_COMM_ROW;
				MPI_Comm MPI_COMM_COL;

				int my_row_id;
				int my_col_id;

				int Rz_comm;  // Value of Rz when used in creating new MPI_Datatype

				int num_p_rows;
				int num_p_cols;
		} mpi;

		struct Misc {
				double timer[22];
		} misc;
	};
}

#endif