
#include "fftk.h"
#include "basis/FFF.h"
#include "basis/SSS.h"
#include "basis/SFF.h"
#include "basis/SSF.h"
#include "basis/FFZ.h"

namespace FFTK {
    void FFTK::Init(string basis_name, int Nx, int Ny, int Nz, int num_p_rows, MPI_Comm MPI_COMMUNICATOR) {
        global = make_shared<Global>();

        global->field.basis_name = basis_name;

        global->field.Nx = Nx;
        global->field.Ny = Ny;
        global->field.Nz = Nz;

        global->mpi.num_p_rows = num_p_rows;

        global->mpi.MPI_COMMUNICATOR = MPI_COMMUNICATOR;

        utilities.global = global;

#ifdef TIMERS
        utilities.Init_timers();
#endif

        if ( basis_name == "FFF" ) {
            Init_with_basis<FFF>();
        }
        else if ( basis_name == "SSS") {
            Init_with_basis<SSS>();
        }
        else if ( basis_name == "SFF") {
            Init_with_basis<SFF>();
        }
        else if ( basis_name == "SSF") {
            Init_with_basis<SSF>();
        }
        else if ( basis_name == "FFZ") {
            Init_with_basis_with_FFZ();
        }

    }

        
    template<class Basis>
    void FFTK::Init_with_basis() {
        auto basis = shared_ptr<Basis>(new Basis(global));

        Forward_transform_2d  = [basis](string basis_option, Array<Real,2> RA, Array<Complex,2> FA) { 
                                    basis->Forward_transform_2d(basis_option, RA, FA);
                                };
                                    
        Inverse_transform_2d  = [basis](string basis_option, Array<Complex,2> FA, Array<Real,2> RA) { 
                                    basis->Inverse_transform_2d(basis_option, FA, RA);
                                };

        Forward_transform_3d  = [basis](string basis_option, Array<Real,3> RA, Array<Complex,3> FA) { 
                                    basis->Forward_transform_3d(basis_option, RA, FA);
                                };
                                
        Inverse_transform_3d  = [basis](string basis_option, Array<Complex,3> FA, Array<Real,3> RA) { 
                                    basis->Inverse_transform_3d(basis_option, FA, RA);
                                };

        Finalize = [basis](){
            basis->Finalize();
        };
        
    }


    void FFTK::Init_with_basis_with_FFZ() {
        auto basis = shared_ptr<FFZ>(new FFZ(global));

        Forward_transform_2d_C2C  = [basis](string basis_option, Array<Complex,2> RA, Array<Complex,2> FA) { 
                                    basis->Forward_transform_2d(basis_option, RA, FA);
                                };
                                    
        Inverse_transform_2d_C2C  = [basis](string basis_option, Array<Complex,2> FA, Array<Complex,2> RA) { 
                                    basis->Inverse_transform_2d(basis_option, FA, RA);
                                };

        Forward_transform_3d_C2C  = [basis](string basis_option, Array<Complex,3> RA, Array<Complex,3> FA) { 
                                    basis->Forward_transform_3d(basis_option, RA, FA);
                                };
                                
        Inverse_transform_3d_C2C  = [basis](string basis_option, Array<Complex,3> FA, Array<Complex,3> RA) { 
                                    basis->Inverse_transform_3d(basis_option, FA, RA);
                                };
                                
        Finalize = [basis](){
            basis->Finalize();
        };
    }

    MPI_Comm FFTK::Get_communicator(string which) {
        if (which=="ROW")
            return global->mpi.MPI_COMM_ROW;
        else if (which=="COL")
            return global->mpi.MPI_COMM_COL;
        else {
            if (global->mpi.my_id==0)
                cerr << "Invalid communicator: " << which << endl;
            MPI_Finalize();
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
    }


    TinyVector<int,3> FFTK::Get_FA_shape()
    {
        return global->field.FA_shape;
    }
    TinyVector<int,3> FFTK::Get_IA_shape()
    {
        return global->field.IA_shape;
    }
    TinyVector<int,3> FFTK::Get_RA_shape()
    {
        return global->field.RA_shape;
    }

    TinyVector<int,3> FFTK::Get_FA_start()
    {
        return TinyVector<int, 3>(global->field.fx_start,global->field.fy_start,global->field.fz_start);
    }
    TinyVector<int,3> FFTK::Get_IA_start()
    {
        return TinyVector<int, 3>(global->field.ix_start,global->field.iy_start,global->field.iz_start);
    }
    TinyVector<int,3> FFTK::Get_RA_start()
    {
        return TinyVector<int, 3>(global->field.rx_start,global->field.ry_start,global->field.rz_start);
    }

    int FFTK::Get_row_id()
    {
        return global->mpi.my_row_id;
    }
    int FFTK::Get_col_id()
    {
        return global->mpi.my_col_id;
    }

    int FFTK::Get_num_p_rows()
    {
        return global->mpi.num_p_rows;
    }
    int FFTK::Get_num_p_cols()
    {
        return global->mpi.num_p_cols;
    }


       
}
