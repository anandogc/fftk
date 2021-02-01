#include "FFZ.h"

namespace FFTK {

    FFZ::FFZ(shared_ptr<Global> global): global(global), fftw_adapter(global), communication(global) {
        set_field_vars();
        communication.Init();
        fftw_adapter.init_transform("FFZ");
        init_transform();
	}

    void FFZ::set_field_vars() {
        Global::Field &field = global->field;
		  
        field.Fx = field.Nx;
        field.Ix = field.Nx;
        field.Rx = field.Nx;

        field.Fy = field.Ny;
        field.Iy = field.Ny;
        field.Ry = field.Ny;

        field.Fz = field.Nz;
        field.Iz = field.Nz;
        field.Rz = field.Nz;

        field.Fz_unit = 2;
        field.Iz_unit = 2;
        field.Rz_unit = 2;
    }   

    /** 
     *  @brief Initialize transform related variables
     *
     */
    void FFZ::init_transform() {
        Global::Field &field = global->field;
        
        normalizing_factor=1;
        IA.resize(field.IA_shape);

                
        normalizing_factor /= Real(field.Nx)*Real(field.Ny)*Real(field.Nz);

        

    }



    void FFZ::Forward_transform_2d(string basis_option, Array<Complex,2> A1, Array<Complex,2> A2) {
        
		TIMER_START(0);
        fftw_adapter.DFT_forward_C2C(basis_option, A1);
        TIMER_END(0);

        TIMER_START(1);
        communication.Transpose_ZX_2d(A1, A2); //yet to impliment
        TIMER_END(1);


        TIMER_START(4);
        fftw_adapter.DFT_forward_x(A2);
        TIMER_END(4);

        TIMER_START(5);
        A2 *= normalizing_factor;
        TIMER_END(5);
    }


    /** @brief Perform the Inverse transform
     *
     *  @param basis_option - basis_option can be "FFZ", "SFF", "SSF", "SSS"
     *                      - F stands for Fourier
     *                      - S stands for Sine/Cosine
     *  @param A  - The Fourier space array
     *  @param Ar - The real space array
     *
     */
    void FFZ::Inverse_transform_2d(string basis_option, Array<Complex,2> A2, Array<Complex,2> A1) {
        
        TIMER_START(6);
        fftw_adapter.DFT_inverse_x(A2);
        TIMER_END(6);

        TIMER_START(7);
        communication.Transpose_XZ_2d(A2, A1); //yet to impliment
        TIMER_END(7);

        TIMER_START(10);
        fftw_adapter.DFT_inverse_C2C(basis_option, A1);
        TIMER_END(10);
    }


    /** @brief Perform the Forward transform
     *
     *  @param basis_option - basis_option can be "FFZ", "SFF", "SSF", "SSS"
     *                      - F stands for Fourier
     *                      - S stands for Sine/Cosine
     *  @param Ar - The real space array
     *  @param A  - The Fourier space array
     *
     */
    void FFZ::Forward_transform_3d(string basis_option, Array<Complex,3> A1, Array<Complex,3> A2) {
        
        TIMER_START(0);
        fftw_adapter.DFT_forward_C2C(basis_option, A1);
        TIMER_END(0);

        TIMER_START(1);
        communication.Transpose_ZY(A1, IA);
        TIMER_END(1);

        TIMER_START(2);
        fftw_adapter.DFT_forward_y(IA);
        TIMER_END(2);

        TIMER_START(3);
        communication.Transpose_YX(IA, A2);
        TIMER_END(3);

        TIMER_START(4);
        fftw_adapter.DFT_forward_x(A2);
        TIMER_END(4);

        TIMER_START(5);
        A2 *= normalizing_factor;
        TIMER_END(5);
    }


    /** @brief Perform the Inverse transform
     *
     *  @param basis_option - basis_option can be "FFZ", "SFF", "SSF", "SSS"
     *                      - F stands for Fourier
     *                      - S stands for Sine/Cosine
     *  @param A  - The Fourier space array
     *  @param Ar - The real space array
     *
     */
    void FFZ::Inverse_transform_3d(string basis_option, Array<Complex,3> A2,  Array<Complex,3> A1) {
        
        TIMER_START(6);
        fftw_adapter.DFT_inverse_x(A2);
        TIMER_END(6);

        TIMER_START(7);
        communication.Transpose_XY(A2, IA);
        TIMER_END(7);

        TIMER_START(8);
        fftw_adapter.DFT_inverse_y(IA);
        TIMER_END(8);

        TIMER_START(9);
        communication.Transpose_YZ(IA, A1);
        TIMER_END(9);

        TIMER_START(10);
        fftw_adapter.DFT_inverse_C2C(basis_option, A1);
        TIMER_END(10);
    }

    void FFZ::Finalize() {
        
        fftw_adapter.Finalize();
        communication.Finalize();

       
    }
}
