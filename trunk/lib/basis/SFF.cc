#include "SFF.h"

namespace FFTK {

    SFF::SFF(shared_ptr<Global> global): global(global), fftw_adapter(global), communication(global) {
        set_field_vars();
        communication.Init();
        fftw_adapter.init_transform("SFF");
        init_transform();
	}

    void SFF::set_field_vars() {
        Global::Field &field = global->field;
		  
        field.Fx = field.Nx;
        field.Ix = field.Nx;
        field.Rx = field.Nx;

        field.Fy = field.Ny;
        field.Iy = field.Ny;
        field.Ry = field.Ny;

        field.Fz = field.Nz/2+1;
        field.Iz = field.Nz/2+1;
        field.Rz = field.Nz+2;

        field.Fz_unit = 2;
        field.Iz_unit = 2;
        field.Rz_unit = 1;
    }   

    /** 
     *  @brief Initialize transform related variables
     *
     */
    void SFF::init_transform() {
        Global::Field &field = global->field;
        
        normalizing_factor=1;
        IA.resize(field.IA_shape);

                
        normalizing_factor /= 2*Real(field.Nx)*Real(field.Ny)*Real(field.Nz);

        

    }

    

    void SFF::Forward_transform_2d(string basis_option, Array<Real,2> Ar, Array<Complex,2> A) {
        
		TIMER_START(0);
        fftw_adapter.DFT_forward_R2C(Ar);
        TIMER_END(0);

        TIMER_START(1);
        communication.Transpose(Ar, A);
        TIMER_END(1);


        TIMER_START(4);
        fftw_adapter.DFT_forward_R2R_x(basis_option, A);
        TIMER_END(4);

        TIMER_START(5);
        A *= normalizing_factor;
        TIMER_END(5);
    }


    /** @brief Perform the Inverse transform
     *
     *  @param basis_option - basis_option can be "SFF", "SFF", "SSF", "SSS"
     *                      - F stands for Fourier
     *                      - S stands for Sine/Cosine
     *  @param A  - The Fourier space array
     *  @param Ar - The real space array
     *
     */
    void SFF::Inverse_transform_2d(string basis_option, Array<Complex,2> A, Array<Real,2> Ar) {
        
        TIMER_START(6);
        fftw_adapter.DFT_inverse_R2R_x(basis_option, A);
        TIMER_END(6);

        TIMER_START(7);
        communication.Transpose(A, Ar);
        TIMER_END(7);

        TIMER_START(10);
        fftw_adapter.DFT_inverse_C2R(Ar);
        TIMER_END(10);
    }


    /** @brief Perform the Forward transform
     *
     *  @param basis_option - basis_option can be "SFF", "SFF", "SSF", "SSS"
     *                      - F stands for Fourier
     *                      - S stands for Sine/Cosine
     *  @param Ar - The real space array
     *  @param A  - The Fourier space array
     *
     */
    void SFF::Forward_transform_3d(string basis_option, Array<Real,3> Ar, Array<Complex,3> A) {
        
        TIMER_START(0);
        fftw_adapter.DFT_forward_R2C(Ar);
        TIMER_END(0);

        TIMER_START(1);
        communication.Transpose_ZY(Ar, IA);
        TIMER_END(1);

        TIMER_START(2);
        fftw_adapter.DFT_forward_y(IA);
        TIMER_END(2);

        TIMER_START(3);
        communication.Transpose_YX(IA, A);
        TIMER_END(3);

        TIMER_START(4);
        fftw_adapter.DFT_forward_R2R_x(basis_option, A);
        TIMER_END(4);

        TIMER_START(5);
        A *= normalizing_factor;
        TIMER_END(5);
    }


    /** @brief Perform the Inverse transform
     *
     *  @param basis_option - basis_option can be "SFF", "SFF", "SSF", "SSS"
     *                      - F stands for Fourier
     *                      - S stands for Sine/Cosine
     *  @param A  - The Fourier space array
     *  @param Ar - The real space array
     *
     */
    void SFF::Inverse_transform_3d(string basis_option, Array<Complex,3> A, Array<Real,3> Ar) {
        
        TIMER_START(6);
        fftw_adapter.DFT_inverse_R2R_x(basis_option, A);
        TIMER_END(6);

        TIMER_START(7);
        communication.Transpose_XY(A, IA);
        TIMER_END(7);

        TIMER_START(8);
        fftw_adapter.DFT_inverse_y(IA);
         TIMER_END(8);

        TIMER_START(9);
        communication.Transpose_YZ(IA, Ar);
        TIMER_END(9);

        TIMER_START(10);
        fftw_adapter.DFT_inverse_C2R(Ar);
        TIMER_END(10);
    }

    void SFF::Finalize() {
        
        fftw_adapter.Finalize();        
        communication.Finalize();
       
    }
}
