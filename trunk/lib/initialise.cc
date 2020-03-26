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
 * \file  initialise.cc
 * @author  A. G. Chatterjee, M. K. Verma
 * @date Jan 2017
 * @bug  No known bugs
 */

#include "fftk.h"


/** @brief Initializes the FFTK accorging to basis, grid size, and division of data
 *
 *  @param basis - basis can be "FFF", "SFF", "SSF", "SSS"
 *  @param Nx, Ny, Nz - grid size. They must be power or prime. For example [512, 512, 512]
 *  @param num_p_rows - Number of rows in processor division
 *
 */
void FFTK::Init(string basis, int Nx, int Ny, int Nz, int num_p_rows)
{
    this->basis=basis;
    this->Nx=Nx;
    this->Ny=Ny;
    this->Nz=Nz;
    this->num_p_rows=num_p_rows;

    //Fx, Fy, Fz represents array dimentions in Fourier space
    //Ix, Iy, Iz represents array dimentions in Intermideate space
    //Rx, Ry, Rz represents array dimentions in Real space

    //*_unit represents size of each unit along a direction. 2 for complex, 1 for real

    if (basis[0] == 'F' || basis[0] == 'S') {
        Fx=Nx;
        Ix=Nx;
        Rx=Nx;
    }
    else {
        cerr << "basis[0] can be S or F";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    if (basis[1] == 'F' || basis[1] == 'S') {
        Fy=Ny;
        Iy=Ny;
        Ry=Ny;
    }
    else {
        cerr << "basis[1] can be S or F";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (basis[2] == 'F') {
        Fz=Nz/2+1;
        Iz=Nz/2+1;
        Rz=Nz+2;

        Fz_unit=2;
        Iz_unit=2;
        Rz_unit=1;
    }
    else if (basis[2] == 'S') {
        Fz=Nz/2;
        Iz=Nz/2;
        Rz=Nz;

        Fz_unit=2;
        Iz_unit=2;
        Rz_unit=1;
    }
    else {
        cerr << "basis[] can be S or F";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    set_vars();
    init_transpose();
    init_transform();
    init_timers();
}


/** 
 *  @brief Initialize internal variables
 *
 */
void FFTK::set_vars() {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    num_p_cols=numprocs/num_p_rows;

    MPI_Comm_split(MPI_COMM_WORLD, my_id/num_p_cols, 0, &MPI_COMM_ROW);
    MPI_Comm_split(MPI_COMM_WORLD, my_id%num_p_cols, 0, &MPI_COMM_COL);

    MPI_Comm_rank(MPI_COMM_ROW, &my_col_id);
    MPI_Comm_rank(MPI_COMM_COL, &my_row_id);

    if (Ny>1) {
        maxfx=Fx;
        maxfy=Fy/num_p_cols;
        maxfz=Fz/num_p_rows;

        maxix=Ix/num_p_cols;
        maxiy=Iy;
        maxiz=Iz/num_p_rows;

        maxrx=Rx/num_p_cols;
        maxry=Ry/num_p_rows;
        maxrz=Rz;
    }
    else {
        maxfx=Fx;
        maxfy=1;
        maxfz=Fz/num_p_cols;

        maxix=Ix/num_p_cols;
        maxiy=1;
        maxiz=Iz/num_p_rows;

        maxrx=Rx/num_p_cols;
        maxry=1;
        maxrz=Rz;   
    }


    if (Ny>1) {
        fx_start=0;
        fy_start=maxfy*my_col_id;
        fz_start=maxfz*my_row_id;

        ix_start=maxix*my_col_id;
        iy_start=0;
        iz_start=maxiz*my_row_id;

        rx_start=maxrx*my_col_id;
        ry_start=maxry*my_row_id;
        rz_start=0;
    }
    else {
        fx_start=0;
        fy_start=0;
        fz_start=maxfz*my_col_id;

        ix_start=-1;
        iy_start=-1;
        iz_start=-1;

        rx_start=maxrx*my_col_id;
        ry_start=0;
        rz_start=0; 
    }

    //Set Array size
    FA_shape=shape(maxfx, maxfy, maxfz);
    IA_shape=shape(maxix, maxiy, maxiz);
    RA_shape=shape(maxrx, maxry, maxrz);

    //FA.resize(FA_shape);
    IA.resize(IA_shape);
    RA.resize(RA_shape);

    fft_plan_Fourier_forward_x=NULL;
    fft_plan_Fourier_inverse_x=NULL;
    fft_plan_Fourier_forward_y=NULL;
    fft_plan_Fourier_inverse_y=NULL;
    fft_plan_Fourier_forward_z=NULL;
    fft_plan_Fourier_inverse_z=NULL;

    fft_plan_Sine_forward_x=NULL;
    fft_plan_Sine_inverse_x=NULL;
    fft_plan_Sine_forward_y=NULL;
    fft_plan_Sine_inverse_y=NULL;
    fft_plan_Sine_forward_z=NULL;
    fft_plan_Sine_inverse_z=NULL;

    fft_plan_Cosine_forward_x=NULL;
    fft_plan_Cosine_inverse_x=NULL;
    fft_plan_Cosine_forward_y=NULL;
    fft_plan_Cosine_inverse_y=NULL;
    fft_plan_Cosine_forward_z=NULL;
    fft_plan_Cosine_inverse_z=NULL;

}


/** 
 *  @brief Initialize data transfer related variables
 *
 */
void FFTK::init_transpose()
{
    //To revise
    if (Ny>1)
        buffer_Y = new Real[size_t(Nx/num_p_cols)*size_t(Ny/num_p_rows)*size_t(Nz+2)];

    buffer_Z = new Real[size_t(Nx/num_p_cols)*size_t(Ny/num_p_rows)*size_t(Nz+2)];

    plane_size_YX = Iz_unit*(Iz/num_p_rows)*Iy;
    copy_size_YX  = Iz_unit*(Iz/num_p_rows)*(Iy/num_p_cols);
    block_size_YX = Iz_unit*(Iz/num_p_rows)*(Iy/num_p_cols)*(Ix/num_p_cols);

    plane_size_YZ = Iz_unit*(Iz/num_p_rows)*Iy;
    copy_size_YZ  = Iz_unit*(Iz/num_p_rows)*(Iy/num_p_rows);
    block_size_YZ = Iz_unit*(Iz/num_p_rows)*(Iy/num_p_rows)*(Ix/num_p_cols);
}

/** 
 *  @brief Initialize transform related variables
 *
 */
void FFTK::init_transform(){
    //Initialize plans
    int Nx_dims[]={Nx};
    int Ny_dims[]={Ny};
    int Nz_dims[]={Nz};

    fftw_r2r_kind kind[1];

    normalizing_factor=1;


#ifdef FFTK_THREADS
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif

    //X transforms
    if ( Nx>1 ) {
        if (basis[0] == 'F') {
            fft_plan_Fourier_forward_x = FFTW_PLAN_MANY_DFT(1, Nx_dims, maxfy*maxfz,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxfy*maxfz, 1,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxfy*maxfz, 1,
                FFTW_FORWARD, FFTW_PLAN_FLAG);

            fft_plan_Fourier_inverse_x = FFTW_PLAN_MANY_DFT(1, Nx_dims, maxfy*maxfz,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxfy*maxfz, 1,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxfy*maxfz, 1,
                FFTW_BACKWARD, FFTW_PLAN_FLAG);

            normalizing_factor/=Nx;

            if (fft_plan_Fourier_forward_x == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Fourier_forward_x'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Fourier_inverse_x == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Fourier_inverse_x'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (basis[0] == 'S') {
            kind[0]=FFTW_RODFT10;
            fft_plan_Sine_forward_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, Fz_unit*maxfz*maxfy,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_RODFT01;
            fft_plan_Sine_inverse_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, Fz_unit*maxfz*maxfy,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_REDFT10;
            fft_plan_Cosine_forward_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, Fz_unit*maxfz*maxfy,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_REDFT01;
            fft_plan_Cosine_inverse_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, Fz_unit*maxfz*maxfy,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Fz_unit*maxfz*maxfy, 1,
                kind, FFTW_PLAN_FLAG);

            normalizing_factor/=2*Nx;

            if (fft_plan_Sine_forward_x == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Sine_forward_x'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Sine_inverse_x == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Sine_inverse_x'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Cosine_forward_x == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_forward_x'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Cosine_inverse_x == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_inverse_x'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            if (my_id==0) cerr << "FFTK::init_transform() - invalid basis" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }



    //Y transforms
    if ( Ny>1 ) {
        if (basis[1] == 'F') {
            fft_plan_Fourier_forward_y = FFTW_PLAN_MANY_DFT(1, Ny_dims, maxiz,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxiz, 1,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxiz, 1,
                FFTW_FORWARD, FFTW_PLAN_FLAG);

            fft_plan_Fourier_inverse_y = FFTW_PLAN_MANY_DFT(1, Ny_dims, maxiz,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxiz, 1,
                reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                maxiz, 1,
                FFTW_BACKWARD, FFTW_PLAN_FLAG);

            normalizing_factor/=Ny;

            if (fft_plan_Fourier_forward_y == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Fourier_forward_y'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Fourier_inverse_y == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Fourier_inverse_y'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (basis[1] == 'S') {
            kind[0]=FFTW_RODFT10;
            fft_plan_Sine_forward_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, Iz_unit*maxiz,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_RODFT01;
            fft_plan_Sine_inverse_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, Iz_unit*maxiz,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_REDFT10;
            fft_plan_Cosine_forward_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, Iz_unit*maxiz,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_REDFT01;
            fft_plan_Cosine_inverse_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, Iz_unit*maxiz,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                reinterpret_cast<Real*>(IA.data()), NULL,
                Iz_unit*maxiz, 1,
                kind, FFTW_PLAN_FLAG);

            normalizing_factor/=2*Ny;

            if (fft_plan_Sine_forward_y == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Sine_forward_y'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Sine_inverse_y == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Sine_inverse_y'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Cosine_forward_y == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_forward_y'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Cosine_inverse_y == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_inverse_y'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            if (my_id==0) cerr << "FFTK::init_transform() - invalid basis" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //Z transforms
    if ( Nz>1) {
        if (basis[2] == 'F') {
            fft_plan_Fourier_forward_z = FFTW_PLAN_MANY_DFT_R2C(1, Nz_dims, maxrx*maxry,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz+2,
                reinterpret_cast<FFTW_Complex*>(RA.data()), NULL,
                1, Nz/2+1,
                FFTW_PLAN_FLAG);

            fft_plan_Fourier_inverse_z = FFTW_PLAN_MANY_DFT_C2R(1, Nz_dims, maxrx*maxry,
                reinterpret_cast<FFTW_Complex*>(RA.data()), NULL,
                1, Nz/2+1,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz+2,
                FFTW_PLAN_FLAG);

            normalizing_factor/=Nz;

            if (fft_plan_Fourier_forward_z == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Fourier_forward_z'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Fourier_inverse_z == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Fourier_inverse_z'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (basis[2] == 'S') {
            kind[0]=FFTW_RODFT10;
            fft_plan_Sine_forward_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, maxrx*maxry,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_RODFT01;
            fft_plan_Sine_inverse_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, maxrx*maxry,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_REDFT10;
            fft_plan_Cosine_forward_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, maxrx*maxry,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                kind, FFTW_PLAN_FLAG);

            kind[0]=FFTW_REDFT01;
            fft_plan_Cosine_inverse_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, maxrx*maxry,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                reinterpret_cast<Real*>(RA.data()), NULL,
                1, Nz,
                kind, FFTW_PLAN_FLAG);

            normalizing_factor/=2*Nz;

            if (fft_plan_Sine_forward_z == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Sine_forward_z'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Sine_inverse_z == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Sine_inverse_z'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Cosine_forward_z == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_forward_z'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (fft_plan_Cosine_inverse_z == NULL){
                if (my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_inverse_z'" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            if (my_id==0) cerr << "FFTK::init_transform() - invalid basis" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}


/** 
 *  @brief Initialize timers
 *
 */
void FFTK::init_timers(){
    timer = new double[22];
    for (int i=0; i<22; i++)
        timer[i]=0;
}