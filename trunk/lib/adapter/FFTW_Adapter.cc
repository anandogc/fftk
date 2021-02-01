#include "FFTW_Adapter.h"
#include "fftk.h"

namespace FFTK {

    FFTW_Adapter::FFTW_Adapter() {}
    FFTW_Adapter::FFTW_Adapter(shared_ptr<Global> g): global(g) {}


    //initialize FFTW plans 
    void FFTW_Adapter::init_transform(string basis) {
        Global::Field &field = global->field;
        Global::MPI &mpi = global->mpi;


        //Initialize plans
        int Nx_dims[] = {field.Nx};
        int Ny_dims[] = {field.Ny};
        int Nz_dims[] = {field.Nz};

        fftw_r2r_kind kind[1];


        FA.resize(field.FA_shape);
        IA.resize(field.IA_shape);
        RA.resize(field.RA_shape);


    #ifdef FFTK_THREADS
        fftw_init_threads();
        fftw_plan_with_nthreads(omp_get_max_threads());
    #endif

        //X transforms
        if ( field.Nx>1 ) {
            if (basis[0] == 'F'){
                 fft_plan_Fourier_forward_x = FFTW_PLAN_MANY_DFT(1, Nx_dims, field.maxfy * field.maxfz,
                    reinterpret_cast<FFTW_Complex*>(FA.data()), NULL,
                    field.maxfy * field.maxfz, 1,
                    reinterpret_cast<FFTW_Complex*>(FA.data()), NULL,
                    field.maxfy * field.maxfz, 1,
                    FFTW_FORWARD, FFTW_PLAN_FLAG);

                fft_plan_Fourier_inverse_x = FFTW_PLAN_MANY_DFT(1, Nx_dims, field.maxfy * field.maxfz,
                    reinterpret_cast<FFTW_Complex*>(FA.data()), NULL,
                    field.maxfy * field.maxfz, 1,
                    reinterpret_cast<FFTW_Complex*>(FA.data()), NULL,
                    field.maxfy * field.maxfz, 1,
                    FFTW_BACKWARD, FFTW_PLAN_FLAG);

                if (fft_plan_Fourier_forward_x == NULL){
                    if (global->mpi.my_id==0) cerr << "FFF basis failed to initialize 'fft_plan_Fourier_forward_x'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Fourier_inverse_x == NULL){
                    if (global->mpi.my_id==0) cerr << "FFF basis failed to initialize 'fft_plan_Fourier_inverse_x'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            else if (basis[0] == 'S') {
                 kind[0]=FFTW_RODFT10;
                 fft_plan_Sine_forward_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, field.Fz_unit*field.maxfz*field.maxfy,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    kind, FFTW_PLAN_FLAG);

                 kind[0]=FFTW_RODFT01;
                 fft_plan_Sine_inverse_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, field.Fz_unit*field.maxfz*field.maxfy,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    kind, FFTW_PLAN_FLAG);

                 kind[0]=FFTW_REDFT10;
                 fft_plan_Cosine_forward_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, field.Fz_unit*field.maxfz*field.maxfy,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    kind, FFTW_PLAN_FLAG);

                 kind[0]=FFTW_REDFT01;
                 fft_plan_Cosine_inverse_x = FFTW_PLAN_MANY_R2R(1, Nx_dims, field.Fz_unit*field.maxfz*field.maxfy,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Fz_unit*field.maxfz*field.maxfy, 1,
                    kind, FFTW_PLAN_FLAG);

         
                if (fft_plan_Sine_forward_x == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Sine_forward_x'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Sine_inverse_x == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Sine_inverse_x'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Cosine_forward_x == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_forward_x'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Cosine_inverse_x == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_inverse_x'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            else {
                if (global->mpi.my_id==0) cerr << "FFTK::init_transform() - invalid basis" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        //Y transforms
        if ( field.Ny>1 ) {
            if (basis[1] == 'F') {
                fft_plan_Fourier_forward_y = FFTW_PLAN_MANY_DFT(1, Ny_dims, field.maxiz,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    field.maxiz, 1,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    field.maxiz, 1,
                    FFTW_FORWARD, FFTW_PLAN_FLAG);

                fft_plan_Fourier_inverse_y = FFTW_PLAN_MANY_DFT(1, Ny_dims, field.maxiz,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    field.maxiz, 1,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    field.maxiz, 1,
                    FFTW_BACKWARD, FFTW_PLAN_FLAG);


                if (fft_plan_Fourier_forward_y == NULL){
                    if (global->mpi.my_id==0) cerr << "FFF basis failed to initialize 'fft_plan_Fourier_forward_y'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Fourier_inverse_y == NULL){
                    if (global->mpi.my_id==0) cerr << "FFF basis failed to initialize 'fft_plan_Fourier_inverse_y'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            else if (basis[1] == 'S') {
                kind[0]=FFTW_RODFT10;
                fft_plan_Sine_forward_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, field.Iz_unit*field.maxiz,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    kind, FFTW_PLAN_FLAG);

                kind[0]=FFTW_RODFT01;
                fft_plan_Sine_inverse_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, field.Iz_unit*field.maxiz,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    kind, FFTW_PLAN_FLAG);

                kind[0]=FFTW_REDFT10;
                fft_plan_Cosine_forward_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, field.Iz_unit*field.maxiz,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    kind, FFTW_PLAN_FLAG);

                kind[0]=FFTW_REDFT01;
                    fft_plan_Cosine_inverse_y = FFTW_PLAN_MANY_R2R(1, Ny_dims, field.Iz_unit*field.maxiz,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    reinterpret_cast<Real*>(IA.data()), NULL,
                    field.Iz_unit*field.maxiz, 1,
                    kind, FFTW_PLAN_FLAG);

            
                if (fft_plan_Sine_forward_y == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Sine_forward_y'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Sine_inverse_y == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Sine_inverse_y'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Cosine_forward_y == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_forward_y'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Cosine_inverse_y == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_inverse_y'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            else {
                if (global->mpi.my_id==0) cerr << "FFTK::init_transform() - invalid basis" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Z transforms
        if ( field.Nz>1) {
            if (basis[2] == 'F') {
                //for debug purpose:
                //cout << "init FFF adapter: " << field.Nz << " " << field.maxrx << " " << field.maxry << " " << field.Nz << endl;
                

                fft_plan_Fourier_forward_z = FFTW_PLAN_MANY_DFT_R2C(1, Nz_dims, field.maxrx * field.maxry,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz+2,
                    reinterpret_cast<FFTW_Complex*>(RA.data()), NULL,
                    1, field.Nz/2+1,
                    FFTW_PLAN_FLAG);

                fft_plan_Fourier_inverse_z = FFTW_PLAN_MANY_DFT_C2R(1, Nz_dims, field.maxrx * field.maxry,
                    reinterpret_cast<FFTW_Complex*>(RA.data()), NULL,
                    1, field.Nz/2+1,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz+2,
                    FFTW_PLAN_FLAG);

                //for debug purpose:
                //cout << "fft_plan_Fourier_forward_z = " << fft_plan_Fourier_forward_z << endl;

                if (fft_plan_Fourier_forward_z == NULL){
                    if (global->mpi.my_id==0) cerr << "FFF basis failed to initialize 'fft_plan_Fourier_forward_z'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Fourier_inverse_z == NULL){
                    if (global->mpi.my_id==0) cerr << "FFF basis failed to initialize 'fft_plan_Fourier_inverse_z'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                //for debug purpose:
                //cout << "fft_plan_Fourier_forward_z = " << fft_plan_Fourier_forward_z << endl;
            }
            else if (basis[2] == 'S') {
                kind[0]=FFTW_RODFT10;
                fft_plan_Sine_forward_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, field.maxrx*field.maxry,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    kind, FFTW_PLAN_FLAG);

                kind[0]=FFTW_RODFT01;
                fft_plan_Sine_inverse_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, field.maxrx*field.maxry,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    kind, FFTW_PLAN_FLAG);

                kind[0]=FFTW_REDFT10;
                fft_plan_Cosine_forward_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, field.maxrx*field.maxry,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    kind, FFTW_PLAN_FLAG);

                kind[0]=FFTW_REDFT01;
                fft_plan_Cosine_inverse_z = FFTW_PLAN_MANY_R2R(1, Nz_dims, field.maxrx*field.maxry,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    reinterpret_cast<Real*>(RA.data()), NULL,
                    1, field.Nz,
                    kind, FFTW_PLAN_FLAG);

            
                if (fft_plan_Sine_forward_z == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Sine_forward_z'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Sine_inverse_z == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Sine_inverse_z'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Cosine_forward_z == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_forward_z'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fft_plan_Cosine_inverse_z == NULL){
                    if (global->mpi.my_id==0) cerr << "failed to initialize 'fft_plan_Cosine_inverse_z'" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            else if (basis[2] == 'Z') {
                fft_plan_Fourier_forward_z = FFTW_PLAN_MANY_DFT(1, Nz_dims, field.maxrx*field.maxry,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    1, field.Nz,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    1, field.Nz,
                    FFTW_FORWARD, FFTW_PLAN_FLAG);

                fft_plan_Fourier_inverse_z = FFTW_PLAN_MANY_DFT(1, Nz_dims, field.maxrx*field.maxry,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    1, field.Nz,
                    reinterpret_cast<FFTW_Complex*>(IA.data()), NULL,
                    1, field.Nz,
                    FFTW_BACKWARD, FFTW_PLAN_FLAG);

            
                if (fft_plan_Fourier_forward_z == NULL){
                    if (global->mpi.my_id==0) cerr << "Failed to initialize 'fft_plan_Fourier_forward_z'" << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                 }
                if (fft_plan_Fourier_inverse_z == NULL){
                    if (global->mpi.my_id==0) cerr << "Failed to initialize 'fft_plan_Fourier_inverse_z'" << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                 }
            }
            else {
                if (global->mpi.my_id==0) cerr << "FFTK::init_transform() - invalid basis" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        FA.resize(shape(0,0,0));
        IA.resize(shape(0,0,0));
        RA.resize(shape(0,0,0));

    }

    //=========================== transform for basis option "F"
    //2d forward transform 
	void FFTW_Adapter::DFT_forward_x(Array<Complex,2> A){
			FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));


		}

	void FFTW_Adapter::DFT_forward_R2C(Array<Real,2> Ar){
            FFTW_EXECUTE_DFT_R2C(fft_plan_Fourier_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<FFTW_Complex*>(Ar.data()));


    }

    //2d inverse transform
    void FFTW_Adapter::DFT_inverse_x(Array<Complex,2> A){
            FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));

        }

    void FFTW_Adapter::DFT_inverse_C2R(Array<Real,2> Ar){
            FFTW_EXECUTE_DFT_C2R(fft_plan_Fourier_inverse_z, reinterpret_cast<FFTW_Complex*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    }
    

    //3d forward transform
    void FFTW_Adapter::DFT_forward_x(Array<Complex,3> A){
            FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));


        }

    void FFTW_Adapter::DFT_forward_R2C(Array<Real,3> Ar){
            //for debug purpose:
            //cout << "Ar shape = " << fft_plan_Fourier_forward_z << " " << Ar.shape() << endl;
            
            FFTW_EXECUTE_DFT_R2C(fft_plan_Fourier_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<FFTW_Complex*>(Ar.data()));


    }

    void FFTW_Adapter::DFT_forward_y(Array<Complex,3> IA){
            for (int ix = 0; ix < global->field.maxix; ix++)
                FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_y, reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()));

        }

    //3d inverse transform

    void FFTW_Adapter::DFT_inverse_x(Array<Complex,3> A){
            FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_x, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));

        }

    void FFTW_Adapter::DFT_inverse_C2R(Array<Real,3> Ar){
            FFTW_EXECUTE_DFT_C2R(fft_plan_Fourier_inverse_z, reinterpret_cast<FFTW_Complex*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
    }

    void FFTW_Adapter::DFT_inverse_y(Array<Complex,3> IA){
        for (int ix = 0; ix < global->field.maxix; ix++)
            FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_y, reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<FFTW_Complex*>(IA(ix,Range::all(),Range::all()).data()));
        }

    //========================

    //=========================== transform for basis option "S"

    //2d forward transform
    void FFTW_Adapter::DFT_forward_R2R(string basis_option, Array<Real,2> Ar){
            if ( (basis_option[2] == 'S') && (fft_plan_Sine_forward_z != NULL) ) {
                FFTW_EXECUTE_R2R(fft_plan_Sine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
                utilities.ArrayShiftRight(Ar, 'Z');
            }
            else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_forward_z != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));

    }

    void FFTW_Adapter::DFT_forward_R2R_x(string basis_option, Array<Complex,2> A){
            if ( (basis_option[0] == 'S') && (fft_plan_Sine_forward_x != NULL) ) {
                FFTW_EXECUTE_R2R(fft_plan_Sine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
                utilities.ArrayShiftRight(A, 'X');
            }
            else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_forward_x != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
 
    }
    //2d inverse transform
    void FFTW_Adapter::DFT_inverse_R2R(string basis_option, Array<Real,2> Ar){
            if ( (basis_option[2] == 'S') && (fft_plan_Sine_inverse_z != NULL) ) {
                utilities.ArrayShiftLeft(Ar, 'Z');
                FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
            }
            else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_inverse_z != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));

    }

    void FFTW_Adapter::DFT_inverse_R2R_x(string basis_option, Array<Complex,2> A){
            if ( (basis_option[0] == 'S') && (fft_plan_Sine_inverse_x != NULL) ) {
                utilities.ArrayShiftLeft(A, 'X');
                FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
            }
            else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_inverse_x != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
 
    }

    //3d forward transform
    void FFTW_Adapter::DFT_forward_R2R(string basis_option, Array<Real,3> Ar){
            if ( (basis_option[2] == 'S') && (fft_plan_Sine_forward_z != NULL) ) {
                FFTW_EXECUTE_R2R(fft_plan_Sine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
                utilities.ArrayShiftRight(Ar, 'Z');
            }
            else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_forward_z != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));

    }

    void FFTW_Adapter::DFT_forward_R2R_x(string basis_option, Array<Complex,3> A){
            if ( (basis_option[0] == 'S') && (fft_plan_Sine_forward_x != NULL) ) {
                FFTW_EXECUTE_R2R(fft_plan_Sine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
                utilities.ArrayShiftRight(A, 'X');
            }
            else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_forward_x != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
 
    }

    void FFTW_Adapter::DFT_forward_R2R_y(string basis_option, Array<Complex,3> IA){
            if ( (basis_option[1] == 'S') && (fft_plan_Sine_forward_y != NULL) ) {
                for (int ix = 0; ix < global->field.maxix; ix++)
                    FFTW_EXECUTE_R2R(fft_plan_Sine_forward_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
                    utilities.ArrayShiftRight(IA, 'Y');
            }
            else if ( (basis_option[1] == 'C') && (fft_plan_Cosine_forward_y != NULL) ){
               for (int ix = 0; ix < global->field.maxix; ix++)
                    FFTW_EXECUTE_R2R(fft_plan_Cosine_forward_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
            }
 
        }
    //3d inverse transform
    void FFTW_Adapter::DFT_inverse_R2R(string basis_option, Array<Real,3> Ar){
            if ( (basis_option[2] == 'S') && (fft_plan_Sine_inverse_z != NULL) ) {
                utilities.ArrayShiftLeft(Ar, 'Z');
                FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));
            }
            else if ( (basis_option[2] == 'C') && (fft_plan_Cosine_inverse_z != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_z, reinterpret_cast<Real*>(Ar.data()), reinterpret_cast<Real*>(Ar.data()));

    }

    void FFTW_Adapter::DFT_inverse_R2R_x(string basis_option, Array<Complex,3> A){
            if ( (basis_option[0] == 'S') && (fft_plan_Sine_inverse_x != NULL) ) {
                utilities.ArrayShiftLeft(A, 'X');
                FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
            }
            else if ( (basis_option[0] == 'C') && (fft_plan_Cosine_inverse_x != NULL) )
                FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_x, reinterpret_cast<Real*>(A.data()), reinterpret_cast<Real*>(A.data()));
 
    }

    void FFTW_Adapter::DFT_inverse_R2R_y(string basis_option, Array<Complex,3> IA){
            if ( (basis_option[1] == 'S') && (fft_plan_Sine_inverse_y != NULL) ) {
                utilities.ArrayShiftLeft(IA, 'Y');
                for (int ix = 0; ix < global->field.maxix; ix++)
                    FFTW_EXECUTE_R2R(fft_plan_Sine_inverse_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
                    
            }
            else if ( (basis_option[1] == 'C') && (fft_plan_Cosine_inverse_y != NULL) ){
               for (int ix = 0; ix < global->field.maxix; ix++)
                    FFTW_EXECUTE_R2R(fft_plan_Cosine_inverse_y, reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()), reinterpret_cast<Real*>(IA(ix,Range::all(),Range::all()).data()));
            }
 
        }

    //=========================== transform for basis option "Z"
    void FFTW_Adapter::DFT_forward_C2C(string basis_option, Array<Complex,2> A){        
        if ( (basis_option[2] == 'Z') && (fft_plan_Fourier_forward_z != NULL) ) 
            FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_z, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    
    }

    void FFTW_Adapter::DFT_inverse_C2C(string basis_option, Array<Complex,2> A){        
        if ( (basis_option[2] == 'Z') && (fft_plan_Fourier_inverse_z != NULL) ) 
            FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_z, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    
    }


    // for 3d
    void FFTW_Adapter::DFT_forward_C2C(string basis_option, Array<Complex,3> A){        
        if ( (basis_option[2] == 'Z') && (fft_plan_Fourier_forward_z != NULL) ) 
            FFTW_EXECUTE_DFT(fft_plan_Fourier_forward_z, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    
    }

    void FFTW_Adapter::DFT_inverse_C2C(string basis_option, Array<Complex,3> A){        
        if ( (basis_option[2] == 'Z') && (fft_plan_Fourier_inverse_z != NULL) ) 
            FFTW_EXECUTE_DFT(fft_plan_Fourier_inverse_z, reinterpret_cast<FFTW_Complex*>(A.data()), reinterpret_cast<FFTW_Complex*>(A.data()));
    
    }
 
 
        

    
    void FFTW_Adapter::Finalize(){

        if (global->field.basis_name[0] == 'F'){
            FFTW_DESTROY_PLAN(fft_plan_Fourier_forward_x);
            FFTW_DESTROY_PLAN(fft_plan_Fourier_inverse_x);
        }
        else if (global->field.basis_name[0] == 'S'){    
            FFTW_DESTROY_PLAN(fft_plan_Sine_forward_x);
            FFTW_DESTROY_PLAN(fft_plan_Sine_inverse_x);
            
            FFTW_DESTROY_PLAN(fft_plan_Cosine_forward_x);
            FFTW_DESTROY_PLAN(fft_plan_Cosine_inverse_x);            
        }

        if (global->field.basis_name[1] == 'F'){
            FFTW_DESTROY_PLAN(fft_plan_Fourier_forward_y);
            FFTW_DESTROY_PLAN(fft_plan_Fourier_inverse_y);
        }
        else if (global->field.basis_name[1] == 'S'){
            FFTW_DESTROY_PLAN(fft_plan_Sine_forward_y);
            FFTW_DESTROY_PLAN(fft_plan_Sine_inverse_y);
            
            FFTW_DESTROY_PLAN(fft_plan_Cosine_forward_y);
            FFTW_DESTROY_PLAN(fft_plan_Cosine_inverse_y);
            
        }

        if (global-> field.basis_name[2] == 'F' or global-> field.basis_name[2] == 'Z' ){
            FFTW_DESTROY_PLAN(fft_plan_Fourier_forward_z);
            FFTW_DESTROY_PLAN(fft_plan_Fourier_inverse_z);


        }

        else if (global->field.basis_name[2] == 'S') {
            FFTW_DESTROY_PLAN(fft_plan_Sine_forward_z);
            FFTW_DESTROY_PLAN(fft_plan_Sine_inverse_z);

            FFTW_DESTROY_PLAN(fft_plan_Cosine_forward_z);
            FFTW_DESTROY_PLAN(fft_plan_Cosine_inverse_z);
        }
        

    }
	
}

