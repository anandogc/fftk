#include "communication.h"



namespace FFTK {

    /** 
    *  @brief Initialize data transfer related variables
    *
    */
    Communication::Communication(shared_ptr<Global> g): global(g), utilities(g) {}

    void Communication::Init() {

        
        Global::Field &field = global->field;
        Global::MPI &mpi = global->mpi;

        MPI_Comm_rank(MPI_COMM_WORLD, &mpi.my_id);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi.numprocs);
        
        mpi.num_p_cols = mpi.numprocs/mpi.num_p_rows;

        if (field.Ny > 1) {
          assert( (field.Nx % mpi.num_p_cols == 0) and (field.Ny % mpi.num_p_cols == 0) and 
                  (field.Ny % mpi.num_p_rows == 0) and (field.Nz % mpi.num_p_rows == 0) );
        }
        if (field.Ny == 1) {
          if (mpi.num_p_rows > 1 or mpi.num_p_cols > field.Nx or mpi.num_p_cols > field.Nz) {
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
        }

        MPI_Comm_split(MPI_COMM_WORLD, mpi.my_id/mpi.num_p_cols, 0, &global->mpi.MPI_COMM_ROW);
        MPI_Comm_split(MPI_COMM_WORLD, mpi.my_id%mpi.num_p_cols, 0, &global->mpi.MPI_COMM_COL);

        MPI_Comm_rank(global->mpi.MPI_COMM_ROW, &mpi.my_col_id);
        MPI_Comm_rank(global->mpi.MPI_COMM_COL, &mpi.my_row_id);

        if (field.Ny>1) {
            field.maxfx = field.Fx;
            field.maxfy = field.Fy/mpi.num_p_cols;
            field.maxfz = field.Fz/mpi.num_p_rows;

            field.maxix = field.Ix/mpi.num_p_cols;
            field.maxiy = field.Iy;
            field.maxiz = field.Iz/mpi.num_p_rows;

            field.maxrx = field.Rx/mpi.num_p_cols;
            field.maxry = field.Ry/mpi.num_p_rows;
            field.maxrz = field.Rz;
        }
        else {
            field.maxfx = field.Fx;
            field.maxfy = 1;
            field.maxfz = field.Fz/mpi.num_p_cols;

            field.maxix = field.Ix/mpi.num_p_cols;
            field.maxiy = 1;
            field.maxiz = field.Iz/mpi.num_p_rows;

            field.maxrx = field.Rx/mpi.num_p_cols;
            field.maxry = 1;
            field.maxrz = field.Rz;   
        }


        if (field.Ny>1) {
            field.fx_start = 0;
            field.fy_start = field.maxfy * mpi.my_col_id;
            field.fz_start = field.maxfz * mpi.my_row_id;

            field.ix_start = field.maxix * mpi.my_col_id;
            field.iy_start = 0;
            field.iz_start = field.maxiz * mpi.my_row_id;

            field.rx_start = field.maxrx * mpi.my_col_id;
            field.ry_start = field.maxry * mpi.my_row_id;
            field.rz_start = 0;

        }
        else {
            field.fx_start = 0;
            field.fy_start = 0;
            field.fz_start = field.maxfz * mpi.my_col_id;

            field.ix_start = -1;
            field.iy_start = -1;
            field.iz_start = -1;

            field.rx_start = field.maxrx * mpi.my_col_id;
            field.ry_start = 0;
            field.rz_start = 0; 
        }

        mpi.Rz_comm = utilities.Get_Rz_comm();

        //Set Array size
        field.FA_shape=shape(field.maxfx, field.maxfy, field.maxfz);
        field.IA_shape=shape(field.maxix, field.maxiy, field.maxiz);
        field.RA_shape=shape(field.maxrx, field.maxry, field.maxrz); 

        
        IA.resize(field.IA_shape);

        // Initialize data transfer related variables

        if (field.Ny > 1) {
         // MPI_Type_vector(count                    , blocklength                        , stride          , oldtype , *newtype                      );
            MPI_Type_vector(1                        , field.Fz_unit*field.maxfz*field.maxfy*(field.Fx/mpi.num_p_cols), 1               , MPI_Real, &X_axes_selector_vector       );
            MPI_Type_vector(field.maxix              , field.Iz_unit*field.maxiz*(field.Iy/mpi.num_p_cols)      , field.Iz_unit*field.maxiz*field.Iy, MPI_Real, &Y_axes_selector_for_XY_vector);
            MPI_Type_vector(field.maxix              , field.Iz_unit*field.maxiz*(field.Iy/mpi.num_p_rows)      , field.Iz_unit*field.maxiz*field.Iy, MPI_Real, &Y_axes_selector_for_YZ_vector);
            MPI_Type_vector(field.maxry*field.maxrx  , (field.Rz_unit*mpi.Rz_comm)/mpi.num_p_rows       , field.Rz_unit*field.Rz      , MPI_Real, &Z_axes_selector_vector       );

            MPI_Type_commit(&X_axes_selector_vector);
            MPI_Type_commit(&Y_axes_selector_for_XY_vector);
            MPI_Type_commit(&Y_axes_selector_for_YZ_vector);
            MPI_Type_commit(&Z_axes_selector_vector);


          //lb -> lower_bound (in bytes)

          //MPI_Type_create_resized(oldtype                      ,lb, extent (in bytes)                                     , *newtype);
            MPI_Type_create_resized(X_axes_selector_vector       , 0, sizeof(Real)*field.Fz_unit*field.maxfz*field.maxfy*(field.Fx/mpi.num_p_cols), &X_axes_selector);
            MPI_Type_create_resized(Y_axes_selector_for_XY_vector, 0, sizeof(Real)*field.Iz_unit*field.maxiz*(field.Iy/mpi.num_p_cols)      , &Y_axes_selector_for_XY);
            MPI_Type_create_resized(Y_axes_selector_for_YZ_vector, 0, sizeof(Real)*field.Iz_unit*field.maxiz*(field.Iy/mpi.num_p_rows)      , &Y_axes_selector_for_YZ);
            MPI_Type_create_resized(Z_axes_selector_vector       , 0, sizeof(Real)*(field.Rz_unit*mpi.Rz_comm)/mpi.num_p_rows       , &Z_axes_selector);

            MPI_Type_commit(&X_axes_selector);
            MPI_Type_commit(&Y_axes_selector_for_XY);
            MPI_Type_commit(&Y_axes_selector_for_YZ);
            MPI_Type_commit(&Z_axes_selector);
        }
        else {
            // for debug purpose:
            //cout << "params:  " << mpi.my_id << " " << field.Fz_unit<< " " << field.maxfz<< " " << field.Fx<< " " << mpi.num_p_cols<< " " << field.Rz_unit<< " " << mpi.Rz_comm << endl;
         

         // MPI_Type_vector(count        , blocklength                                  , stride           , oldtype , *newtype                      );
            MPI_Type_vector(1            , field.Fz_unit*field.maxfz*(field.Fx/mpi.num_p_cols), 1                , MPI_Real, &X_axes_selector_vector       );
            MPI_Type_vector(field.maxrx  , (field.Rz_unit*mpi.Rz_comm)/mpi.num_p_cols             , field.Rz_unit*field.Rz , MPI_Real, &Z_axes_selector_vector       );

            MPI_Type_commit(&X_axes_selector_vector);
            MPI_Type_commit(&Z_axes_selector_vector);

          //lb -> lower_bound (in bytes)
          //MPI_Type_create_resized(oldtype                      ,lb, extent (in bytes)                                     , *newtype);
            MPI_Type_create_resized(X_axes_selector_vector       , 0, sizeof(Real)*field.Fz_unit*field.maxfz*(field.Fx/mpi.num_p_cols), &X_axes_selector);
            MPI_Type_create_resized(Z_axes_selector_vector       , 0, sizeof(Real)*(field.Rz_unit*mpi.Rz_comm)/mpi.num_p_cols , &Z_axes_selector);

            MPI_Type_commit(&X_axes_selector);
            MPI_Type_commit(&Z_axes_selector);
        }
    }


    /** @brief Transpose a 2d Fourier space array into 2d real array
     *  @param FA - The Fourier space array
     *  @param RA - The real space array
     *
     */
    void Communication::Transpose(Array<Complex,2> FA, Array<Real,2> RA)
    {
        if (global->mpi.num_p_cols>1) {
            MPI_Alltoall(FA.data(), 1, X_axes_selector, RA.data(), 1, Z_axes_selector, global->mpi.MPI_COMM_ROW);
        }
        else
            RA=Array<Real,2>(reinterpret_cast<Real*>(FA.data()), RA.shape(), neverDeleteData);


        if (global->mpi.Rz_comm < global->field.Rz) {
            TIMER_START(18);
            utilities.Zero_pad_last_plane(RA);
            TIMER_END(18);
        }
    }

    /** @brief Transpose a 2d Fourier space array into 2d real space array
     *  @param RA - The real space array
     *  @param FA - The Fourier space array
     *
     */
    void Communication::Transpose(Array<Real,2> RA, Array<Complex,2> FA)
    {
        if (global->mpi.num_p_cols>1) {
            MPI_Alltoall(RA.data(), 1, Z_axes_selector, FA.data(), 1, X_axes_selector, global->mpi.MPI_COMM_ROW);
        }
        else
            FA=Array<Complex,2>(reinterpret_cast<Complex*>(RA.data()), FA.shape(), neverDeleteData);
    }

    //****************

    /** @brief Transpose a 2d Fourier space array into 2d Fourier array
     *  @param A1 - The Fourier space array
     *  @param A2 - The Fourier space array
     *
     */

    void Communication::Transpose_XZ_2d(Array<Complex,2> A1, Array<Complex,2> A2)
    {
        if (global->mpi.num_p_cols>1) {
            MPI_Alltoall(A1.data(), 1, X_axes_selector, A2.data(), 1, Z_axes_selector, global->mpi.MPI_COMM_ROW);
        }
        else
            A2=Array<Complex,2>(reinterpret_cast<Complex*>(A1.data()), A2.shape(), neverDeleteData);

    }

    /** @brief Transpose a 2d Fourier space array into 2d Fourier array
     *  @param A2 - The Fourier space array
     *  @param A1 - The Fourier space array
     *
     */

    void Communication::Transpose_ZX_2d(Array<Complex,2> A2, Array<Complex,2> A1)
    {
        if (global->mpi.num_p_cols>1) {
            MPI_Alltoall(A2.data(), 1, Z_axes_selector, A1.data(), 1, X_axes_selector, global->mpi.MPI_COMM_ROW);
        }
        else
            A1=Array<Complex,2>(reinterpret_cast<Complex*>(A2.data()), A1.shape(), neverDeleteData);
    }

    // ***************

    /** @brief Transpose the processor division of a 3d real space array into 3d Fourier space array
     *  @param RA - The real space array
     *  @param FA - The Fourier space array
     *
     */


    void Communication::Transpose(Array<Real,3> Ar, Array<Complex,3> A)
    {
        Transpose_ZY(Ar,IA);
        Transpose_YX(IA,A);
    }

    /** @brief Transpose the processor division of a 3d Fourier space array into 3d real space array
     *  @param RA - The real space array
     *  @param FA - The Fourier space array
     *
     */
    void Communication::Transpose(Array<Complex,3> A, Array<Real,3> Ar)
    {
        Transpose_XY(A,IA);
        Transpose_YZ(IA,Ar);
    }

    /** @brief Transpose the processor division of a 3d Fourier space array into 3d real space array
     *  @param RA - The real space array
     *  @param FA - The Fourier space array
     */
    void Communication::Transpose_XY(Array<Complex,3> FA, Array<Complex,3> IA)
    {
        TIMER_START(11);
        if (global->mpi.num_p_cols>1)
            MPI_Alltoall(FA.data(), 1, X_axes_selector, IA.data(), 1, Y_axes_selector_for_XY, global->mpi.MPI_COMM_ROW);
        else
            IA=Array<Complex,3>(reinterpret_cast<Complex*>(FA.data()), global->field.IA_shape, neverDeleteData);
        TIMER_END(11);
    }

    /** @brief Transpose the processor division of pencils along Y to pencils along X
     *  @param IA - The intermideate space array
     *  @param FA - The Fourier space array
     */
    void Communication::Transpose_YX(Array<Complex,3> IA, Array<Complex,3> FA)
    {
        TIMER_START(14);
        if (global->mpi.num_p_cols>1)
            MPI_Alltoall(IA.data(), 1, Y_axes_selector_for_XY, FA.data(), 1, X_axes_selector_vector, global->mpi.MPI_COMM_ROW);
        else 
            FA=Array<Complex,3>(reinterpret_cast<Complex*>(IA.data()), global->field.FA_shape, neverDeleteData);
        TIMER_END(14);
    }

    /** @brief Transpose the processor division of pencils along Y to pencils along Z
     * 
     *  @param IA - The intermideate space array
     *  @param RA - The real space array
     */
    void Communication::Transpose_YZ(Array<Complex,3> IA, Array<Real,3> RA)
    {
        TIMER_START(16);

        if (global->mpi.num_p_rows>1)
            MPI_Alltoall(IA.data(), 1, Y_axes_selector_for_YZ, RA.data(), 1, Z_axes_selector, global->mpi.MPI_COMM_COL);
        else 
            RA=Array<Real,3>(reinterpret_cast<Real*>(IA.data()), global->field.RA_shape, neverDeleteData);

        TIMER_END(16);

        if (global->mpi.Rz_comm < global->field.Rz) {
            TIMER_START(18);
            utilities.Zero_pad_last_plane(RA);
            TIMER_END(18);
        }
    }

    /** @brief Transpose the processor division of pencils along Y to pencils along Z
     *
     *  @param RA - The real space array
     *  @param IA - The intermideate space array
     */
    void Communication::Transpose_ZY(Array<Real,3> RA, Array<Complex,3> IA)
    {
        TIMER_START(20);
        if (global->mpi.num_p_rows>1)
            MPI_Alltoall(RA.data(), 1, Z_axes_selector, IA.data(), 1, Y_axes_selector_for_YZ, global->mpi.MPI_COMM_COL);
        else 
            IA=Array<Complex,3>(reinterpret_cast<Complex*>(RA.data()), global->field.FA_shape, neverDeleteData);

        TIMER_END(20);
    }

    /** @brief Transpose the processor division of pencils along Y to pencils along Z
     * 
     *  @param IA - The intermideate space array
     *  @param RA - The real space array
     */
    void Communication::Transpose_YZ(Array<Complex,3> IA, Array<Complex,3> RA)
    {
         TIMER_START(16);

        if (global->mpi.num_p_rows>1)
            MPI_Alltoall(IA.data(), 1, Y_axes_selector_for_YZ, RA.data(), 1, Z_axes_selector, global->mpi.MPI_COMM_COL);
        else 
            RA=Array<Complex,3>(reinterpret_cast<Complex*>(IA.data()), global->field.RA_shape, neverDeleteData);

        TIMER_END(16);
    }

    /** @brief Transpose the processor division of pencils along Y to pencils along Z
     *
     *  @param RA - The real space array
     *  @param IA - The intermideate space array
     */
    void Communication::Transpose_ZY(Array<Complex,3> RA, Array<Complex,3> IA)
    {
         TIMER_START(20);

        if (global->mpi.num_p_rows>1) 
            MPI_Alltoall(RA.data(), 1, Z_axes_selector, IA.data(), 1, Y_axes_selector_for_YZ, global->mpi.MPI_COMM_COL);
        else
            IA=Array<Complex,3>(reinterpret_cast<Complex*>(RA.data()), global->field.FA_shape, neverDeleteData);

        TIMER_END(20);
    }
  
    void Communication::Finalize(){
        MPI_Type_free(&X_axes_selector_vector);  
        MPI_Type_free(&Z_axes_selector_vector);
        MPI_Type_free(&X_axes_selector);
        MPI_Type_free(&Z_axes_selector);

        if (global->field.Ny > 1){
          MPI_Type_free(&Y_axes_selector_for_XY_vector);
          MPI_Type_free(&Y_axes_selector_for_YZ_vector);

          MPI_Type_free(&Y_axes_selector_for_XY);
          MPI_Type_free(&Y_axes_selector_for_YZ);
        }
    }
}