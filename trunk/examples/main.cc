#include <mpi.h>
#include <fftw3.h>

#include <fftk.h>


void test(string basis, string basis_option, int Nx, int Ny, int Nz, int iter, int rows);
Real f(string basis, string basis_option, int rx, int ry, int rz);
void Print_large_Fourier_elements(Array<Complex,3> A, string array_name="A");

int Get_kx(int lx);
int Get_ky(int ly);
int Get_kz(int lz);


int my_id,numprocs, Nx,Ny,Nz, fx_start,fy_start,fz_start;

string basis,basis_option;

int main(int argc, char** argv)
{

  	MPI_Init(&argc, &argv);

  	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    basis=argv[1];
    basis_option=argv[2];

  	Nx=atoi(argv[3]);
  	Ny=atoi(argv[4]);
  	Nz=atoi(argv[5]);

    int iter = atoi(argv[6]);

    int rows = 1;

    if (argc >=8)
    	rows = atoi(argv[7]);
    else if (numprocs > Nx)
    	rows = numprocs / Nx;


    if (my_id==0) {
    	int cols = numprocs/rows;
   		assert( (Nx % cols == 0) and (Ny % cols == 0) and (Ny % rows == 0) and (Nz % rows == 0) );
   	}


  	test(basis, basis_option, Nx, Ny, Nz, iter, rows);

	MPI_Finalize();
	return 0;
}



void test(string basis1, string basis_option1, int Nx, int Ny, int Nz, int iter, int rows)
{
	basis=basis1;
	basis_option=basis_option1;

    if (my_id==0) {
        cout << "grid = " << Nx << " " << Ny << " " << Nz << endl;
        cout << "numprocs = " << numprocs << endl;
        cout << "rows = " << rows << endl;
        cout << "iter = " << iter << endl;
    }



	FFTK fftk;
	fftk.Init(basis,Nx,Ny,Nz,rows);

	Array<Complex,3> A(fftk.Get_FA_shape());
	Array<Real,3> Ar(fftk.Get_RA_shape());
	Array<Real, 3> Ar_init(fftk.Get_RA_shape());

	Ar_init=0;

	int maxrx=Ar.extent(0);
	int maxry=Ar.extent(1);
	int maxrz=Ar.extent(2);


	TinyVector<int,3> FA_start = fftk.Get_FA_start();
	fx_start=FA_start(0);
	fy_start=FA_start(1);
	fz_start=FA_start(2);

	TinyVector<int,3> RA_start = fftk.Get_RA_start();
	int rx_start=RA_start(0);
	int ry_start=RA_start(1);
	int rz_start=RA_start(2);

	for (int rx=0; rx<maxrx; rx++)
		for (int ry=0; ry<maxry; ry++)
			for (int rz=0; rz<maxrz; rz++){
				Ar_init(rx, ry, rz) = f(basis, basis_option, rx_start + rx, ry_start + ry, rz);
				// Ar_init(rx, ry, rz) =  my_id*Ar_init.size() + (rx*Ar_init.extent(1)*Ar_init.extent(2)) + ry*Ar_init.extent(2) + rz;
			}

	//Zero pad
	if (basis[2] != 'S')
		fftk.Zero_pad_last_plane(Ar_init);

	// cout << Ar_init << endl;

/*	if (my_id==1)
		sleep(1);
	cout << Ar_init << endl;	*/	


	// double local_energy = sum(sqr(Ar_init));
	// double total_energy;
	// MPI_Reduce(&local_energy, &total_energy, 1, MPI_Real, MPI_SUM, 0, MPI_COMM_WORLD);

/*	if (my_id == 0)
		cout << "Before transform energy = " << total_energy << endl;*/

    Real start_time;
    Real end_time;

	Ar = Ar_init;

    Real local_time=0, local_time_max;
    MPI_Barrier(MPI_COMM_WORLD);


	time_t start;
	time_t end;


	time (&start);
	if (Ny>1) {
		fftk.Forward_transform(basis_option, Ar, A);
		Print_large_Fourier_elements(A);
    	fftk.Inverse_transform(basis_option, A, Ar);

        for (int i=1; i< iter; i++) {
            fftk.Forward_transform(basis_option, Ar, A);
	    	fftk.Inverse_transform(basis_option, A, Ar);
        }
	}
	else {
	    fftk.Forward_transform(basis_option, Ar(Range::all(),0,Range::all()), A(Range::all(),0,Range::all()));
   		Print_large_Fourier_elements(A);
   		fftk.Inverse_transform(basis_option, A(Range::all(),0,Range::all()), Ar(Range::all(),0,Range::all()));

        for (int i=0; i< iter; i++) {
		    fftk.Forward_transform(basis_option, Ar(Range::all(),0,Range::all()), A(Range::all(),0,Range::all()));
    		fftk.Inverse_transform(basis_option, A(Range::all(),0,Range::all()), Ar(Range::all(),0,Range::all()));
        }
	}	
    time (&end);


	double *timer_avg, *timer_max, *timer_min;
	Real max_error, local_max_error;

	timer_avg = new double[22];
	timer_max = new double[22];
	timer_min = new double[22];


	MPI_Reduce(fftk.timer, timer_avg, 22, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(fftk.timer, timer_max, 22, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(fftk.timer, timer_min, 22, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	local_max_error = max(abs(Ar-Ar_init));

	MPI_Reduce(&local_max_error, &max_error, 1, MPI_Real, MPI_MAX, 0, MPI_COMM_WORLD);

	Real dif = difftime(end, start);
	if (my_id == 0) {
		cout << "Time taken on master process = " << dif << endl;
		cout << "max(abs(Ar-Ar_init)) on all process = " << max_error << endl;
		for (int i=0; i<22; i++) {
            timer_avg[i] *= 1000.0/(numprocs*iter);
            timer_max[i] *= 1000.0/iter;
            timer_min[i] *= 1000.0/iter;
		}

        double T_comp = timer_avg[0] + timer_avg[2] + timer_avg[4] + timer_avg[6] + timer_avg[8] + timer_avg[10];
        double T_comm = timer_avg[1] + timer_avg[3] + timer_avg[7] + timer_avg[9];

        cout << "Averaged time per pair of transform over all processors (ms): \n";
        cout << "Total Time: " << T_comp + T_comm << endl;
        cout << "Computation time: " << T_comp << endl;
        cout << "Communication time: " << T_comm << endl;
        Real N3 = Real(Nx)*Real(Ny)*Real(Nz);
        cout << "GFlop: " << 10*N3*log(N3)*iter/(log(2)*(T_comp + T_comm)*numprocs*1e9) << endl;
		for (int i=0; i<22; i++)
			printf("timer[%02d] (avg/max/min): %E %E %E \n", i, timer_avg[i], timer_max[i], timer_min[i]);

	}
}

void Print_large_Fourier_elements(Array<Complex,3> A, string array_name)
{

	for (int p=0; p<numprocs; p++) {
		if (my_id==p) {
			for (int lx=0; lx<A.extent(0); lx++)
				for (int ly=0; ly<A.extent(1); ly++)
					for (int lz=0; lz<A.extent(2); lz++)
						if (abs(A(lx, ly, lz)) > 1E-6) {
							cout << "my_id = " << my_id <<  " vect(k) = (" << Get_kx(lx) << "," << Get_ky(ly) << "," << Get_kz(lz) <<");  " << array_name << "(k) = " << A(lx, ly, lz) << '\n';
						}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

Real f(string basis, string basis_option, int rx, int ry, int rz)
{
	Real k0 = 1;
	Real x,y,z;
	
	if (basis == "FFF") {
		Real Lx=2*M_PI;
    	Real Ly=2*M_PI;
    	Real Lz=2*M_PI;

    	x = rx*Lx/Nx;
		y = ry*Ly/Ny;
		z = rz*Lz/Nz;
		if (Ny>1)
			return 8*sin(k0*x)*sin(k0*y)*sin(k0*z);
		else
			return 8*sin(k0*x)*sin(k0*z);
	}
	
	else if (basis == "SFF") {
		Real Lx=M_PI;
    	Real Ly=2*M_PI;
    	Real Lz=2*M_PI;

    	x = (rx+0.5)*Lx/Nx;
   		y = ry*Ly/Ny;
		z = rz*Lz/Nz;

		if (basis_option=="SFF") {
			if (Ny>1)
				return 8*sin(k0*x)*cos(k0*y)*cos(k0*z);
			else
				return 8*sin(k0*x)*cos(k0*z);
		}
		else if (basis_option=="CFF") {
			if (Ny>1)
				return 8*cos(k0*x)*cos(k0*y)*cos(k0*z);
			else
				return 8*cos(k0*x)*cos(k0*z);
		}
	}
	
	else if (basis == "SSF") {
		Real Lx=M_PI;
    	Real Ly=M_PI;
    	Real Lz=2*M_PI;

    	x = (rx+0.5)*Lx/Nx;
		y = (ry+0.5)*Ly/Ny;
		z = rz*Lz/Nz;

		if (basis_option=="SSF") {
			if (Ny>1)
				return 8*sin(k0*x)*sin(k0*y)*cos(k0*z);
			else
				return 8*sin(k0*x)*cos(k0*z);
		}
		else if (basis_option=="CCF") {
			if (Ny>1)
				return 8*cos(k0*x)*cos(k0*y)*cos(k0*z);
			else
				return 8*cos(k0*x)*cos(k0*z);
		}
	}
	
	else if (basis == "SSS") {
		Real Lx=M_PI;
    	Real Ly=M_PI;
    	Real Lz=M_PI;

    	x = (rx+0.5)*Lx/Nx;
		y = (ry+0.5)*Ly/Ny;
		z = (rz+0.5)*Lz/Nz;

		if (basis_option=="SSS") {
			if (Ny>1)
				return 8*sin(k0*x)*sin(k0*y)*sin(k0*z);
			else
				return 8*sin(k0*x)*sin(k0*z);
		}
		else if (basis_option=="CCC") {
			if (Ny>1)
				return 8*cos(k0*x)*cos(k0*y)*cos(k0*z);
			else
				return 8*cos(k0*x)*cos(k0*z);
		}
	}
}

int Get_kx(int lx) {
	if (basis[0]=='F') {
		if (lx+fx_start>Nx/2)
			return (lx+fx_start-Nx);
		else
			return (lx+fx_start);
	}
	else if (basis[0]=='S') {
			return (lx+fx_start);
	}
	return 0;
}

int Get_ky(int ly) {
	if (basis[1]=='F') {
		if (ly+fy_start>Ny/2)
			return (ly+fy_start-Ny);
		else
			return (ly+fy_start);
	}
	else if (basis[0]=='S') {
			return (ly+fy_start);
	}

	return 0;
}

int Get_kz(int lz) {
	if (basis[2]=='F') {
		if (lz+fz_start>Nz/2)
			return (lz+fz_start-Nz);
		else
			return (lz+fz_start);
	}
	else if (basis[0]=='S') {
			return (lz+fz_start);
	}

	return 0;
}