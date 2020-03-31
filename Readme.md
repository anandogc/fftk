# User Maual of FFTK

### Typical Compilation:

    cd fftk/trunk
    CXX=mpicxx cmake . -DCMAKE_INSTALL_PREFIX=$HOME/local
    make install

### CMake supports the following options

    -DREAL=FLOAT           - It will treats Real as FLOAT
                             Default: DOUBLE
    
    -DCMAKE_INSTALL_PREFIX - Installation Folder
    
    -DFIND_LIBRARIES=OFF   - It will not try fo find Blitz++ and FFTW library
                             Default: On
    
    -DENABLE_THREAD=TRUE   - Enable OpenMP threads for computing local FFT
                             Default: FALSE
    
    -DFFTW_PLAN=PATIENT    - It can be one of ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE
                             Default: PATIENT


### Using FFTK
    mpirun -np 4 ./test_fftk   basis   sincos_option   Nx   Ny   Nz   iter   [rows]
e.g.
    
    mpirun -np 4 ./test_fftk SSS SCC 32 32 32 10 2

#### Option Description

    basis        - One of FFF, SFF, SSF, SSS.
                   Here F reprasents Fourier Transform, and
                   S represents Sinusoidal Transform
    
    basis_option - It says whether Sicusoidal is Sin or Cos
                   For basis = FFF, it can only be FFF.
                   For basis = SFF, it can be SFF or CFF.
                   For basis = SSF, it can be SSF, SCF, CSF, CCF.
                   For basis = SSS, it can be SSS, SSC, SCS, SCC, ... .
    	
    Nx, Ny, Nz   - The grid size. For 2D use Ny = 1
    
    iter          - Number of pair of transforms to perform
    
    rows          - Number of Rows in a pencil division.
                    Default:
                      rows = 1 if numprocs < Nx
                      rows = numprocs / Nx is numprocs>Nx
             
For pencil division this assertion must hold:

    (Nx%cols == 0) and (Ny%cols == 0) and (Ny%rows == 0) and (Nz%rows == 0)

where cols = numprocs/rows.


License
----
BSD-3
