rm -rf build
mkdir build
cd ./build 
cmake .. -DCMAKE_INSTALL_PREFIX=~/DREAMPlace/D_install -DPYTHON_EXECUTABLE=$(which python)
make -j 32
make install
cd ..
cd D_install
