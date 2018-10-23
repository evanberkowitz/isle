#!/bin/sh

#
# This script installs a recent version of Doxygen from source.
# DO NOT use this manually, the scipt is meant to be run during CI.
#

set -ex

mkdir doxygen
cd doxygen

echo "Downloading doxygen"
wget https://github.com/doxygen/doxygen/archive/Release_1_8_13.tar.gz
tar -xzvf Release_1_8_13.tar.gz
cd doxygen-Release_1_8_13

mkdir build
cd build

echo "Configuring doxygen"
cmake .. -DCMAKE_INSTALL_PREFIX=/usr

echo "Compiling doxygen"
cmake --build .

echo "Installing doxygen"
sudo cmake --build . --target install
