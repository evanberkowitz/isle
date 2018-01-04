#!/bin/sh

#
# Generate Doxygen documentation.
# Apply that style if possible.
#

if [ -e that_style/doxyfile.conf ]; then
    cat doxyfile.conf that_style/doxyfile.conf > input.conf
else
    cat doxyfile.conf > input.conf
    echo "Doxygen style not found. Make sure git submodule that_style is initialized.\n"
fi
doxygen input.conf
rm -f input.conf
