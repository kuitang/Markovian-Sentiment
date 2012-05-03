#!/bin/sh
ROOT=$PWD
cd wordnet-1.6/src/lib
make PLATFORM=$(uname) 
cp libwn* $ROOT
cd $ROOT

