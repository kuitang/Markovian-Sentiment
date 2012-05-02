#!/bin/sh
if [ "x$1" == "x" ]; then
  echo "Usage: $0 www.blog1.tumblr.com NPOSTS" > /dev/stderr
  exit 1
fi

mkdir "data/html/$1"
pushd "data/html/$1"
curl -o 1 "$1" 
curl -O "$1/page/[2-$2]"
popd

