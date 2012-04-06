#!/bin/sh
if [ "x$1" == "x" ]; then
  echo "Usage: $0 www.blog1.tumblr.com [www.blog2.tumblr.com ...]" > /dev/stderr
  exit 1
fi

for blog in $@; do
  mkdir "data/html/$blog"
  pushd "data/html/$blog"
  curl -o 1 "$blog" 
  # In general, we have no way to know how many pages a blog has. So just download
  # 100 pages to be safe. TODO: Verify completion on the last page.
  #N=$(sed -n 's|^.*of \([^<]*\)<a href="/page/2">.*$|\1|p' 1)
  #echo Detected $N posts.
  curl -O "$blog/page/[2-100]"
  popd
done

