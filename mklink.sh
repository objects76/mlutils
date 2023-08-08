#!/bin/bash

srcpath=$(realpath "${BASH_SOURCE:-$0}")
srcdir=$(dirname $srcpath)

# echo script: $srcpath
echo srcdir: $srcdir

echo curdir is $(pwd)
ln -sf $srcdir
echo .
ls -l mlutils
