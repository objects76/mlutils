
** Windows **
admin>
	mklink /d mlutils d:\Utils\mlutils

mklink.cmd
	@echo off
	echo %~dp0

	mklink /d utils %~dp0

	pause
	echo on

** linux **

mklink.sh
	#!/bin/bash

	srcpath=$(realpath "${BASH_SOURCE:-$0}")
	srcdir=$(dirname $srcpath)

	# echo script: $srcpath
	echo srcdir: $srcdir

	echo curdir is $(pwd)
	ln -sf $srcdir
	echo .
	ls -l mlutils
