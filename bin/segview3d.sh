#!/usr/bin/env bash

# doc
echo ""
echo "----segview3d----"
echo "Given a list of state files, view 3d one by one."
echo "Usage: segview3d.sh *.mrc"
echo ""
# iterate over tomos
for f in $@
do
    read -p "segview.py 3d ${f}? yes(y,default)/skip(s)/exit(e): " open
    case $open in
	s* ) continue ;;
	e* ) break ;;
	* ) segview.py 3d $f ;;
    esac
done 

