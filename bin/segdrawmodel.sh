#!/usr/bin/env bash

# doc
echo ""
echo "----segdrawmodel----"
echo "Given a list of tomos, draw model one by one."
echo "Usage: segdrawmodel.sh *.mrc"
echo "Requirement: imod version that can create model file if not exists."
echo ""


# file counting
ntot=$#
icurr=0
echo "Tomos (n=$ntot): $@"

# iterate over tomos
for fmrc in $@
do
    # setup
    icurr=$(( $icurr + 1 ))
    fmod="${fmrc/.mrc/.mod}"
    # draw on mod file
    echo -n "($icurr/$ntot) open $fmrc $fmod? yes(y,default)/skip(s): "
    read selection
    case $selection in
    s* ) echo "skipping"; continue ;;
    * ) 3dmod -E v $fmrc $fmod ;;
    esac
done
