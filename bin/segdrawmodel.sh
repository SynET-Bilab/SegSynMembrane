#!/usr/bin/env bash

# doc
echo ""
echo "----segdrawmodel----"
echo "Given a list of tomos, draw model one by one."
echo "Usage: segdrawmodel.sh *.mrc"
echo ""

# get empty model
sh_path=$(which segdrawmodel.sh)
mod_empty=${sh_path%/*}/empty.mod
echo "Checking empty model: ${mod_empty}"
if [[ -f $mod_empty ]]
then
    echo "Empty model found. Continue."
else
    echo "Empty model not found. Exiting."
    exit
fi
echo ""

# iterate over tomos
for fmrc in $@
do
    echo "--${fmrc}--"

    # default mod filename
    fmod="${fmrc%.*}.mod"
    echo "default model name: ${fmod}"

    if [[ -f $fmod ]]
    then
        echo "the model exists. skipping."
        continue
    fi
    
    # draw
    read -p "draw model on ${fmrc}? yes(y,default)/skip(s)/exit(e): " open_mod
    case $open_mod in
	s* ) 
        echo "skipped, reading next"
        continue
        ;;
	e* )
        echo "exit" 
        break
        ;;
	* )
        cp $mod_empty $fmod
        echo "3dmod $fmrc $fmod"
	    3dmod $fmrc $fmod

        read -p "successful? yes(y,default)/no(n,delete model): " succ_mod
        case $succ_mod in
        n* ) 
            echo "rm $fmod"
            rm $fmod ;;
        * ) ;;
        esac
    esac
done 

