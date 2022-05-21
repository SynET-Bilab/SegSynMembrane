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
    echo ""
    echo "--${fmrc}--"

    # default mod filename
    fmod="${fmrc%.*}.mod"
    echo "Default model name: ${fmod}"
    
    # prompt for existence
    if [[ -f $fmod ]]
    then
        echo "The model file exists."
    fi
    
    # draw
    read -p "Draw model? yes(y,default)/skip(s)/exit(e): " open_mod
    case $open_mod in
	s* ) 
        echo "Skipping."
        continue
        ;;
	e* )
        echo "Exiting." 
        break
        ;;
	* )
        # copy/paste an empty model file if it does not exist
        if [[ ! -f $fmod ]]
        then
            cp $mod_empty $fmod
        fi
        echo "3dmod $fmrc $fmod"
	    3dmod $fmrc $fmod

        # whether to delete the model file
        read -p "successful? yes(y,default)/no(n,delete model): " succ_mod
        case $succ_mod in
        n* ) 
            echo "rm $fmod"
            rm $fmod ;;
        * ) ;;
        esac
    esac
done 

