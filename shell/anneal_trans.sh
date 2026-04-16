#!/bin/bash

# 1. Accept the 3 arguments passed by auto_anneal.sh
SRC=$1
DST=$2
NEW_TEMP=$3

echo "Source: $SRC"
echo "Target: $DST"
echo "Target Temp: $NEW_TEMP"

# 2. Create the new directory
mkdir -p $DST

# 3. Copy the weights (savings) and the blueprint (parameters)
cp -r $SRC/savings $DST/
cp $SRC/parameters.hdf5 $DST/

# 4. Use Python to update the Temperature in the new parameters.hdf5
python3 -c "
import h5py

dst = '$DST/parameters.hdf5'
Tc = float($NEW_TEMP)

try:
    with h5py.File(dst, 'r+') as f:
        if 'T' in f:
            old_T = f['T'][()]
            print('Original T: {}'.format(old_T))
            del f['T']
        f.create_dataset('T', data=Tc)
        print('New T successfully set to: {}'.format(Tc))
except Exception as e:
    print('Error updating HDF5: {}'.format(e))
"