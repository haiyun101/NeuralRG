#!/bin/bash
SRC="./opt/32Ising_T2.5_weightTying"
DST="./opt/32Ising_T2.5_weightTying_continue"

echo "Source: $SRC"
echo "Target: $DST"

# 创建目录并复制
mkdir -p $DST
cp -r $SRC/savings $DST/
cp $SRC/parameters.hdf5 $DST/

# 使用 Python 修改参数 (修复了引号语法)
python3 -c "
import h5py
dst = '$DST/parameters.hdf5'

##########
Tc = 2.5 #
##########
try:
    with h5py.File(dst, 'r+') as f:
        if 'T' in f:
            old_T = f['T'][()]
            print('Original T: {}'.format(old_T))
            del f['T']
        f.create_dataset('T', data=Tc)
        print('New T set to: {}'.format(Tc))
except Exception as e:
    print('Error: {}'.format(e))
"
