#!/bin/sh
#PBS -l nodes=1:ppn=1,walltime=24:00:00
#PBS -N SIMSruns_LCDM

# I think the 'oe' prints outputs and errors to the j1 (name) file

#PBS -j oe

# The -t switch tells sciama how many split catalogues there are
# this means it iterates through them as the ${PBS_ARRAYID} variable

#PBS -t 1-500
#PBS -V

cd /users/moricex/PAPER2/MG-PICOLA-PUBLIC/paramfiles
echo "Current working directory is `pwd`"
runnum=${PBS_ARRAYID}
runname=20042017_lcdm_500

RUNDIR=/mnt/lustre/moricex/MGPICOLAruns/${runname}/run_${runnum}/
mkdir -p $RUNDIR
TARGET_KEY1="OutputDir"
TARGET_KEY2="Seed"

cp lcdm_parameterfile.txt lcdm_parameterfile.txt.${runnum}
CONFIG_FILE=lcdm_parameterfile.txt.${runnum}

REPLACEMENT_VALUE1=${RUNDIR}
REPLACEMENT_VALUE2=$(python -S -c "import random; print(random.randrange(0,9999999))")
sed -c -i '' -e "s#\($TARGET_KEY1 * *\).*#\1$REPLACEMENT_VALUE1#" -e "s/\($TARGET_KEY2 * *\).*/\1$REPLACEMENT_VALUE2/" $CONFIG_FILE
mv -f $CONFIG_FILE $RUNDIR
chmod 755 $RUNDIR$CONFIG_FILE

while [ "$(head $RUNDIR$CONFIG_FILE | wc -l)" = "0" ]
do
	sleep 1
done

mpirun -np 1 /users/moricex/PAPER2/MG-PICOLA-PUBLIC/MG_PICOLA_FOFR ${RUNDIR}${CONFIG_FILE}

