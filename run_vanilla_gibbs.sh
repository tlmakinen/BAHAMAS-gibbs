#PBS -N gibbs_jla
#PBS -lselect=1:ncpus=16:mem=16gb:mpiprocs=16:ompthreads=1
#PBS -lwalltime=4:0:0
#PBS -J 1-2

module load anaconda3/personal
source activate bahamas
module load mpi

# start from working directory--label it something useful

cp $PBS_O_WORKDIR/* .
cp -r ~/repositories/BAHAMAS_gibbs/* .
# Generate new set of supernovae
cd data/
cp ~/repositories/BAHAMAS_gibbs/data/* .
python sn1a_generator.py 0.00 0.00 lc_params.txt sim_statssys.txt kde
# BAHAMAS_gibbs Main
cd ..
# run gibbs sampler
nohup mpiexec python run_gibbs_jla.py $PBS_ARRAY_INDEX 10 5
# send email when done
# python email_me.py $PBS_JOBID $SECONDS

# copy + paste output in work directory
mkdir $PBS_O_WORKDIR/$PBS_JOBID

cp -r * $PBS_O_WORKDIR/$PBS_JOBID/
