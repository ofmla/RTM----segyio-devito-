# RTM w segyio and devito frameworks
Reverse time migration belongs to the class of two-way full wave migration solutions. Rather than performing imaging by extrapolating the data in depth, as in traditional one-way full wave solutions, RTM solves the wave equation forward in time for the source modeling field and backward in time for the recorded receiver field for that shot. At each time step, the depth image is obtained by cross-correlating the two fields. 

## Install dependencies  
To install [devito](https://www.devitoproject.org/) follow the instructions from [Devito documentation](https://www.devitoproject.org/devito/download.html). I recommend to set up a python environment with conda as also suggested in the [installation web page](https://www.devitoproject.org/devito/download.html#conda-environment) 
```
git clone https://github.com/devitocodes/devito.git
cd devito
conda env create -f environment-dev.yml
source activate devito
pip install -e .
```  
All other packages (i.e. [dask jobqueue](https://github.com/dask/dask-jobqueue), [segyio](https://github.com/equinor/segyio)) can be easily installed via `pip` once you have activated the devito environment. To install them, run the following:
```
pip install segyio
pip install dask-jobqueue --upgrade
```
## Notes and workarounds  

The cluster (OGÚN) I use is managed by [Slurm](https://slurm.schedmd.com/overview.html), but, for reasons I am unaware of, it does not allow manually setting the memory of nodes when submitting my computation in a batch mode. Because of this, I implemented a workaround that allowed me to submit my computation (See [dask-jobqueue issue #238](https://github.com/dask/dask-jobqueue/issues/238#issuecomment-468376008)). However, quite recently I realized that my workaround does not work for latest versions (`0.7.0, 0.7.1`) of dask-jobqueue. Conversely, those newer versions provide a way to adress the issue by allowing users to skip some lines in the header when configuring the cluster (See [dask-jobqueue issue #238 reply](https://github.com/dask/dask-jobqueue/issues/238#issuecomment-629994873)). I tried the newly developed solution but I faced a problem, similar to above. The manager set a standard value of `1.5 MB` for the memory of nodes, causing the computation to fail. If you face the same problems you can follow these steps:
```
pip unistall dask-jobqueue
pip install dask-jobqueue==0.4.1
```
You will soon find out that this bring further problems. However, those problems should be solved by downgrading some packages. Install the following packages in the same way as above  
```
distributed==1.27.0
dask==1.1.4
msgpack==0.6.1
```
You would then execute the script normally (see [Run description](#run))

## Get data
In order to run this code, you first need to download the files from [SEG wiki](https://wiki.seg.org/wiki/2007_BP_Anisotropic_Velocity_Benchmark). I organized the shot gathers and anisotropic parameters in two separated folders, `ModelShots` and `ModelParameters` respectively. You can proceed as follows:   
```
mkdir ModelShots && cd ModelShots
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part1.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part2.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part3.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part4.sgy.gz
for i in {1..4}; do gunzip Anisotropic_FD_Model_Shots_part${i}.sgy.gz; done
rm *.gz && cd ..
mkdir ModelParams && cd ModelParams
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/ModelParams.tar.gz
gunzip ModelParams.tar.gz
rm *.gz && cd ..
```
## Run
I create a SLURM job file `dask_launcher.job`. To run, simply submit the batch job
```
sbatch dask_launcher.job
```
You can check the progress of the code with:  
```
tail -f dask_launcher.o<jid>
```
Below, you can see the final image (built only with 5 shots) overlaid with the background velocity model

![alt text](https://github.com/ofmla/RTM----segyio-devito-/blob/master/rtm_figure.png?raw=true)

