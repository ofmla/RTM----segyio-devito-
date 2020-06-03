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
To run, simply execute the script file
```
./run.sh
```
