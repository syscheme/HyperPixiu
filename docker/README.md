# Dockerfiles

## bizdata
is designed to analyze data quickly. It is based on docker.io/tensorflow/tensorflow, and extends with the kits for TA-lib, TuShare, Cryptory and so on in order to perform data analysis on-fly

the recommended command line to start the docker is:
   nvidia-docker run --name=bizdata -p 8888:8888 -v path/to/HyperPixiu:/proj -v /mnt/bigdata/:/data syscheme/bizdata
   
the mount points:
- /proj maps to the project of HyperPixiu
- /data maps where the financial data are stored
