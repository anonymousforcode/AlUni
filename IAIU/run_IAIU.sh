unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python scripts/run.py IAIU_configs/IAIU_drq_easy/cup.json
#python scripts/run.py IAIU_configs/IAIU_drqv2_easy/cup.json

