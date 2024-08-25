cd ./pytorch_afo_testkit
pip install -e .
cd ..
path=`pwd`
cd ../set_bc_ib
bash set_bc_ib.sh
cd path
