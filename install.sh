# cd ./pytorch_afo_testkit
# pip install -e .
# cd ..

echo '* soft nofile unlimited' >> /etc/security/limits.conf
echo '* hard nofile unlimited' >> /etc/security/limits.conf

path=`pwd`
cd ../set_bc_ib
bash set_ib.sh
cd $path
# python scan_multinode.py --seq_length=2048
bash tmp_run_multinode_tmp.sh
rm tmp_run_multinode_tmp.sh

