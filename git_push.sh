git status
echo "Are you sure you want to commit all of these push them? (y/n)"
read -p "Enter y to continue: " answer
if [ "$answer" = "y" ]; then
    git add --all
    git commit -m "update"
    #git push --all
    git push -u origin yushengsu-dev-databricks
    echo "Push executed."
else
    echo "Push aborted."
fi

# check log to debug: cat /home/yushensu/projects/Megatron-LM/experiment/4nodes_rank3_train_8B_mbs5_bs320_tp1_pp1_optim_sgd_iter10/nocompile0_TE_FP16_0/2024-09-06_15-52-56/output_perf.log
