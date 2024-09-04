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

