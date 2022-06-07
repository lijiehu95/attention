for file in ./preprocess/*
do
  if test -d $file
    echo $file is dir
    then
        for file2 in $file/*
        do
          if [[ "$file2" == "*pre.sh" ]];then
          echo "==/"a*/""
          fi
        done
    fi
done
