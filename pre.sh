for file in ./preprocess/*
do
  if test -f $file
    then
        echo $file 是文件
        if [[ "$file" == "pre.sh" ]];then
          echo "==/"a*/""
        fi
    fi
done
