env="xai"
source activate $env
python -m spacy download en
for file in ./preprocess/*
do
  if test -d $file
    echo $file is dir
    then
        for file2 in $file/*
        do
          if [[ $file2 =~ "pre.sh" ]];then
            bash $file2 $env
          fi
        done
    fi
done
