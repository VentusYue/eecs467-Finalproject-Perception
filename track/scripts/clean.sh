#!/bin/bash
# FILE=../targets/*.jpg
# if test -f "$FILE"; then
#     echo "$FILE exist"
# fi
count=`ls -1 ../targets/*.jpg 2>/dev/null | wc -l`
if [ $count != 0 ]; then
    rm -rf ../targets/*.jpg
    echo true
fi

count1=`ls -1 ../frames/*.jpg 2>/dev/null | wc -l`
if [ $count != 0 ]; then
    rm -rf ../frames/*.jpg
    echo true
fi
