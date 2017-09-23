#!/bin/sh

# output file 
FILE=./download.log

# url to retrieve
TRAIN_URL=https://www.dropbox.com/s/h5yora9j0onglw6/train.zip?dl=0
TEST_URL=https://www.dropbox.com/s/c94io61nmldcgv8/test.zip?dl=0

# write header information to the log file
start_date=`date`
echo "START-------------------------------------------------" >> $FILE
echo "" >> $FILE

# retrieve the web page using curl. time the process with the time command.
time (curl --connect-timeout 100 $TRAIN_URL) >> $FILE 2>
time (curl --connect-timeout 100 $TEST_URL) >> $FILE 2>

# write additional footer information to the log file
echo "" >> $FILE
end_date=`date`
echo "STARTTIME: $start_date" >> $FILE
echo "END TIME:  $end_date" >> $FILE
echo "" >> $FILE

