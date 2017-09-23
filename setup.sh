#!/bin/sh

# output file 
FILE=./download.log

# url to retrieve
DATA_URL=
TRAIN_URL=
TEST_URL=

# write header information to the log file
start_date=`date`
echo "START-------------------------------------------------" >> $FILE
echo "" >> $FILE

# retrieve the web page using curl. time the process with the time command.
time (curl --connect-timeout 100 $DATA_URL) >> $FILE 2>
time (curl --connect-timeout 100 $TRAIN_URL) >> $FILE 2>
time (curl --connect-timeout 100 $TEST_URL) >> $FILE 2>

# write additional footer information to the log file
echo "" >> $FILE
end_date=`date`
echo "STARTTIME: $start_date" >> $FILE
echo "END TIME:  $end_date" >> $FILE
echo "" >> $FILE
