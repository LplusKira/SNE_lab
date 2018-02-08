#!/bin/bash

function assertCurDir {
  ### The following allows: 
  #### ./thisScript.sh 
  #### sh thisScript.sh 
  ### The following disallows: 
  #### ../bin/../bin/thisScript.sh
  #### sh /some/long/path/thisScript.sh

  IsLegal=`echo $1 | awk -F'/' -v SlashCnt=1 '{print (NF <= SlashCnt + 1);}'`
  if [ "${IsLegal}" = "0" ]; then
    echo "[Error] Please go to projectRoot/bin to execute this script"
    exit 1;
  fi
}
