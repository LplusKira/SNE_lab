pairs: 
  ##############################################################
  whats's inside '1000runs' & 'timeneeded1000Runs'
  ```
    ITEM_FIELDS_NUM=4 MAX_TRAIN_NUM=1000 LEARNING_RATE=0.0002 MOMENTUM=2.0 LAMBDA=1 time python run.py data/u.data.filtered.sorted > 1000runs 2> timeneeded1000Runs
  ```
  in timeneeded1000Runs
    0. the units in timeneeded1000Runs are seconds
  
  in 1000runs
    0. l2 norm of W in each updation           (by usr)
    1. 5N run's avg loss and snapshot of W, V  (by run)
    2. microF1 at each run                     (by run)
  ##############################################################


  ##############################################################
  whats's inside '1000runsLessReg' & 'timeneeded1000RunsLessReg'
  ```
    ITEM_FIELDS_NUM=4 MAX_TRAIN_NUM=1000 LEARNING_RATE=0.0002 MOMENTUM=2.0 LAMBDA=0.1 time python run.py data/u.data.filtered.sorted > 1000runsLessReg 2> timeneeded1000RunsLessReg
  ```
  ##############################################################

