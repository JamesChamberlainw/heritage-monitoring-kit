# PlotToSat Files Go HERE! 

Copy files from PlotToSat into this directory, and follow PlotToSat install instructions
https://github.com/Art-n-MathS/PlotToSat 

Then add the python files found here here loop through all files within a GEE directory and run PlotToSat - both files will require adjusting to your specifc area if you are not doing Iraq. Please note do not overwrite anything directly from PlotToSat, instead rename the files here, but as of currently writing this there are no naming overlaps.  

Main files (these are the ones you will need to adjust to your specific site)
`pts_runner.py` - contains the file to loop through multiple shapefiles and will require adjusting to your project requirements. 
`pts_samarra` - personal version of `PlotToSat_test3_shp 1` that can be found in [PlotToSat](https://github.com/Art-n-MathS/PlotToSat), that you should already have if you have reached htis point. NOTE: the `pts_samarra` file will be uploaded when the next PlotToSat update becomes public

Utility file 
`pts_check4usage.py` - contains 3 util functions, the main one you likely would want is `wait_until_idle` that does as it staes in the naming, and uses 1 of 2 potential functions to check if GEE is idle, set by setting legacy to `true` or `false`, it is recommended to test this before you run as the legacy version likely runs faster than the non-legacy due to collecting less unessusary data from GEE, however it relys on `ee.data.getTaskList()` what is listed as legacy code and might one-day be removed! 
