# Author: James William Chamberlain
# Date: 17-06-2025 

"""
    This is a very simple script file for checking activeness of GEE

    These may take a while to run, the more tasks you have the longer this will take - a longer delay between checks is recommended due to this as it will likely take longer to check than to wait between checks
"""

import ee
import time

ee.Authenticate()
ee.Initialize(project="jameswilliamchamberlain")

def check4usage_legacy():
    """
        Checks the Google Earth Engine (GEE) API for active useage and returns the status as true or false deping on if there are any active or queued tasks.

        This function is marked as legacy as it uses the older `ee.data.getTaskList()` method, which is not recommended for use in new code. 
        However it is still the fastest way to check as it retrieves less data about the task. if this function breaks and alternative one can be found below `check4usage` and this function can be removed.

        Returns:
            bool: True if there are active or queued tasks, False otherwise.
    """

    tl = ee.data.getTaskList() 

    # tl = ee.data.listOperations() # listOperations is the new fn to use as recommended by gee check `check4usage` for implementation
    # states include: PENDING, RUNNING, CANCELLING, SUCCEEDED, CANCELLED, or FAILED 

    if tl is None:
        return False
    else:
        for task in tl:
            if task['state'] in ['READY', 'RUNNING', 'PENDING', 'CANCELING']: # alternative states include 'FAILED', 'CANCELLED', 'CANCELING'
                return True
    return False  

def check4usage():
    """
        Checks the Google Earth Engine (GEE) API for active useage and returns the status as true or false deping on if there are any active or queued tasks.
    """

    # retrive task list 
    tl = ee.data.listOperations() #  if limit is None else ee.data.listOperations(limit=limit)
    # states include: PENDING, RUNNING, CANCELLING, SUCCEEDED, CANCELLED, or FAILED 

    if tl is None:
        return False   
    else:
        for task in tl:
            if task['metadata']['state'] in ['READY', 'RUNNING', 'PENDING', 'CANCELING']:
                return True
    return False


def wait_until_idle(check_interval_s=60, legacy=True, logtime=False):
    """
        Waits until there are no active or queued tasks in the Google Earth Engine (GEE) API.

        Legacy enabled by default, however this uses code that may be deprecated in the future within the GEE API, so disable to use the newer `check4usage` function.

        Args:
            check_interval_s (int): The interval in seconds to wait before checking the task list again.    Default is 60 seconds.
            legacy (bool): If True, uses the legacy `check4usage_legacy` function. If False, uses the updated `check4usage` function. Default is True.
    """

    while check4usage_legacy() if legacy else check4usage():
        if logtime: 
            print(f"Waiting for GEE tasks to complete... TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}") 
        time.sleep(check_interval_s)

# TEST CASE 
# print(f"Legacy getTaskList Code: {check4usage_legacy()}")
# print(f"Update istOperations Code: {check4usage()}")
# wait_until_idle()