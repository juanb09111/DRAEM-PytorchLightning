import os.path

def get_anomaly_files(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)

    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_anomaly_files(fullPath)
        else:
            # if is image
            if fullPath.find("png") != -1 or fullPath.find("jpg") != -1 or fullPath.find("jpeg") != -1:
                allFiles.append(fullPath)

    return allFiles