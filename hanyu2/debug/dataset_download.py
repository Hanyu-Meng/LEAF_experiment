import os

SAVE_DIR = "./dataset/downloads/voxforge"

def loadSourceFile(path):
    f = open(path)
    data = []
    for file in f.readlines():
        data.append(file.strip())
    return data

def processDownloadfiles(file_list):
    for idx,file in enumerate(file_list):
        localPath = SAVE_DIR+file
        #Check missing & download files
        if not os.path.exists(localPath):
            url = "http://"+file
            print(file)
            #wget.download(url)
        
        #print("CHECKED {0} OF {1}".format(idx,len(file_list)))

if __name__ == '__main__':
    file_list = loadSourceFile("./voxforge_urls.txt")
    processDownloadfiles(file_list)

