import requests
import os
import subprocess
import csv
import time


# rootPath="./"
# rootPath = "D:\\yuansen\\ImPro\\impropy\tests\\wifiCardTests\\"
rootPath = "D:/yuansen/ImPro/impropy/tests/wifiCardTests/"
confFile='wifiCardConf.csv'

# root path
if len(rootPath) <= 0:
    rootPath = "./"
    
# get settings from csv file
file = open(rootPath + confFile)
csvreader = csv.reader(file)
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)
file.close()    
    
# data check

# data
nCard = len(header)
nStep = len(rows)

# print info
print("You requested to acquire files from %d wifi card(s):\n" % (nCard))
for i in range(nCard):
    print("[%d] %s\n" % (i, header[i]), end="")
    print("  to retrive %d images, from '%s' to '%s'.\n" \
          % (len(rows), rows[0][i], rows[-1][i]), end="")



# check connection of each cards
toCheckConnection = False
if (toCheckConnection == True):
    for i in range(nCard):
        print('Checking card %d: %s ...\n' % (i, header[i]))
        retNetshConnect = os.system('cmd /c "netsh wlan connect name=%s"' \
                                    % (header[i]))
    
# get files
for i in range(nCard):
    # connect to card 
    print('Connecting to card %d %s\n' % (i, header[i]))
    os.system('cmd /c "netsh wlan connect name=" %s' % (header[i]))
    # check if the wifi is connected 
    # (and try to re-connect for several times)
    nTrial = 10
    isConnected = False
    for iTrial in range(3):
        # get SSID name --> ssid_str
        netsh_ret = subprocess.check_output(['netsh', 'WLAN', 'show', 'interfaces'])
        netsh_utf8 = netsh_ret.decode('utf-8')
        ssidp = netsh_utf8.find('SSID')
        if (ssidp >= 0):
            netsh_split = netsh_utf8[ssidp:-1].split()
            ssid_str = netsh_split[2]
            # check SSID name
            if (ssid_str == header[i]):
                isConnected = True
                print('  The current SSID is %s. Success.' % (ssid_str))            
                break
        if (ssidp < 0): 
            print('  No SSID is found. Trying again.')
        else:
            print('  The current SSID is %s, not %s.' % (ssid_str, header[i]))            
        time.sleep(1)
    if isConnected == False:
        print('  Tried to connect to %s for %d times. Trying next card.' %\
               (header[i], nTrial))
        continue
    
    # print info
    print('  Create folder %s' % (header[i]))
    os.system('cmd /c cd %s & mkdir \"%s\"' % (rootPath, header[i]))
    for j in range(nStep):
        # define url (theUrl)
        theUrl = rows[j][i]
        if (len(theUrl) > 1):
            print("  Trying to get %s %s" % \
                  (header[i], theUrl))
            # get url 
            retRequest1 = requests.get(theUrl, allow_redirects=True)
            # check status
            if (retRequest1.ok == True):
                print("  The URL request is ok")
                print("  URL size: %d bytes." % (len(retRequest1.content)))
                # define filename
                fileName = header[i] + '/' + theUrl[-12:]
                print("  Saving URL to %s " % (fileName))
                retOpen = open(fileName, 'wb').write(retRequest1.content)
                retRequest1.close()
            else:
               print("  The URL does not exist yet.")


# get all directories under rootPath
# directory_contents = os.listdir(rootPath)
# wifi_card = []
# n_wifi_card = 0
# for item in directory_contents:
#     if os.path.isdir(rootPath + item):
#         n_wifi_card = n_wifi_card + 1
#         wifi_card.append(item)

# # print info
# print("There are %d demanded wifi cards:\n" % (n_card_wifi))
# for i in range(n_wifi_card):
#     print("[%d] %s\n" % (i, wifi_card[i])) 
      
# get file list of each wifi card

    
#     # read files from config ("wifiCardConf.txt")
#         fid1 = open(rootPath + confFile, "r")
        
    
    
    
# cards = [ \
#          'flashair001', \
#         ]

# # connect to the AP
# retNetshConnect = os.system('cmd /c "netsh wlan connect name=%s"' % (cards[0]))

# # Get filesub
# urlPre = 'http://flashair/'
# dirName = 'DCIM/'
# fileName = 'DSC01029.JPG'

# url = urlPre + dirName + fileName
# retRequest1 = requests.get(url, allow_redirects=True)
# print('The request status is ', retRequest1.ok)
# if (retRequest1.ok == True):
#     retOpen = open(fileName, 'wb').write(retRequest1.content)

