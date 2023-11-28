from dataAnalysis.showPlot_Acc import showRawData
from util import *
import pandas as pd


# file slicer
def slicer(key=0):
    key = str(key)
    count = 0
    folder_path = 'data/' + key + '/selected/'
    file_list = os.listdir(folder_path)
    file_count = int(len(file_list) / 2)
    for i in range(1, file_count + 1):
        showRawData('data/' + key + '/selected/acc' + key + '_' + str(i) + '.csv', True)

        shift = float(input('shift? : '))
        interval = int(input('interval : '))

        # slice acc data
        dataAcc = pd.read_csv('data/' + key + '/selected/acc' + key + '_' + str(i) + '.csv')
        dataAudioDir = 'data/' + key + '/selected/audio' + key + '_' + str(i) + '.mp4'
        shift *= 10 ** 9
        originalSet = 5.7 * 10 ** 9 + shift
        count = sliceData(dataAcc, dataAudioDir, originalSet, interval, 'data/splitData/', key, count)

        print(f'{count} data created')

        showRawData('data/splitData/acc/acc' + key + '/acc' + key + '_' + str(i) + '.csv', False)


for i in range(10, 14, 1):
    slicer(key=i)


# count = 0
for i in range(10, 14, 1):
    print(f'working on label : {i}')
    key = str(i)
    writeCount = extractFeaturesToCSV('keyboard_tap_full.csv', key, label=i)
    print(f'{writeCount} rows inserted')
