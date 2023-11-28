import csv
import math
import os
import librosa
from scipy.signal import argrelextrema
from featureExtractor import *
from random import randint
import soundfile as sf

np.set_printoptions(precision=32, suppress=True)

def highest_peak(data, time=None, start_time=-1, end_time=-1):
    if start_time != -1 and end_time != -1:
        mask = (time >= start_time) & (time <= end_time)
        data = data[mask]

    # Find indices of local maxima
    maxima_indices = argrelextrema(data, np.greater)[0]

    # Identify the index of the highest peak
    if len(maxima_indices) == 0:
        return None, None
    index_highest_peak = maxima_indices[np.argmax(data[maxima_indices])]
    return index_highest_peak, data[index_highest_peak]


def crop_accelerometer_data(data, start_time, end_time):
    # Assuming there's a 'time' column in the CSV
    cropped_data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]

    return cropped_data


def crop_audio_data(audio_file, save_file, start, end):
    y, sr = librosa.load(audio_file, sr=None)
    ny = y[start * sr:end * sr]
    librosa.output.write_wav(save_file + '.mp4', ny, sr)


def frameLimitChecker(df, target_size):
    original_size = len(df)

    # Check if the DataFrame is already at or below the target size
    if original_size <= target_size:
        return df

    # Calculate the difference ratio
    diff_ratio = original_size / target_size

    # If the ratio is close to 1, remove rows at random intervals
    if diff_ratio < 2:
        new_df = df.copy()
        while len(new_df) > target_size:
            # Remove a random row
            index_to_remove = randint(0, len(new_df) - 1)
            new_df = new_df.drop(new_df.index[index_to_remove])
        return new_df, None

    # If the ratio is 2 or more, return even or odd indexed rows
    else:
        # Decide whether to take even or odd indexed rows
        # even_or_odd = randint(0, 1)
        evenDf = df.iloc[::2]
        oddDf = df.iloc[1::2]
        return evenDf, oddDf


def sliceData(originalAccData, originalAudioDataDir, originalStartTime, interval, root, key, start, frameLimit=40):

    # audio init
    y, sr = librosa.load(originalAudioDataDir, sr=None)

    # acc init
    nsToSecs = originalAccData['time'].values
    nsToSecs -= nsToSecs[0]
    nsToSecs = nsToSecs/1000000000.0  # 1000000000ns = 1sec

    lastSec = math.floor(nsToSecs[-1])
    # print(lastSec)
    count = start
    for i in range(0, lastSec-2, interval):
        try:
            # start : 4.7 * 10 ** 9 | end : inf
            startTime = originalStartTime + (i * 10 ** 9)
            data = crop_accelerometer_data(originalAccData, startTime, startTime + (1 * 10 ** 9))
            time = data['time'].values.astype('float64')  # in nanoseconds
            if len(time) == 0:
                break
            time -= time[0]
            time /= 1000000000.0  # 1000000000ns = 1sec
            accel_z = data['z'].values  # 전화기 새웠을 때 가로

            index_peak_z, value_peak_z = highest_peak(data=accel_z, time=data['time'])
            time *= 1000000000.0  # 1000000000ns = 1sec
            highest_peak_time = startTime + time[index_peak_z]

            data = crop_accelerometer_data(originalAccData, highest_peak_time - (0.2 * 10 ** 9),
                                           highest_peak_time + (0.6 * 10 ** 9))

            time = data['time'].values.astype('float64')  # in nanoseconds
            time -= time[0]
            time /= 1000000000.0  # 1000000000ns = 1sec

            # audio ###########
            highest_peak_timeToSec = highest_peak_time / 1000000000.0
            cropStart = round((highest_peak_timeToSec - 0.2) * sr)
            cropEnd = round((highest_peak_timeToSec + 0.6) * sr)
            ny = y[cropStart:cropEnd]
            # print(f' test : {len(ny)}')
            # #################

            accel_x = data['x'].values  # 전화기 디스플레이 기준으로 수평으로 위 아래
            accel_y = data['y'].values  # 전화기 새웠을 때 세로
            accel_z = data['z'].values  # 전화기 새웠을 때 가로

            # Create a DataFrame from the lists
            df = pd.DataFrame({
                'time': time,
                'x': accel_x,
                'y': accel_y,
                'z': accel_z
            })
            df1, df2 = frameLimitChecker(df, frameLimit)

            # Write the DataFrame to a new CSV & mp4 file
            df1.to_csv(root + 'acc/acc' + key + '/' + 'acc' + key + '_' + str(count) + '.csv', index=False)
            # print(f' test : {len(ny)}')
            sf.write(root + 'audio/audio' + key + '/' + 'audio' + key + '_' + str(count) + '.wav', ny, int(sr), 'PCM_24')
            count += 1
            if df2 is not None:
                df2.to_csv(root + 'acc/acc' + key + '/' + 'acc' + key + '_' + str(count) + '.csv', index=False)
                sf.write(root + 'audio/audio' + key + '/' + 'audio' + key + '_' + str(count) + '.wav', ny, int(sr), "PCM_24")
                count += 1
        except Exception as e:
            print(e)
    return count


def dataStrToNum(rawData):
    data = list()
    # print(' - - - - ')

    for row_str in rawData:
        try:
            row = row_str.strip().replace('[', '').replace(']', '').replace('\n', '')
            row = row.split(',')
            rowListed = [float(i) for i in row]
            data.append(rowListed)
        except:
            pass

    data = np.array(data)

    return data


def standarize(data):
    return (data - data.mean()) / data.std()


def extractFeaturesToCSV(write, key, label):
    writeCount = 0

    if not os.path.isfile(write):
        df = pd.DataFrame(None,
                          columns=['label', 'psd_x', 'psd_y', 'psd_z', 'fft_x', 'fft_y', 'fft_z', 'acf_x', 'acf_y', 'acf_z', 'audio_mfcc', 'audio_fft'])
        df.to_csv(write, index=False)

    folder_path = "data/splitData/acc/acc" + key
    file_list = os.listdir(folder_path)
    file_count = len(file_list)

    for i in range(file_count):
        try:
            accData = pd.read_csv('data/splitData/acc/acc' + key + '/' + 'acc' + key + '_' + str(i) + '.csv')
            audio, sr = librosa.load('data/splitData/audio/audio' + key + '/' + 'audio' + key + '_' + str(i) + '.wav', sr=None)
            time = accData['time'].values
            time -= time[0]

            # acc data
            accel_x = accData['x'].values  # 전화기 디스플레이 기준으로 수평으로 위 아래
            accel_y = accData['y'].values  # 전화기 새웠을 때 세로
            accel_z = accData['z'].values  # 전화기 새웠을 때 가로

            # feature extraction
            x_f, x_S = psd(accel_x)
            y_f, y_S = psd(accel_y)
            z_f, z_S = psd(accel_z)

            fft_x = fft(accel_x)
            fft_y = fft(accel_y)
            fft_z = fft(accel_z)

            acf_x = [acf(accel_x, k) for k in range(20)]
            acf_y = [acf(accel_y, k) for k in range(20)]
            acf_z = [acf(accel_z, k) for k in range(20)]

            audio_mfcc = compute_mfcc(audio, sr)
            audio_fft = fft(audio, 300)

            oneRowLen = len(x_S) * 3 + len(fft_x) * 3 + len(acf_x) * 3 + len(audio_mfcc) + len(audio_fft)

            if oneRowLen == 1583:
                f = open(write, 'a', newline='')
                wr = csv.writer(f)
                wr.writerow([label, x_S, y_S, z_S, fft_x, fft_y, fft_z, acf_x, acf_y, acf_z, audio_mfcc, audio_fft])
                writeCount += 1
            else:
                print('data is missing SKIP')
        except Exception as e:
            print(e)
        finally:
            f.close()

    return writeCount

