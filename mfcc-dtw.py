import librosa
import numpy as np
import sklearn.preprocessing as preprocessing

#get_mfccで、MFCCをモデルとSSが入力された時点に計算。
#Sの音声が入力されたら、get_mfcc(S)でSのmfccも計算し、get_with_mfcc(model_mfcc, S_mfcc, SS_mfcc)で最終結果を出す。
#12秒の音声に対して、全体の時間はおよそ10秒であるが、mfccの計算は4sである。
#なお、librosaのDTWは自分の環境で試せていない。
#SystemError: CPUDispatcher(<function __dtw_calc_accu_cost at 0x7f73bb87c730>) returned a result with an error set
#上記のエラーが出ていて、numpyのバージョンが問題らしいが、解決策を見つけられていない。

def get_mfcc(a): #音声ファイルからMFCCを抽出
    y, sr = librosa.load(a, sr = 16000)
    mfccs = librosa.feature.mfcc(y=y, sr=16000, win_length = 400, n_mfcc=13, hop_length = 320)
    #今のMFCCの次元は13次元、n_mfccで調整する。
    #今のフレーム長は20ms、hop_lengthを二倍にすると、フレーム長も二倍になる。
    res = mfccs.T
    return res


def dtw_with_index_path(x, y):
    #二つの音声MFCCにDTWをかけて、PATHを出力する。
    #MODEL, SSに対して用いる。x = model_mfcc, y = SS_mfcc
    x_len, y_len = len(x), len(y)
    dist_matrix = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            dist_matrix[i, j] = np.linalg.norm(x[i] - y[j])

    dp = np.zeros((x_len + 1, y_len + 1))
    dp[1:, 0] = float('inf')
    dp[0, 1:] = float('inf')

    for i in range(1, x_len + 1):
        for j in range(1, y_len + 1):
            cost = dist_matrix[i-1, j-1]
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    i, j = x_len, y_len
    path = []
    while i > 0 and j > 0:
        path.append([i-1, j-1])
        if dp[i-1, j-1] <= dp[i-1, j] and dp[i-1, j-1] <= dp[i, j-1]:
            i, j = i-1, j-1
        elif dp[i-1, j] <= dp[i, j-1]:
            i -= 1
        else:
            j -= 1

    path.reverse()

    res = [[] for i in range(x_len)]

    for i in path:
        res[i[0]].append(i[1])
    
    return res


def dtw_with_index_all(x, y):
    #mfccに対してDTWをかけ、コストを出力する。
    #SSとSの時に用いる。 x = SS_mfcc, y = S_mfcc
    #出力は0-1に正規化している。
    x_len, y_len = len(x), len(y)
    dist_matrix = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            dist_matrix[i, j] = np.linalg.norm(x[i] - y[j])

    dp = np.zeros((x_len + 1, y_len + 1))
    dp[1:, 0] = float('inf')
    dp[0, 1:] = float('inf')

    for i in range(1, x_len + 1):
        for j in range(1, y_len + 1):
            cost = dist_matrix[i-1, j-1]
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    i, j = x_len, y_len
    path = []
    while i > 0 and j > 0:
        path.append([i-1, j-1, dist_matrix[i-1, j-1]])
        if dp[i-1, j-1] <= dp[i-1, j] and dp[i-1, j-1] <= dp[i, j-1]:
            i, j = i-1, j-1
        elif dp[i-1, j] <= dp[i, j-1]:
            i -= 1
        else:
            j -= 1

    path.reverse()

    res = [[] for i in range(x_len)]

    for i in path:
        res[i[0]].append(i[2])
    
    for j in range(len(res)):
        res[j] = np.mean(res[j])

    newresnew = preprocessing.minmax_scale(res)

    return newresnew

def get_with_mfcc(model_mfcc, S_mfcc, SS_mfcc):
    #mfccを三つ出して、一気にxとyの配列を出す。
    relation = dtw_with_index_path(model_mfcc, SS_mfcc)
    dist = dtw_with_index_all(SS_mfcc, S_mfcc)

    M_dist = []
    for i in range(len(relation)):
        tmp = []
        for j in relation[i]:
            tmp.append(dist[j])
        M_dist.append(np.mean(tmp))

    xy = []

    i = 0

    for i in range(len(M_dist)):
        tmpx = np.mean([max(0, i - 5), min(len(M_dist), i + 5)]) * 0.02
        tmpy = np.mean(M_dist[max(0, i - 5):min(len(M_dist), i + 5)])

        xy.append({"x":tmpx, "y":tmpy})

    return xy
