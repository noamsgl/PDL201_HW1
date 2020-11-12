from scipy.io import loadmat
x = loadmat('NNdata/PeaksData.mat')
Ct, Cv, Yt, Yv = x['Ct'], x['Cv'], x['Yt'], x['Yv']

