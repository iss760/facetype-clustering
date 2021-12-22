import pandas as pd
import pickle
from sklearn.cluster import KMeans


N_CLUSTERS = 10


def save_pickle(d, file_name='test'):
    f = open(file_name + '.pickle', 'wb')
    pickle.dump(d, f)
    f.close()


# Data load
src_data = pd.read_csv('face_features.csv', index_col=0)
print(src_data)

# file name 칼럼 제거
data = src_data.drop(columns=['fileName'])

# k-means clustering 학습
kMeans = KMeans(n_clusters=N_CLUSTERS)
kMeans.fit(data)

# 결과 확인
result = src_data.copy()
result["cluster"] = kMeans.labels_
result.to_csv('./result/result.csv')

# 모델 저장
save_pickle(kMeans, file_name='model')
