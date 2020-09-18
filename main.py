from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digits=load_digits()
tsne=TSNE(n_components=2,init='pca',random_state=501)
# tsne.fit(digits.data)
data=tsne.fit_transform(digits.data)
print(data.shape)
print(data)