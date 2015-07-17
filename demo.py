from ller import *
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt


def demo(k):
    X,t = make_swiss_roll(noise=1)

    lle = LocallyLinearEmbedding(n_components=2,n_neighbors=k)
    lle_X = lle.fit_transform(X)

    ller = LLER(n_components=2,n_neighbors=k)
    ller_X = ller.fit_transform(X,t)

    fig = plt.figure(figsize=plt.figaspect(0.33))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],c=t,s=50)
    ax.set_title('Swiss Roll')
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(lle_X[:,0],lle_X[:,1],c=t,s=50)
    ax.set_title('LLE Embedding')
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(ller_X[:,0],ller_X[:,1],c=t,s=50)
    ax.set_title('LLER Embedding')
    plt.show()


if __name__ == "__main__":
    op = OptionParser()
    op.add_option('--n_neighbors', type=int, metavar='k', default=7,
                  help='# of neighbors for LLE & LLER [7]')
    opts, args = op.parse_args()
    demo(opts.n_neighbors)

