from utils.Dataset import Dataset
from model3 import model
from utils.print_result import print_result
import os
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimization
'''
if __name__ == '__main__':
    data = Dataset('/MSRCV1_6views')
    x1, x2, x3,gt = data.load_data()

    pca = PCA(n_components=20)
    x2 = pca.fit_transform(x2)

    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    x3 = data.normalize(x3, 0)
    n_clusters = len(set(gt))

    act_ae1, act_ae2, act_dg1, act_dg2 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    dims_ae1 = [1302, 200]
    dims_ae2 = [20, 200]
    dims_ae3 = [512, 200]

    dims_dg1 = [100, 200]
    dims_dg2 = [100, 200]
    dims_dg3 = [100, 200]  

    para_lambda = 1
    batch_size = 200
    lr_pre = 1.0e-3
    lr_ae = 1.0e-3
    lr_dg = 1.0e-3
    lr_h = 1.0e-1
    epochs_pre = 50
    epochs_total = 30
    act = [act_ae1, act_ae2, act_dg1, act_dg2]
    dims = [dims_ae1, dims_ae2,dims_ae3, dims_dg1, dims_dg2, dims_dg3]
    lr = [lr_pre, lr_ae, lr_dg, lr_h]
    epochs_h = 100
    epochs = [epochs_pre, epochs_total, epochs_h]

    H, gt = model(x1, x2,x3, gt, para_lambda, dims, act, lr, epochs, batch_size)
    print_result(n_clusters, H, gt)
