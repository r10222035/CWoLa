import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.cluster import KMeans, SpectralClustering

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# solve the problem of "libdevice not found at ./libdevice.10.bc"
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/r10222035/.conda/envs/tf2'

# prepare fake data: three 5D Gaussian distributions

muA = [0, 0, 0, 0, 0]
covA = np.eye(5)
muB = [1, 1, 1, 1, 1]
covB = np.eye(5)
muC = [-1, -1, -1, -1, -1]
covC = np.eye(5)

# # dominated case 1
# f1A, f1B, f1C = 0.7, 0.2, 0.1
# f2A, f2B, f2C = 0.1, 0.7, 0.2
# f3A, f3B, f3C = 0.2, 0.1, 0.7

# # dominated case 2
# f1A, f1B, f1C = 0.80, 0.10, 0.10
# f2A, f2B, f2C = 0.70, 0.25, 0.05
# f3A, f3B, f3C = 0.60, 0.20, 0.20

# ambiguous case
case_name = 'Ambiguous'
f1A, f1B, f1C = 0.80, 0.10, 0.10
f2A, f2B, f2C = 0.70, 0.20, 0.10
f3A, f3B, f3C = 0.60, 0.25, 0.15

# set random seed
np.random.seed(24)

def generate_data(n_samples=10000):
    
    
    fractions = np.array([[f1A, f1B, f1C], [f2A, f2B, f2C], [f3A, f3B, f3C]])

    # normalize fractions
    fractions = fractions / fractions.sum(axis=1)[:, None]

    n_A_1, n_B_1, n_C_1 = (fractions[0] * n_samples).astype(int)
    n_A_2, n_B_2, n_C_2 = (fractions[1] * n_samples).astype(int)
    n_A_3, n_B_3, n_C_3 = (fractions[2] * n_samples).astype(int)
    
    M1 = np.concatenate([np.random.multivariate_normal(muA, covA, size=n_A_1),
                         np.random.multivariate_normal(muB, covB, size=n_B_1),
                         np.random.multivariate_normal(muC, covC, size=n_C_1)])
    
    M2 = np.concatenate([np.random.multivariate_normal(muA, covA, size=n_A_2),
                            np.random.multivariate_normal(muB, covB, size=n_B_2),
                            np.random.multivariate_normal(muC, covC, size=n_C_2)])
    
    M3 = np.concatenate([np.random.multivariate_normal(muA, covA, size=n_A_3),
                            np.random.multivariate_normal(muB, covB, size=n_B_3),
                            np.random.multivariate_normal(muC, covC, size=n_C_3)])
    
    X = np.concatenate([M1, M2, M3])
    # one-hot encoding
    y = np.concatenate([np.tile([1, 0, 0], (M1.shape[0], 1)),
                        np.tile([0, 1, 0], (M2.shape[0], 1)),
                        np.tile([0, 0, 1], (M3.shape[0], 1))])
    
    y_true = np.concatenate([np.tile([1, 0, 0], (n_A_1, 1)),
                             np.tile([0, 1, 0], (n_B_1, 1)),
                             np.tile([0, 0, 1], (n_C_1, 1)),
                             np.tile([1, 0, 0], (n_A_2, 1)),
                             np.tile([0, 1, 0], (n_B_2, 1)),
                             np.tile([0, 0, 1], (n_C_2, 1)),
                             np.tile([1, 0, 0], (n_A_3, 1)),
                             np.tile([0, 1, 0], (n_B_3, 1)),
                             np.tile([0, 0, 1], (n_C_3, 1))])
    
    return X, y, y_true


def build_model(input_dim, n_class=3):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def get_accuracy(y_true, y_pred):

    # 定義映射規則 (以陣列形式表示)
    mappings = np.array([
        [0, 1, 2],  # 原始
        [0, 2, 1],  # 0->0, 1->2, 2->1
        [2, 1, 0],  # 0->2, 1->1, 2->0
        [1, 0, 2],  # 0->1, 1->0, 2->2
        [2, 0, 1],  # 0->2, 1->0, 2->1
        [1, 2, 0],  # 0->1, 1->2, 2->0
    ])

    all_arrays = mappings[:, y_pred]

    # evaluate ACC for all mappings
    ACC = 0
    for i, arr in enumerate(all_arrays):
        ACC_tem = (y_true == arr).mean()
        if ACC_tem > ACC:
            ACC = ACC_tem
            y_pred = arr

    ACC = (y_true == y_pred).mean()
    
    ACCs = []
    for i in range(3):
        ACCs.append((y_true[y_true == i] == y_pred[y_true == i]).mean())

    return ACC, ACCs


def main():
    # generate data
    X_train, y_train, y_true = generate_data()
    X_val, y_val, y_true_val = generate_data(n_samples=1000)

    n_test = 1000
    X_test = np.concatenate([np.random.multivariate_normal(muA, covA, size=n_test), 
                            np.random.multivariate_normal(muB, covB, size=n_test),
                            np.random.multivariate_normal(muC, covC, size=n_test)])
    y_test = np.concatenate([np.tile([1, 0, 0], (n_test, 1)),
                            np.tile([0, 1, 0], (n_test, 1)),
                            np.tile([0, 0, 1], (n_test, 1))])
    
    # 檢查每個類別的數量
    n_A_1 = np.sum(y_true[y_train.argmax(axis=1) == 0].argmax(axis=1) == 0)
    n_B_1 = np.sum(y_true[y_train.argmax(axis=1) == 0].argmax(axis=1) == 1)
    n_C_1 = np.sum(y_true[y_train.argmax(axis=1) == 0].argmax(axis=1) == 2)
    n_A_2 = np.sum(y_true[y_train.argmax(axis=1) == 1].argmax(axis=1) == 0)
    n_B_2 = np.sum(y_true[y_train.argmax(axis=1) == 1].argmax(axis=1) == 1)
    n_C_2 = np.sum(y_true[y_train.argmax(axis=1) == 1].argmax(axis=1) == 2)
    n_A_3 = np.sum(y_true[y_train.argmax(axis=1) == 2].argmax(axis=1) == 0)
    n_B_3 = np.sum(y_true[y_train.argmax(axis=1) == 2].argmax(axis=1) == 1)
    n_C_3 = np.sum(y_true[y_train.argmax(axis=1) == 2].argmax(axis=1) == 2)
    print(f'M1: A: {n_A_1}, B: {n_B_1}, C: {n_C_1}')
    print(f'M2: A: {n_A_2}, B: {n_B_2}, C: {n_C_2}')
    print(f'M3: A: {n_A_3}, B: {n_B_3}, C: {n_C_3}')


    # 建立模型並訓練
    model = build_model(X_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, patience=5)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=1024, verbose=1, callbacks=[early_stopping,])


    # evaluate performance for training and testing data
    results_train = model.evaluate(x=X_train, y=y_train, batch_size=1024)
    print(f'Training Loss = {results_train[0]:.3}, Training Accuracy = {results_train[1]:.3}')
    results_test = model.evaluate(x=X_test, y=y_test, batch_size=1024)
    print(f'Testing Loss = {results_test[0]:.3}, Testing Accuracy = {results_test[1]:.3}')


    # evaluate accuracy for argmax
    y_test_predict = model.predict(X_test, batch_size=1024)
    ACC_argmax, ACCs_argmax = get_accuracy(y_test.argmax(axis=1), y_test_predict.argmax(axis=1))
    print(f'Argmax ACC: {ACC_argmax:.3f}, ACCs: {ACCs_argmax}')

    # evaluate accuracy for k-means
    kmeans = KMeans(n_clusters=3, random_state=45)
    kmeans.fit(y_test_predict)
    labels = kmeans.labels_

    ACC_k_means, ACCs_k_means = get_accuracy(y_test.argmax(axis=1), labels)
    print(f'K-means ACC: {ACC_k_means:.3f}, ACCs: {ACCs_k_means}')

    # evaluate accuracy for spectral clustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=45)
    labels = spectral.fit_predict(y_test_predict)

    ACC_spectral, ACCs_spectral = get_accuracy(y_test.argmax(axis=1), labels)
    print(f'Spectral ACC: {ACC_spectral:.3f}, ACCs: {ACCs_spectral}')
    

    # Write results
    now = datetime.datetime.now()
    file_name = 'multi-class_CWoLa_training_results.csv'
    data_dict = {
                'Train size': [y_train.shape[0]],
                'Validation size': [y_val.shape[0]],
                'Test size': [y_test.shape[0]],
                'Case': [case_name],
                'Mixing fractions': [[f1A, f1B, f1C, f2A, f2B, f2C, f3A, f3B, f3C]],
                'Training loss': [results_train[0]],
                'Training accuracy': [results_train[1]],
                'Testing loss': [results_test[0]],
                'Testing accuracy': [results_test[1]],
                'Argmax ACC': [ACC_argmax],
                'Argmax ACCs': [ACCs_argmax],
                'K-means: ACC': [ACC_k_means],
                'K-means: ACCs': [ACCs_k_means],
                'Spectral: ACC': [ACC_spectral],
                'Spectral: ACCs': [ACCs_spectral],
                'Training epochs': [len(history.history['loss']) + 1],
                'time': [now],
                }

    df = pd.DataFrame(data_dict)
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)


if __name__ == '__main__':
    main()