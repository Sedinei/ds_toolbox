from pyod import IsolationForest
import numpy as np

def show_isolation_forest(df, feature):
    X = df[feature].values.reshape(-1, 1)
    isolation_forest = IsolationForest(n_estimators = 100)
    isolation_forest.fit(X)
    xx = np.linspace(df[feature].min(), df[feature].max(),len(df).reshape(-1, 1))
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)
    plt.figure(figsize=(10, 4))
    plt.plot(xx, anomaly_score, label='anomaly score')
    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), where= outlier==-1, color = 'r')
    plt.legend()
    plt.xlabel('anomaly score')
    plt.ylabel('fare')
    plt.show()