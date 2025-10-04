## 为什么attention要除以d_k**1/2

因为在经历softmax之前，Q*K^T得出的结果的方差很大，如果直接softmax，最后出来的结果就有点倾向于one_hot