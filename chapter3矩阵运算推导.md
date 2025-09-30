X 是 [32, 3] 3代表的是blocksize=3 也就是3个字符作为上下文， 32是选的前32个单词作为输入

C = torch.randn((27, 2)) 因为有27个字符， 每个字符用2个element表示

C[X].shape = (32, 3, 2)

C' = C[X].reshape(32, 6)

W1 = torch.randn((6, 100))
b1 = torch.randn(100)

softmax(C' * W1 + b1)  (32, 100)





