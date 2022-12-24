if __name__ == '__main__':
    import torch
    p = torch.tensor([0, 0, 0])
    t = torch.tensor([1, 2, 3])
    print(len(torch.unique(t[p == 1]).tolist()))