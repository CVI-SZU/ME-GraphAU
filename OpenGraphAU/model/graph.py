import torch


#Used in stage 1 (ANFL)
def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A


#Used in stage 2 (MEFL)
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end


