import numpy as np
import time

n = 3
r = 3

'''n1=2
n2=3
n3=3
n4=4'''


with open("4D_complex.txt", "r") as f:
    content = f.readlines()
    last = content[-1]
    last_ele = last.split()
    n1 = int(last_ele[0])+1
    n2 = int(last_ele[1])+1
    n3 = int(last_ele[2])+1
    n4 = int(last_ele[3])+1
    A = np.zeros((n1,n2,n3,n4), dtype = 'complex128')
    for i,a in enumerate(content):
        line = a.split()
        i = int(line[0])
        j = int(line[1])
        k = int(line[2])
        l = int(line[3])
        rel = float(line[4])
        img = float(line[5])
        A[i,j,k,l] = rel+1.0j*img

#A = np.random.random((n1,n2,n3,n4)) + 1.j*np.random.random((n1,n2,n3,n4))
a = np.matrix(np.random.random((n1,r)) + 1.j*np.random.random((n1,r)) )
b = np.matrix(np.random.random((n2,r)) + 1.j*np.random.random((n2,r)) )
c = np.matrix(np.random.random((n3,r)) + 1.j*np.random.random((n3,r)) )
d = np.matrix(np.random.random((n4,r)) + 1.j*np.random.random((n4,r)) )


def normalize_cols(mat):
    nr,nc = mat.shape
    col_norm = 0.0
    normalized_mat = np.zeros((mat.shape),dtype = mat.dtype)
    for a in range(nc):
        for b in range(nr):
            col_norm += np.abs(mat[b,a])**2
        for b in range(nr):
            normalized_mat[b,a] = mat[b,a]/ np.sqrt(col_norm)
        col_norm=0.0
    return np.matrix(normalized_mat)

a = normalize_cols(a)
b = normalize_cols(b)
c = normalize_cols(c)
d = normalize_cols(d)
err = 0.0
for i in range(1000):
    W = np.einsum( 'fq,qr,fs,sr,ft,tr->fr', b.H,b ,c.H,c,d.H,d)
    V = np.einsum( 'pqst,rq,rs,rt->rp',A,b.H,c.H,d.H)
    a1 = np.linalg.solve(W,V).T
    a1=normalize_cols(a1)
    W = np.einsum('fp,pr,fs,sr,ft,tr->fr', a1.H,a1 ,c.H,c,d.H,d)
    V = np.einsum( 'pqst,rp,rs,rt->rq',A,a1.H,c.H,d.H)
    b1 = np.linalg.solve(W,V).T
    b1 = normalize_cols(b1)

    W = np.einsum('fp,pr,fq,qr,ft,tr->fr', a1.H,a1 ,b1.H,b1,d.H,d)
    V = np.einsum( 'pqst,rp,rq,rt->rs',A,a1.H,b1.H,d.H)
    c1 = np.linalg.solve(W,V).T
    c1 = normalize_cols(c1)

    W = np.einsum('fp,pr,fq,qr,fs,sr->fr', a1.H,a1 ,b1.H,b1,c1.H,c1)
    V = np.einsum( 'pqst,rp,rq,rs->rt',A,a1.H,b1.H,c1.H)
    d1 = np.linalg.solve(W,V).T
    A_ = np.einsum('pr,qr,sr,tr->pqst' , a1,b1,c1,d1)
    d1 = normalize_cols(d1)
    new_err = 0.5*(np.linalg.norm( A.reshape((n1*n2*n3*n4)) - A_.reshape((n1*n2*n3*n4))))**2
    del_err = np.abs(err - new_err)
    err = new_err
    print( i, del_err , new_err)
    if(del_err<1e-12):
        break
    a = a1
    b=b1
    c=c1
    d=d1



