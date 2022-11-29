import numpy as np

A = np.random.random((4,2,7,3)) + 1.j*np.random.random((4,2,7,3)) - (np.random.random((4,2,7,3)) + 1.j*np.random.random((4,2,7,3)))

with open('4D_complex.txt','w') as f:
    for i in range(4):
        for j in range(2):
            for k in range(7):
                for l in range(3):
                    f.write( str(i) + " " +str(j) + " "+str(k) + " "+str(l)+ " "+   str(A[i,j,k,l].real)+" " + str(A[i,j,k,l].imag) + "\n")



