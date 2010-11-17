import numpy as np

def optx(G,m):
    '''
    Discrete constrained optimization G must be a 3xN "gain array" of reals with
    an all zero first row (Nothing invested nothing gained!), G[1,i] gives the
    gain when spending 1 d.o.f. on cell #1, G[2,i] the gain from spending 2 d.o.f.
    Returns the optimal "investment vector" {0,1,2}^N and the optimal gain
    '''
    N = G.shape[1]
    m = min(m,2*N)
    V = np.ones(m+1)*G[2,0]
    V[0] = 0; V[1] = G[1,0]
    T = 2*np.ones((m+1,N),dtype=np.int)
    T[0,:] = 0; T[1,:] = 1; T[2:,0] = 2

    for k in xrange(2,N+1):
        for l in xrange(min(m,2*k),1,-1):
            gain = V[l-2:l+1] + G[2::-1,k-1]
            i = np.argmax(gain)
            T[l,0:(k-1)] = T[l-2+i,0:(k-1)].copy()
            T[l,k-1] = 2-i
            V[l] = gain[i]
        gain = V[0:2] + G[1::-1,k-1]
        i = np.argmax(gain);
        T[1,0:(k-1)] = T[i,0:(k-1)].copy()
        T[1,k-1] = 1-i
        V[1] = gain[i]
    return T[m,:],V[m]

G = np.zeros((3,5));
G[1,:] = np.random.rand(5);
G[2,:] = np.random.rand(5);
x,v = optx(G,6);

print('x = ', x)
print('v = ', v)

