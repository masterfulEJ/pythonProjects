import numpy as np

# linear kernel
def kfunc_linear(x1, x2, h=1, l=0):
    #print('kfunc_linear')
    X1, X2 = np.meshgrid(x1, x2)
    return (h * (X1-l) * (X2-l)).T
def kfunc_linear2(x1, x2, l=1):
    #print('kfunc_martern')
    X1, X2 = np.meshgrid(x1, x2)
    return np.abs(X1-X2).T
def kfunc_lin(x1, x2, b=0, v=1, c=0):
    X1, X2 = np.meshgrid(x1, x2)
    return (b**2 + v**2 * (X1-c) * (X2-c)).T

# exponential kernel
def kfunc_exp(x1, x2, l=1):
    #print('kfunc_exp')
    X1, X2 = np.meshgrid(x1, x2)
    return np.exp(-1 * np.abs(X1-X2)).T

# squared exponential kernel
def kfunc_exp_sq(x1, x2, l=1):
    #print('kfunc_exp_sq')
    X1, X2 = np.meshgrid(x1, x2)
    return np.exp(-0.5 * np.abs((X1-X2)/l)**2).T

# browninan kernel
def kfunc_brown(x1, x2, l=1):
    #print('kfunc_brown')
    X1, X2 = np.meshgrid(x1, x2)
    return np.minimum(X1, X2).T

# matern kernel
def kfunc_matern(x1, x2, l=1):
    #print('kfunc_martern')
    X1, X2 = np.meshgrid(x1, x2)
    return ((1 + np.abs(X1-X2)) * np.exp(-1 * np.abs(X1-X2))).T

# gaussian kernel
def kfunc_gauss(x1, x2, l=1):
    #print('kfunc_gauss')
    X1, X2 = np.meshgrid(x1, x2)
    return np.exp(-0.5 * np.abs((X1-X2)/l)**2).T

# sin kernel
def kfunc_sinc(x1, x2, l=1):
    #print('kfunc_sinc')
    X1, X2 = np.meshgrid(x1, x2)
    Sigma = (np.sin(np.abs(X1-X2))/np.abs(X1-X2)).T
    np.fill_diagonal(Sigma, 1)
    return Sigma

# squared exponential kernel
def kfunc_se(x1, x2, h=1, l=1):
    X1, X2 = np.meshgrid(x1, x2)
    return h**2 * np.exp(-1 * ((X1-X2)/l)**2).T

# rational quadratic kernel
def kfunc_rq(x1, x2, h=1, l=1, a=0.5):
    X1, X2 = np.meshgrid(x1, x2)
    return h**2 * ((1 + ((X1-X2)**2)/(a*l**2))**(-a)).T

# periodicic kernel
def kfunc_per(x1, x2, h=1, l=1, p=2):
    X1, X2 = np.meshgrid(x1, x2)
    return h**2 * np.exp(-2/l**2 * np.sin(np.pi * np.abs((X1-X2)/p))**2).T
    #return h**2 * (np.exp(-2/l * np.sin(np.pi * np.abs((X1-X2)/p)))**2).T

# locally periodicic kernel
def kfunc_local_per(x1, x2, h=1, l1=1, p=1, l2=1):
    X1, X2 = np.meshgrid(x1, x2)
    return h**2 * (np.exp(-2/l1**2 * np.sin(np.pi * np.abs((X1-X2)/p))**2) * np.exp(-1 * ((X1-X2)/l2)**2)).T

def kfunc_per_add_lin(x1, x2, h, l, p, b, v, c):
    return kfunc_per(x1, x2, h, l, p) + kfunc_lin(x1, x2, b, v, c)

def kfunc_per_add_lin_gauss(x1, x2, h, l, p, b, v, c, gh, gl):
    return  kfunc_per(x1, x2, h, l, p) + kfunc_lin(x1, x2, b, v, c) + kfunc_se(x1, x2, gh, gl)

if __name__ == '__main__':
    kfunc_linear()
    kfunc_linear2()
    kfunc_lin()
    kfunc_exp()
    kfunc_exp_sq()
    kfunc_brown()
    kfunc_matern()
    kfunc_gauss()
    kfunc_sinc()
    kfunc_se()
    kfunc_rq()
    kfunc_per()
    kfunc_local_per()
    kfunc_per_add_lin()
    kfunc_per_add_lin_gauss()
