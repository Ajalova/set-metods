from matplotlib import pyplot as plt
import numpy as np
import math
n=63
N=1


aa=n-64
bb=N/4
k=40
phi_begin=aa
phi_end=3*aa+bb*2
A=[]

h=(1+1)/(k)
for i in range(k+1):
    A.append(-1+i*h)

F=[]
F.append([0] * k)
print(A)
def q(x):
    return -4
def p(x):
    return x
def v(x):
    return (-x**3+12*x**2+6*x-4)*aa+(-2*x**2-3*x+2)*bb

for i in range(1,k-1):
    F.append([0] * k)
    F[i][i - 1] = 1 / h ** 2 - p(A[i]) / (2 * h)
    F[i][i] = -2 / h ** 2 + q(A[i])
    F[i][i + 1] = 1 / h ** 2 + p(A[i]) / 2 / h

F[0][0]=1
F.append([0] * k)
F[k-1][k-1]=1
print('\n'.join([''.join(['{:8}'.format(round(item,2)) for item in row]) for row in F]))

L=[]
M=[]
L.append(-F[0][1]/F[0][0])
M.append(phi_begin/F[0][0])
for i in range(1,k):

    if (i != k - 1):
        a = F[i][i - 1]
        b = F[i][i]
        c = F[i][i + 1]
        L.append(-c / (L[i-1]*a+b))
        M.append((v(A[i])-M[i-1]*a)/(L[i-1]*a+b))
    else:
        L.append(-c / (L[i - 1] * a + b))
        M.append((v(A[i]) - M[i - 1] * a) / (L[i - 1] * a + b))
        a = F[i][i - 1]
        b = F[i][i]
        M.append((phi_end - M[i - 1] * a) / (L[i - 1] * a + b))


#M.append((phi_end-M[k-1]*F[k-2][k-1])/(L[k-1]*F[k-2][k-1]+F[k-1][k-1]))
x=[0]*k
x[k-1]=M[k]
for i in range(k-2,-1,-1):
    x[i]=L[i+1]*x[i+1]+M[i+1]

print(' '.join(['{:8}'.format(round(item,2)) for item in M]))
print(' '.join(['{:8}'.format(round(item,2)) for item in L]))
print(' '.join(['{:8}'.format(round(item,2)) for item in x]))

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()

t = np.arange(-1,1,h)
y1 = x
y2= [-i**4-i**3+1/4*i**2+1/4*i-1 for i in t]
pog=[math.sqrt((y1[i]-y2[i+1])**2) for i in range(k-1)]
print(max(pog))
x2=[-0.9829426793082596, -0.9690457237448047, -0.9580420152297798, -0.9496738087513427, -0.9436927323625273, -0.9398597871780351, -0.9379453473709525, -0.9377291601693956, -0.9390003458530836, -0.9415573977498399, -0.9452081822320235, -0.9497699387128897, -0.9550692796428831, -0.9609421905058619, -0.967234029815256, -0.9737995291101591, -0.9805027929513586, -0.9872172989173001, -0.9938258975999943, -1.0002208126008618, -1.0063036405265224, -1.0119853509845274, -1.0171862865790389, -1.0218361629064583, -1.0258740685510035, -1.0292484650802407, -1.0319171870405703, -1.0338474419526709, -1.0350158103069018, -1.0354082455586704, -1.0350200741237634, -1.0338559953736453, -1.0319300816307286, -1.029265778163617, -1.0258959031823234, -1.0218626478334698, -1.0172175761954663, -1.0120216252736767, -1.0063451049955712, -1.000267698205871, -0.9938784606616852, -0.987275821027647, -0.980567580871047, -0.9738709146569722, -0.9673123697434491, -0.9610278663765978, -0.9551626976857961, -0.9498715296788603, -0.9453184012372431, -0.9416767241112529, -0.939129282915297, -0.9378682351231509, -0.9380951110632577, -0.9400208139140596, -0.9438656196993637, -0.9498591772837458, -0.9582405083679941, -0.9692580074845955, -0.9831694419932662, -1.0002419520765304, -1.020752050735348, -1.0449856237847934, -1.0732379298497892, -1.1058136003608936, -1.1430266395501476, -1.1852004244469794, -1.232667704874171, -1.2857706034438865, -1.3448606155537652, -1.4102986093830796, -1.4824548258889592, -1.5617088788026838, -1.6484497546260444, -1.7430758126277741, -1.8459947848400493, -1.9576237760550637, -2.0783892638216726, -2.2087270984421106, -2.349082502968782, -2.5]
x22=[x2[i] for i in range(0,2*k,2)]
print(' '.join(['{:8}'.format(round(item,2)) for item in x]))
print(' '.join(['{:8}'.format(round(item,2)) for item in x22]))
pog2=[math.sqrt((x22[i]-x[i])**2) for i in range(k)]
print(max(pog)/3)

plt.plot(t,y1,"r",label = "численное решение")
plt.plot(t,y2, "*b",label = "аналитическое решение")
ax.legend()
plt.show()
