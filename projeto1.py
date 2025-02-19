import math

def simpson(f, a, b, n):
    soma=0
    h=(b-a)/n

    for i in range(1, n, 2):
        soma = soma + 2*f(a + i*h) + f(a+(i+1)*h)

    resp= h/3*((f(a)+ 2*soma- f(b)))

    return resp

def trapezio(f, a, b, n):
    soma = 0
    h=(b-a)/n

    for i in range(1, n, 1):
        soma=soma + f(a+i*h)

    resp = h/2*(f(a) + 2*soma + f(b))

    return resp