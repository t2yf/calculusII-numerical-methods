import math
import matplotlib.pyplot as plt 
import numpy as np
    
#Calcular o erro entre a derivada por diferenças finitas avançadas e a derivada real
def fd_error(f, df, x0, h0, hn, n): 
    #Criar um intervalo, transformando as entradas em números lineares e depois passando 
    #para a escala log/ 10^z
    h_values = np.linspace(np.log10(h0), np.log10(hn), n+1)
    h_values = 10**h_values

    #Valores dos erros
    errors = []

    #Valor exato
    df_exato = df(x0)
    
    for h in h_values:
        #Aproximação da derivada no ponto x0 com diferenças finitas avançadas
        df_difAvanc = (f(x0 + h) -f(x0))/h 

        #Erro absoluto, ponto a ponto
        erro = abs(df_difAvanc - df_exato)
        errors.append(erro)

    #Setar gráfico hXe na escala logXlog e com eixo x invertido
    plt.loglog(h_values,errors) 
    plt.gca().invert_xaxis()

    plt.show()

def ode_solver(f, x0, y0, xn, n, plot):
    #Variáveis e dimensões inicializadas
    h = (xn - x0)/n 
    vet_x = np.linspace(x0, xn, n+1)
    x = np.zeros(n+1)
    y = np.zeros(n+1)

    for i in range(n+1):
        x[i] = x0 
        y[i] = y0
        y0 += h*f(x0, y0)
        x0 += h

    if(plot):
        plt.plot(x, y)
        plt.show()

    return x, y