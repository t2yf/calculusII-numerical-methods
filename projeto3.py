import math
import matplotlib.pyplot as plt
import numpy as np

##### Calculo da Derivada Parcial com Diferenças Finitas Centradas

def fin_diff(f, x, degree, h):
    #Derivadas de primeira ordem
    if(degree ==1):
        #Coordenadas de x determinam o tamanho do vetor gradiente
        n = len(x)
        gradient = np.zeros(n)

        for i in range(n):
            ei = np.zeros_like(x)
            #Vetor unitario na posicao i = 1
            ei[i] = 1
            #Derivada parcial na pos i
            gradient[i] = (f(x + h * ei) - f(x - h * ei)) / (2 * h)

        return gradient

    #Matriz Hessiana
    if(degree == 2):
        #Tamanho de x determina a matriz hessiana
        n = len(x)
        hess = np.zeros((n,n))

        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n)
                ej = np.zeros(n)
                ei[i] = h
                ej[j] = h

                #Diagonal da matriz -> Derivadas puras
                if i == j:
                    hess[i,i] = (f(x + ei) - 2*f(x) + f(x - ei))/h**2

                #Derivadas mistas
                else:
                    f1 = f(x + ei + ej)  # f(x + h_i + h_j)
                    f2 = f(x + ei - ej)  # f(x + h_i - h_j)
                    f3 = f(x - ei + ej)  # f(x - h_i + h_j)
                    f4 = f(x - ei - ej)

                    hess[i,j] = (f1 - f2 - f3 + f4)/4*h**2
                    #Simetria
                    hess[j,i] = hess[i,j]

        return hess


##### Método do Gradiente

def gd(f, x0, grad, eps, alpha, itmax, fd, h, plot, search): 
    x = x0
    k = 0

    #Guardar os pontos percorridos
    path = [x0]

    #Calcular por Diferenças Finitas
    if(fd):
        while np.linalg.norm(fin_diff(f, x, 1, 1e-5))> eps and k < itmax:
            k += 1
            #Busca Linear
            if search: 
                 alpha = linesearch(f, x, fin_diff(f, x, 1, 1e-5), -fin_diff(f, x, 1, 1e-5))

            x = x - alpha*fin_diff(f, x, 1, 1e-5)
            #Guardar novo ponto percorrido
            path.append(x)
    else:
        while np.linalg.norm(grad(x))> eps and k < itmax:
            k += 1
            #Busca Linear
            if(search):
                alpha = linesearch(f, x, grad(x), -grad(x))

            x = x - alpha*grad(x)
            #Guardar novo ponto percorrido
            path.append(x)

    #Output gráfico, apenas para R²
    if(plot and len(x0) == 2):
        #Transformar path para array para ser possível plotar o caminho
        path = np.array(path)

        x_min, x_max = path[:, 0].min(), path[:, 0].max()
        y_min, y_max = path[:, 1].min(), path[:, 1].max()    

        eixo_x = np.linspace(x_min - 1, x_max + 1, 100)
        eixo_y = np.linspace(y_min - 1, y_max + 1, 100)
        X, Y = np.meshgrid(eixo_x, eixo_y)
        Z = f_grid(X, Y)

        #Contorno com escala (viridis) em amarelo (alto) - roxo (baixo)
        contour = plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.clabel(contour, inline=True, fontsize=8)
        #Legenda para as cores
        plt.colorbar(label="f(x, y)")

        #Pegar pedaços do caminho guardado em path e plotar
        plt.plot(path[:, 0], path[:, 1], 'o-', color='red', label="Caminho do gradiente")
        #Marcar o ponto inicial, chute inicial x0
        plt.scatter(x0[0], x0[1], color='blue', label='Ponto inicial', zorder=5)

        plt.title("Minimização pelo Método do Gradiente")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    return x, k


##### Método de Newton 

def newton(f, x0, grad, hess, eps,  alpha, itmax, fd, h, plot, search): 
    x = x0
    k = 0

    #Guardar os pontos percorridos
    path = [x0]

    while np.linalg.norm(grad(x)) > eps and k < itmax:
        k+=1

        #Busca linear
        if(search and fd):
            alpha = linesearch(f, x, fin_diff(f, x, 1, 1e-5), -fin_diff(f, x, 1, 1e-5))
        if(search):
            alpha = linesearch(f, x, grad(x), -grad(x))

        
        try:
            #Resolver sistema linear
            if(fd):
                H = fin_diff(f, x, 2, 1e-5)
                d = np.linalg.solve(H, -grad(x))
            else:
                H = hess(x)
                d = np.linalg.solve(H, -grad(x))

         
        except LinAlgError:
            #Salva-guarda matriz singular
            H =0.9*H + 0.1* np.eye(H.shape[0]) 
            #Resolver sistema linear
            if(fd):
                H = fin_diff(f, x, 2, 1e-5)
                d = np.linalg.solve(H, -grad(x))
            else:
                H = hess(x)
                d = np.linalg.solve(H, -grad(x))

        #Salva-guarda direção
        # Produto escalar entre gradiente e direção
        prod_esc = np.dot(grad(x), d)  
        while prod_esc > -1e-3*np.linalg.norm(grad(x)) * np.linalg.norm(d):
            H =0.9*H + 0.1* np.eye(H.shape[0]) 
            #Resolver sistema linear
            if(fd):
                H = fin_diff(f, x, 2, 1e-5)
                d = np.linalg.solve(H, -grad(x))
            else:
                H = hess(x)
                d = np.linalg.solve(H, -grad(x))


        x = x + alpha * d
        #Guardar novo ponto percorrido
        path.append(x)

    #Output gráfico, apenas para R²
    if(plot and len(x0) == 2):
        #Transformar path para array para ser possível plotar o caminho
        path = np.array(path)

        x_min, x_max = path[:, 0].min(), path[:, 0].max()
        y_min, y_max = path[:, 1].min(), path[:, 1].max()    

        eixo_x = np.linspace(x_min - 1, x_max + 1, 100)
        eixo_y = np.linspace(y_min - 1, y_max + 1, 100)
        X, Y = np.meshgrid(eixo_x, eixo_y)
        Z = f_grid(X, Y)

        #Contorno com escala (viridis) em amarelo (alto) - roxo (baixo)
        contour = plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.clabel(contour, inline=True, fontsize=8)
        #Legenda para as cores
        plt.colorbar(label="f(x, y)")

        #Pegar pedaços do caminho guardado em path e plotar
        plt.plot(path[:, 0], path[:, 1], 'o-', color='red', label="Caminho")
        #Marcar o ponto inicial, chute inicial x0
        plt.scatter(x0[0], x0[1], color='blue', label='Ponto inicial', zorder=5)

        plt.title("Minimização pelo Método de Newton")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    return x, k

##### Busca Linear 

def linesearch(f, x, g, d):
    a = 1
    tau = 1e-3
    gamma = 0.5
    while f(x+ a*d) > f(x) + tau*a*np.dot(g, d):
        a *= gamma
    return a