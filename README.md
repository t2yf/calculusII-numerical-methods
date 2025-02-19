# 📚 | Cálculo II - Métodos Númericos
## Sobre o repositório
Esse repositório contém os projetos realizados durante a disciplina de Cálculo II Honors. O objetivo é ir além da teoria e observar na prática o funcionamento de alguns métodos numéricos utilizando python e bibliotecas como o numpy e o matplotlib.



---
### Projeto 1: Integração numérica
Esse projeto consiste na implementação de métodos para a aproximação da integral pelo método numérico, no caso, a **Regra do Trapézio** e a **Regra de Simpson**.

---

### Projeto 2: Diferenças finitas 
Esse projeto consiste na implementação de duas funções, uma para resolver numericamente uma EDO (equação diferencial ordinária) com a condição inicial na forma y'= f(x,y), y(x0) = y0 e outra para analisar como a escolha do "passo" impacta no erro das **diferenças finitas avançadas**, assim calculando o erro entre a derivada por diferenças finitas avançadas e a derivada real. Ambos geram gráficos no matplotlib, segue exemplos:



**Ode Solver**

Input: ode solver(lambda x,y: 10*np.sqrt(y)*np.sin(x)+x,0,0,100,500,True 

![ode_solver](https://github.com/user-attachments/assets/ea33e992-7889-4973-bcf8-999ec85d1917)

**Erro nas Diferenças finitas**

Input: fd error(lambda x: np.atan(x),lambda x:1/(1+x**2),1,1e-15,1e-1,100)

![fd_error](https://github.com/user-attachments/assets/d07fcc00-c73e-423a-9104-79cb7c019f3f)

---

### Projeto 3: Otimização
Esse projeto tem o objetivo de implementar vários métodos para otimização e outros auxiliares. Os métodos implementados foram:
  - **Derivada Parcial com Diferenças Finitas Centradas**
  - **Busca Linear*
  - **Método do Gradiente**
  - **Método de Newton**
  - Salvaguardas para o Método de Newton
    - Matriz singular
    - Manter a direção de descida
Segue exemplo:


Input: 

def f(x):
    return np.log(np.exp(-x[0])) + np.exp(-x[1]) + np.exp(x[0]+x[1])

def f_grid(X, Y):
    return np.log(np.exp(-X)) + np.exp(-Y) + np.exp(X + Y)

def hess(x):
    return np.array([[np.exp(x[0]+x[1]), np.exp(x[0]+x[1])],[np.exp(x[0]+x[1]),np.exp(-x[1])+np.exp(x[0]+x[1])]])

def grad(x):
    return np.array([ -1 + np.exp(x[0]+x[1]), -np.exp(-x[1]) + np.exp(x[0] + x[1]) ])

x,k = gd(f,np.array([2.0,-0.5]),grad,1e-5, 0.1, 10000,  True, 1e-5,  True, True)
print(f"x = {x}")
print(f"k = {k}")

x,k = newton(f,np.array([2,-0.5]),grad,hess,1e-5, 0.1, 10000, True, 1e-5,  True, True)
print(f"x = {x}")
print(f"k = {k}")

**Minimização pelo Método do Gradiente**

![metodo_gradiente](https://github.com/user-attachments/assets/4e0132f0-363d-48ac-b018-41fb9d68f311)


**Minimização pelo Método de Newton**

![metodo_newton](https://github.com/user-attachments/assets/5e334bfb-ac09-4769-ad0a-827213fea587)


---
