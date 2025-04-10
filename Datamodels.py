import random
import numpy as np
import copy

class DataModel:
    def __init__(self, num_items = 3, num_prices = 3, num_rates = 3):
        self.c_i = [] #1d array
        self.r_i = [] #1d array
        self.q_k = [] #1d array
        self.d_ik = [] #2d array
        self.alpha_hik = [] #3d array
        self.Omega_ivw = [] #3d array
        self.L_ivw = [] #3d array
        self.U_ivw = [] #3d array
        self.lambda_i = [] #1d array
        self.eta_i = [] #1d array

        self.N = range(num_items)
        self.K = range(num_prices)
        self.W = range(num_rates)  

        self.a_i = [] #price sensitivity parameter

        self.m = 2

        self.cost_modifiers = []

    def clone(self):
        return copy.deepcopy(self)

    def get_random_cost_modifiers(self):
        arr = []
        for i in self.N:
            arr.append(random.uniform(-0.15, 0.15))
        return arr

    def update_alpha_hik(self, m_alt = None):
        if m_alt is not None:
            self.m = m_alt
        self.gen_alpha_hik(self.m)  

    def update_demand(self, a_i_alt = None):
        if a_i_alt is not None:
            self.a_i = a_i_alt
        self.gen_d_ik()
        

    def gen_DataModel(self, omega_max=0.15, eta=1.05, lambda_max=0.2, l=0, u=0.5, profit_margin=0.4, 
                      a_i = None, c_i = None, r_i = None, Omega_ivw = None, lambda_i = None, eta_i = None, m = None,
                      L_ivw = None, U_ivw = None, cost_modifiers = None):
        print("Generating Data Model...")

        if cost_modifiers is not None:
            self.cost_modifiers = cost_modifiers
        else: 
            self.cost_modifiers = self.get_random_cost_modifiers()

        if m is not None:
            self.m = m

        if a_i is not None:
            self.a_i = a_i
        else:
            self.gen_a_i()
        
        if c_i is not None:
            self.c_i = c_i
        else:
            self.gen_c_i()

        if r_i is not None:
            self.r_i = r_i
        else:
            self.gen_r_i(profit_margin)

        self.gen_q_k()
        self.gen_d_ik()
        self.gen_alpha_hik(self.m) # m = cross-item sensitivity parameter

        if Omega_ivw is not None:
            self.Omega_ivw = Omega_ivw
        else:
            self.gen_Omega_ivw(omega_max) #omega_max = max price discount, each Omega_ivw is uniformly distributed between 0 and omega_max 
        
        if lambda_i is not None:
            self.lambda_i = lambda_i
        else:
            self.gen_lambda_i(lambda_max) #lambda_i is random uniformly distributed between 0 and lambda_max
        
        if eta_i is not None:
            self.eta_i = eta_i
        else:
            self.gen_eta_i(eta) #eta_i is eta

        if L_ivw and U_ivw is not None:
            self.L_ivw = L_ivw
            self.U_ivw = U_ivw
        else:
            self.gen_L_U_ivw(l, u)

        print("Data Model generated successfully.\n")


    def gen_a_i(self):
        print("Generating a_i...")
        for i in self.N:
            value = random.uniform(0.1, 0.5)
            self.a_i.append(value)
            print(f'a_{i} = {value}')
        print('a_i generated successfully.\n')
    

    def gen_c_i(self):
        print("Generating c_i...")
        if not self.a_i:
            raise ValueError("a_i must be initialized using gen_a_i before calling gen_c_i.")
        for i in self.N:
            value = ((np.log(1/10) + 8) / self.a_i[i]) + self.cost_modifiers[i] * ((np.log(1/10) + 8) / self.a_i[i])
            self.c_i.append(value)
            print(f'c_{i} = {value}')
        print('c_i generated successfully.\n')


    def gen_r_i(self, profit_margin):
        print("Generating r_i...")
        if not self.c_i:
            raise ValueError("c_i must be initialized using gen_c_i before calling gen_r_i.")
        for i in self.N:
            value = (1 + profit_margin) * self.c_i[i]
            self.r_i.append(value)
            print(f'r_{i} = {value}')
        print('r_i generated successfully.\n')


    def gen_q_k(self):
        print("Generating q_k...")
        for k in self.K:
            value = 1 - 0.1 * k
            self.q_k.append(value)
            print(f'q_{k} = {value}')
        print('q_k generated successfully.\n')


    def calculate_demand(self, prices, a):
        return 10 / (0.1 + np.exp(a * prices - 8)) 


    def gen_d_ik(self):
        print("Generating d_ik...")
        if not self.r_i or not self.q_k:
            raise ValueError("r_i and q_k must be initialized using gen_r_i and gen_q_k before calling gen_d_ik.")

        self.d_ik = []

        for i in self.N:
            row = []
            for k in self.K:
                p_ik = self.r_i[i] * self.q_k[k]
                demand = self.calculate_demand(p_ik, self.a_i[i])
                row.append(demand)
                print(f'd_{i}{k} = {demand}')
            self.d_ik.append(row) 
        print('d_ik generated successfully.\n')


    def gen_alpha_hik(self, m):
        print ("Generating alpha_hik...")
        if not self.d_ik:
            raise ValueError("d_ik must be initialized using gen_d_ik before calling gen_alpha_hik.")
        
        self.alpha_hik = []
        for h in self.N:
            alpha_h = []
            for i in self.N:
                alpha_h_i = []
                for k in self.K:
                    if h == i:
                        alpha_value = 0
                    else:
                        alpha_value = ((self.d_ik[h][0] - self.d_ik[h][k]) / (m * self.d_ik[h][k]))
                    alpha_h_i.append(alpha_value)
                    print(f'alpha_{h}{i}{k} = {alpha_value}')
                alpha_h.append(alpha_h_i)
            self.alpha_hik.append(alpha_h)
            
        print('alpha_hik generated successfully.\n')


    def gen_Omega_ivw(self, omega_max):
        print("Generating Omega_ivw...")
        for i in self.N:
            Omega_i = []
            for v in range(2):
                Omega_v = []
                for w in self.W:
                    if w == 0:
                        value = 0
                    elif w == len(self.W) - 1:
                        value = omega_max
                    else:
                        value = (w / (len(self.W) - 1)) * omega_max
                    Omega_v.append(value)
                    print(f'Omega_{i}{v}{w} = {value}')
                Omega_i.append(Omega_v)
            self.Omega_ivw.append(Omega_i)

        print('Omega_ivw generated successfully.\n')

    def gen_eta_i(self, eta):
        print("Generating eta_i...")
        for i in self.N:
            self.eta_i.append(eta)
            print(f'eta_{i} = {eta}')
        print('eta_i generated successfully.\n')

    def gen_lambda_i(self, lambda_max):
        print("Generating lambda_i...")
        for i in self.N:
            value = random.uniform(0, lambda_max)
            self.lambda_i.append(value)
            print(f'lambda_{i} = {value}')

    def gen_L_U_ivw(self, l, u, modifier = None):
        print("Generating L_ivw and U_ivw...")

        Q = 0.001
        M = 8000
        num_intervals = len(self.W)
        points = [0,0.5,1.5,2]
        #points = np.linspace(l, u, num_intervals + 1)

        # Initialize L_ivw and U_ivw as 3D lists with correct dimensions
        self.L_ivw = [[[0 for _ in self.W] for _ in range(2)] for _ in self.N]
        self.U_ivw = [[[0 for _ in self.W] for _ in range(2)] for _ in self.N]

        for i in self.N:
            for v in range(2):
                for w in self.W:
                    L = points[w]
                    U = points[w + 1] - Q

                    if v == 0:
                        value_l = (1 + L) * self.eta_i[i] * self.d_ik[i][0]
                        
                        if w == self.W[-1]:
                            value_u = M
                        else:
                            value_u = (1 + U) * self.eta_i[i] * self.d_ik[i][0]
                    
                    else:
                        value_l = (1 + L) * self.d_ik[i][0]
                        if w == self.W[-1]:
                            value_u = M
                        else:
                            value_u = (1 + U) * self.d_ik[i][0]

                    if modifier is not None:
                        self.L_ivw[i][v][w] = (1 + modifier) * value_l
                        self.U_ivw[i][v][w] = (1 + modifier) * value_u
                    else:
                        self.L_ivw[i][v][w] = value_l
                        self.U_ivw[i][v][w] = value_u

                    print(f"L_ivw for i={i}, v={v}, w={w}: {self.L_ivw[i][v][w]}")
                    print(f"U_ivw for i={i}, v={v}, w={w}: {self.U_ivw[i][v][w]}")
            