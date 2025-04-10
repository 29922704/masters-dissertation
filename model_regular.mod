#Set definitions:
set N; 
set K;
set V = 0 .. 1 by 1;
set W; 

#Parameter definitions:
param c_i {i in N};
param r_i {i in N};
param q_k {k in K};
param d_ik {i in N, k in K};
param lambda_i {i in N};
param Omega_ivw {i in N, v in V, w in W};
param eta_i {i in N};
param alpha_hik {h in N, i in N, k in K};
param L_ivw {i in N, v in V, w in W}; 
param U_ivw {i in N, v in V, w in W}; 
param M = 9999;
param Q = 0.1;

#Variable declarations:
var gamma_ik {N,K} binary;
var phi_i {N};
var O_i {N};
var omega_ivw {N, V, W};
var x_ivw {N, V, W} binary;
var y_ivw {N, V, W} binary;
var p_i {N};

#Objective function:
maximize Total_Profit: 
  sum {i in N}(
    sum{k in K} (q_k[k] * r_i[i] * gamma_ik[i,k] * (d_ik[i,k] + phi_i[i]))
	+ lambda_i[i] * c_i[i] * (O_i[i] - phi_i[i] - sum{k in K} gamma_ik[i,k] * (d_ik[i,k] + phi_i[i]))
	+ c_i[i] * sum{w in W}(
	  O_i[i] * omega_ivw[i,0,w] + 
	  omega_ivw[i,1,w] * sum{k in K} gamma_ik[i,k] * (d_ik[i,k] + phi_i[i]))
	- c_i[i] * O_i[i]
);


#Constraints:
#Single price assignment constraint
subj to C1 {i in N}: sum{k in K} gamma_ik[i,k] = 1;
#Cross-promotional sales adjustment assignment constraint
subj to C2 {i in N}:  phi_i[i] = sum {h in N} sum {k in K} alpha_hik[h,i,k]*d_ik[h,k]*gamma_ik[h,k]; 
#Order quantity assignment constraint
subj to C3 {i in N}: O_i[i] = eta_i[i]*(sum{k in K} gamma_ik[i,k]*(d_ik[i,k] + phi_i[i]));
#Trade spending eligibility constraints
subj to C4 {i in N, w in W}: O_i[i] >= L_ivw[i,0,w] - M*(1-x_ivw[i,0,w]);
subj to C5 {i in N, w in W}: O_i[i] + Q <= L_ivw[i,0,w] + M*x_ivw[i,0,w];
subj to C6 {i in N, w in W}: O_i[i] <= U_ivw[i,0,w] + M*(1-y_ivw[i,0,w]);
subj to C7 {i in N, w in W}: O_i[i] >= U_ivw[i,0,w] - M*y_ivw[i,0,w] + Q;

subj to C12 {i in N,w in W}: sum{k in K} gamma_ik[i,k]*(d_ik[i,k]+phi_i[i]) >= L_ivw[i,1,w] - M*(1-x_ivw[i,1,w]);
subj to C13 {i in N,w in W}: sum{k in K} gamma_ik[i,k]*(d_ik[i,k]+phi_i[i]) + Q <= L_ivw[i,1,w] + M*x_ivw[i,1,w];
subj to C14 {i in N,w in W}: sum{k in K} gamma_ik[i,k]*(d_ik[i,k]+phi_i[i]) <= U_ivw[i,1,w] + M*(1-y_ivw[i,1,w]);
subj to C15 {i in N,w in W}: sum{k in K} gamma_ik[i,k]*(d_ik[i,k]+phi_i[i]) >= U_ivw[i,1,w] - M*y_ivw[i,1,w] +Q;
#Trade spending rate assignment constraints
subj to C8 {i in N, v in V, w in W}: Omega_ivw[i,v,w] -M*(2-x_ivw[i,v,w] - y_ivw[i,v,w]) <= omega_ivw[i,v,w];
subj to C9 {i in N, v in V, w in W}: omega_ivw[i,v,w] <= Omega_ivw[i,v,w] + M*(2 - x_ivw[i,v,w] - y_ivw[i,v,w]);
subj to C10 {i in N, v in V, w in W}: -x_ivw[i,v,w]*M <= omega_ivw[i,v,w];
subj to C10_2 {i in N, v in V, w in W}: omega_ivw[i,v,w] <= x_ivw[i,v,w]*M;
subj to C11 {i in N, v in V, w in W}: -y_ivw[i,v,w]*M <= omega_ivw[i,v,w];
subj to C11_2 {i in N, v in V, w in W}: omega_ivw[i,v,w] <= y_ivw[i,v,w]*M;
#Item price calculation constraint
subj to AUX1 {i in N}: p_i[i] = sum{k in K} q_k[k] * r_i[i] * gamma_ik[i,k]; 

#Constraint to force sale at regular price
subj to AUX2 {i in N}: gamma_ik[i,0] = 1;

