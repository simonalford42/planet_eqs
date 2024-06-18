'''
Baseline calculation from https://arxiv.org/pdf/2403.17928 eq. 6
'''
import numpy as np

A = 11.9
B = -7.67
C = 5.20
D = -3.26

def instability_time(P_1, m_i, M_plus, e_i, e_cross, a_i_plus_1, a_i, M_star):
    return (A + B * (e_i / e_cross)) * np.log10( ((a_i_plus_1 - a_i) / (a_i_plus_1 + a_i)) * (M_star / m_i) ** 0.25) + C + D * (e_i / e_cross) - np.log10(m_i) + np.log10(P_1) + np.log10(M_plus)

