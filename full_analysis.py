import numpy as np
import matplotlib.pyplot as plt
import random as ran
import pandas as pd
import statsmodels.tsa.seasonal as sm
from scipy.interpolate import CubicSpline
import numpy.ma as ma
#%%
#t, u = data from file
#range = t[-1] - t[0]
#spacing = t[1] - t[0]
#tau = sp.arange(0, range, spacing)
def R(t, u): #autocorrelation
    R = np.zeros(len(t) - 1)
    u_bar = np.mean(u)
    uprime = u - u_bar
    var = np.var(uprime)
    for i in range(0, len(t) - 1):
        correlation = np.zeros(len(t) - i)
        for j in range(0, len(t) - i):
            correlation[j] = uprime[j] * uprime[i + j]
        R[i] = np.mean(correlation) / var
    return R
#%%


filename_w = 'StormDennis_OverWater2.csv'
filename_g = 'StormDennis_OverGround2.csv'

data_w = pd.read_csv(filename_w)
data_g = pd.read_csv(filename_g)

T1u_w = np.add(data_w['348 [°C]'].to_numpy(), 273.15)
T2u_w = np.add(data_w['261 [°C]'].to_numpy(), 273.15)
T3u_w = np.add(data_w['789 [°C]'].to_numpy(), 273.15)

T1w_w = np.add(data_w['154 [°C]'].to_numpy(), 273.15)
T2w_w = np.add(data_w['590 [°C]'].to_numpy(), 273.15)
T3w_w = np.add(data_w['150 [°C]'].to_numpy(), 273.15)

T1_w = np.zeros_like(T1u_w)
T2_w = np.zeros_like(T2u_w)
T3_w = np.zeros_like(T3u_w)

T1u_g = np.add(data_g['348 [°C]'].to_numpy(), 273.15)
T2u_g = np.add(data_g['261 [°C]'].to_numpy(), 273.15)
T3u_g = np.add(data_g['789 [°C]'].to_numpy(), 273.15)

T1w_g = np.add(data_g['154 [°C]'].to_numpy(), 273.15)
T2w_g = np.add(data_g['590 [°C]'].to_numpy(), 273.15)
T3w_g = np.add(data_g['150 [°C]'].to_numpy(), 273.15)

T1_g = np.zeros_like(T1u_g)
T2_g = np.zeros_like(T2u_g)
T3_g = np.zeros_like(T3u_g)

for i in range(len(T1_w)):
    T1_w[i] = np.mean([T1u_w[i],T1w_w[i]])
    T2_w[i] = np.mean([T2u_w[i],T2w_w[i]])
    T3_w[i] = np.mean([T3u_w[i],T3w_w[i]])

for i in range(len(T1_g)):
    T1_g[i] = np.mean([T1u_g[i],T1w_g[i]])
    T2_g[i] = np.mean([T2u_g[i],T2w_g[i]])
    T3_g[i] = np.mean([T3u_g[i],T3w_g[i]])

u1_w = data_w['348 [m/s]'].to_numpy()
u2_w = data_w['261 [m/s]'].to_numpy()
u3_w = data_w['789 [m/s]'].to_numpy()

w1_w = data_w['154 [m/s]'].to_numpy()
w2_w = data_w['590 [m/s]'].to_numpy()
w3_w = data_w['150 [m/s]'].to_numpy()

u1_g = data_g['348 [m/s]'].to_numpy()
u2_g = data_g['261 [m/s]'].to_numpy()
u3_g = data_g['789 [m/s]'].to_numpy()

w1_g = data_g['154 [m/s]'].to_numpy()
w2_g = data_g['590 [m/s]'].to_numpy()
w3_g = data_g['150 [m/s]'].to_numpy()

t_w = np.arange(0, (len(u1_w) - 1) * 2 + 0.01, 2)
t_g = np.arange(0, (len(u1_g) - 1) * 2 + 0.01, 2)
"""
autocorr3 = R(t, w3 - np.mean(w3))
plt.plot(t[:-1], autocorr3)
"""
"""
plt.plot(t, u1, 'r-')
plt.plot(t, u2, 'b-')
plt.plot(t, u3, 'g-')
plt.show()
"""

decompose1_w = sm.seasonal_decompose(u1_w, model = 'additive', freq = 1000)
decompose2_w = sm.seasonal_decompose(u2_w, model = 'additive', freq = 1000)
decompose3_w = sm.seasonal_decompose(u3_w, model = 'additive', freq = 1000)
turb1_w = np.sqrt((u1_w - decompose1_w.trend)**2 + w1_w**2)
turb2_w = np.sqrt((u2_w - decompose2_w.trend)**2 + w2_w**2)
turb3_w = np.sqrt((u3_w - decompose3_w.trend)**2 + w3_w**2)

decompose1_g = sm.seasonal_decompose(u1_g, model = 'additive', freq = 1000)
decompose2_g = sm.seasonal_decompose(u2_g, model = 'additive', freq = 1000)
decompose3_g = sm.seasonal_decompose(u3_g, model = 'additive', freq = 1000)
turb1_g = np.sqrt((u1_g - decompose1_g.trend)**2 + w1_g**2)
turb2_g = np.sqrt((u2_g - decompose2_g.trend)**2 + w2_g**2)
turb3_g = np.sqrt((u3_g - decompose3_g.trend)**2 + w3_g**2)

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(t_w, turb1_w, 'r-')
ax1.plot(t_w, turb2_w, 'b-')
ax1.plot(t_w, turb3_w, 'g-')

ax2.plot(t_g, turb1_g, 'r-')
ax2.plot(t_g, turb2_g, 'b-')
ax2.plot(t_g, turb3_g, 'g-')
#%%
#calculate scaling and error estimates for each sensor
#order of sensors:
#590, 150, 154, 261, 789, 348
#setting 1
means1 = np.array([7.317, 5.362, 6.452, 6.662, 6.891, 7.207])
errors1 = np.array([0.331, 0.196, 0.214, 0.547, 0.345, 0.624])
#setting 2
means2 = np.array([8.657, 6.165, 6.853, 8.307, 7.162, 8.22])
errors2 = np.array([0.448, 0.184, 0.272, 0.341, 0.5, 0.432])
#setting 3
means3 = np.array([9.466, 6.721, 8.191, 9.371, 8.722, 9.486])
errors3 = np.array([0.521, 0.299, 0.399, 0.408, 0.499, 0.418])
#other fan
means4 = np.array([3.818, 2.621, 3.055, 3.42, 3.636, 3.547])
errors4 = np.array([0.146, 0.141, 0.189, 0.243, 0.149, 0.187])

#estimate 'true' windspeed for each fan setting
true1 = np.mean(means1)
error1 = np.sqrt(np.sum(errors1 ** 2) / 6)

true2 = np.mean(means2)
error2 = np.sqrt(np.sum(errors2 ** 2) / 6)

true3 = np.mean(means3)
error3 = np.sqrt(np.sum(errors3 ** 2) / 6)

true4 = np.mean(means4)
error4 = np.sqrt(np.sum(errors4 ** 2) / 6)

#calculate scaling factors for each sensor at each setting
scalings1 = true1 / means1
scalingerror1 = np.sqrt((true1 * errors1 / (means1 ** 2)) ** 2 + (error1 / means1) ** 2)
scalings2 = true2 / means2
scalingerror2 = np.sqrt((true2 * errors2 / (means2 ** 2)) ** 2 + (error2 / means2) ** 2)
scalings3 = true3 / means3
scalingerror3 = np.sqrt((true3 * errors3 / (means3 ** 2)) ** 2 + (error3 / means3) ** 2)
scalings4 = true4 / means4
scalingerror4 = np.sqrt((true4 * errors4 / (means4 ** 2)) ** 2 + (error4 / means4) ** 2)

#calculate a final scaling to use for each sensor
weights1 = 1 / scalingerror1 ** 2
weights2 = 1 / scalingerror2 ** 2
weights3 = 1 / scalingerror3 ** 2
weights4 = 1 / scalingerror4 ** 2
normalise = weights1 + weights2 + weights3 + weights4
scalingfinal = (scalings1 * weights1 + scalings2 * weights2 + scalings3 * weights3 + scalings4 * weights4) / normalise

#find error relationship with windspeed
from scipy.optimize import curve_fit

def straight_fit(x, m):
    return m * x

grads = np.zeros(6)
gradserr = np.zeros(6)
x = np.linspace(0, 10, 101)
for i in range(0, 6):
    speeds = np.array([means1[i], means2[i], means3[i], means4[i]])
    errors = np.array([errors1[i], errors2[i], errors3[i], errors4[i]])
    grads[i], gradserr[i] = curve_fit(straight_fit, speeds, errors, p0 = 0.05)
    plt.plot(speeds, errors, 'kx')
    plt.plot(x, straight_fit(x, grads[i]), 'r-')
    plt.xlabel('Measured Wind Speed (m/s)')
    plt.ylabel('Estimated Error (m/s)')
    plt.grid()
    plt.show()
gradserr = np.sqrt(gradserr)
#%%
from matplotlib.ticker import AutoMinorLocator

x = np.linspace(0, 10, 101)

fig, ax = plt.subplots()

ax.plot(x, straight_fit(x, grads[0]), color='forestgreen', ls='-', label='348')
ax.plot(x, straight_fit(x, grads[1]), color='cornflowerblue', ls='-', label='789')
ax.plot(x, straight_fit(x, grads[4]), color='crimson', ls='-', label='261')

ax.plot(np.array([means1[0], means2[0], means3[0], means4[0]]), np.array([errors1[0], errors2[0], errors3[0], errors4[0]]), color='forestgreen', linestyle='', marker='^', ms=8)
ax.plot(np.array([means1[1], means2[1], means3[1], means4[1]]), np.array([errors1[1], errors2[1], errors3[1], errors4[1]]), color='cornflowerblue', linestyle='', marker='s', ms=8)
ax.plot(np.array([means1[4], means2[4], means3[4], means4[4]]), np.array([errors1[4], errors2[4], errors3[4], errors4[4]]), color='crimson', linestyle='', marker='o', ms=8)

ax.set_xlim(0, 10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Mean Wind Speed (m/s)', fontsize=15)
ax.set_ylabel('Statistical Error', fontsize=15)
ax.tick_params(axis='both', labelsize=13)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, which='major')
ax.legend(fontsize=13)
fig.show()
#%%
u1_w *= scalingfinal[5]
u2_w *= scalingfinal[3]
u3_w *= scalingfinal[4]
w1_w *= scalingfinal[2]
w2_w *= scalingfinal[0]
w3_w *= scalingfinal[1]

u1err_w = u1_w * grads[5]
u2err_w = u2_w * grads[3]
u3err_w = u3_w * grads[4]
w1err_w = w1_w * grads[2]
w2err_w = w2_w * grads[0]
w3err_w = w3_w * grads[1]

u1_g *= scalingfinal[5]
u2_g *= scalingfinal[3]
u3_g *= scalingfinal[4]
w1_g *= scalingfinal[2]
w2_g *= scalingfinal[0]
w3_g *= scalingfinal[1]

u1err_g = u1_g * grads[5]
u2err_g = u2_g * grads[3]
u3err_g = u3_g * grads[4]
w1err_g = w1_g * grads[2]
w2err_g = w2_g * grads[0]
w3err_g = w3_g * grads[1]
#%%
#Richardson Number
Ri_w = np.zeros(len(u2_w))
weight = np.ones(len(Ri_w))


for i in range(len(Ri_w)):
    du_w = (u3_w[i] - u1_w[i])/2 
    dT_w = (T3_w[i] - T1_w[i])/2
    if abs(du_w) < 0.0001:
        du_w = np.nan
    Ri_w[i] = (10/T2_w[i])*((dT_w)/(du_w)**2)
    
    if np.isnan(Ri_w[i]):
        weight[i] = 0

Ri_w_ = np.where(np.isnan(Ri_w), None, Ri_w)
Ri_w_ = ma.masked_where(Ri_w_ == None, Ri_w_)
Ri_w__ = ma.compressed(Ri_w_)

R_w_num = []
for i in range(len(Ri_w__)):
    if abs(Ri_w__[i]) < 1:
        R_w_num.append(Ri_w__[i])

Ri_g = np.zeros(len(u2_g))
weight = np.ones(len(Ri_g))


for i in range(len(Ri_g)):
    du_g = (u3_g[i] - u1_g[i])/2 
    dT_g = (T3_g[i] - T1_g[i])/2
    if abs(du_g) < 0.0001:
        du_g = np.nan
    Ri_g[i] = (10/T2_g[i])*((dT_g)/(du_g)**2)
    
    if np.isnan(Ri_g[i]):
        weight[i] = 0

Ri_g_ = np.where(np.isnan(Ri_g), None, Ri_g)
Ri_g_ = ma.masked_where(Ri_g_ == None, Ri_g_)
Ri_g__ = ma.compressed(Ri_g_)

R_g_num = []
for i in range(len(Ri_g__)):
    if abs(Ri_g__[i]) < 1:
        R_g_num.append(Ri_g__[i])

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.hist(R_w_num, bins=500, density=False, ec='k', fc='cornflowerblue')
ax1.set_xlim(-0.1, 0.005)
ax1.tick_params(labelsize=13)

ax2.hist(R_g_num, bins=500, density=False, ec='k', fc='forestgreen')
ax2.set_xlim(-0.1, 0.005)
ax2.set_xlabel('Gradient Richardson Number', fontsize=20)
ax2.tick_params(labelsize=13)

fig.text(0.05, 0.5, 'Frequency', rotation=90, fontsize=20)
fig.show()
#%%
#error stuff

filename1 = 'ResponseTime.csv'
response = pd.read_csv(filename1)

u_lab = response['348 [m/s]'].to_numpy()
t_lab = np.arange(0, (len(u_lab) - 1)*2 + 0.01, 2)

avg1 = np.ones(18)*7.207
avg2 = np.ones(20)*8.220
avg3 = np.ones(20)*9.486

times = np.linspace(0, 180, 2000)
spl = CubicSpline(t_lab, u_lab)

#plt.plot(t_lab, u_lab, 'x')
plt.plot(times, spl(times), color='cornflowerblue', lw=2)
plt.plot(t_lab[14:32], avg1, color = 'k', linestyle='--', lw=2)
plt.plot(t_lab[35:55], avg2, color = 'k', linestyle='--', lw=2)
plt.plot(t_lab[56:76], avg3, color = 'k', linestyle='--', lw=2)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Time Elapsed in Seconds', fontsize=15)
plt.ylabel('Horizontal Velocity', fontsize=15)
#%%

def decomposer(u, N):
    ubar = np.zeros(len(u) - N)
    uprime = np.zeros(len(u) - N)
    for i in range(0, len(ubar)):
        ubar[i] = np.mean(u[i : i + N])
        uprime[i] = u[i + 1 + int(N / 2)] - ubar[i]
    return ubar, uprime

def running_mean_turbulence(u, ubar):
    N = len(u) - len(ubar)
    mean = np.zeros(len(ubar))
    for i in range(0, len(mean)):
        mean[i] = np.sqrt(np.mean((u[i : i + N] - ubar[i]) ** 2))
    return mean

uprime1_w = u1_w - np.mean(u1_w)
uprime2_w = u2_w - np.mean(u2_w)
uprime3_w = u3_w - np.mean(u3_w)

uprime1_g = u1_g - np.mean(u1_g)
uprime2_g = u2_g - np.mean(u2_g)
uprime3_g = u3_g - np.mean(u3_g)

autocorr1_w = R(t_w, uprime1_w)
autocorr2_w = R(t_w, uprime2_w)
autocorr3_w = R(t_w, uprime3_w)

autocorr1_g = R(t_g, uprime1_g)
autocorr2_g = R(t_g, uprime2_g)
autocorr3_g = R(t_g, uprime3_g)

def exp_fit(tau, decay):
    return np.exp(-tau / decay)

autocorrlist_w = [autocorr1_w, autocorr2_w, autocorr3_w]
autocorrlist_g = [autocorr1_g, autocorr2_g, autocorr3_g]
decaytimes_w = np.zeros(3)
decaytimeserr_w = np.zeros(3)
decaytimes_g = np.zeros(3)
decaytimeserr_g = np.zeros(3)

for i in range(0, 3):
    decaytimes_w[i], decaytimeserr_w[i] = curve_fit(exp_fit, t_w[:-1], autocorrlist_w[i], p0 = 50)

for i in range(0, 3):
    decaytimes_g[i], decaytimeserr_g[i] = curve_fit(exp_fit, t_g[:-1], autocorrlist_g[i], p0 = 50)

decaytimeserr_w = np.sqrt(decaytimeserr_w)
decaytimeserr_g = np.sqrt(decaytimeserr_g)

N_w = int(10 * max(decaytimes_w))
N_g = int(35 * max(decaytimes_g))

if N_w % 2 == 0:
    N_w += 1

if N_g % 2 == 0:
    N_g += 1

u1bar_w, u1prime_w = decomposer(u1_w, N_w)
u2bar_w, u2prime_w = decomposer(u2_w, N_w)
u3bar_w, u3prime_w = decomposer(u3_w, N_w)
w1bar_w, w1prime_w = decomposer(w1_w, N_w)
w2bar_w, w2prime_w = decomposer(w2_w, N_w)
w3bar_w, w3prime_w = decomposer(w3_w, N_w)

u1bar_g, u1prime_g = decomposer(u1_g, N_g)
u2bar_g, u2prime_g = decomposer(u2_g, N_g)
u3bar_g, u3prime_g = decomposer(u3_g, N_g)
w1bar_g, w1prime_g = decomposer(w1_g, N_g)
w2bar_g, w2prime_g = decomposer(w2_g, N_g)
w3bar_g, w3prime_g = decomposer(w3_g, N_g)

t2_w = np.arange(N_w - 1, N_w - 0.99 + 2 * (len(u1bar_w) - 1), 2)
t2_g = np.arange(N_g - 1, N_g - 0.99 + 2 * (len(u1bar_g) - 1), 2)

u1barerr_w = u1bar_w * grads[5]
u2barerr_w = u2bar_w * grads[3]
u3barerr_w = u3bar_w * grads[4]

u1barerr_g = u1bar_g * grads[5]
u2barerr_g = u2bar_g * grads[3]
u3barerr_g = u3bar_g * grads[4]

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(t2_w, u1bar_w, color='crimson', ls='-', label = '0.75m')
ax1.plot(t2_w, u2bar_w, color='cornflowerblue', ls='-', label = '1.75m')
ax1.plot(t2_w, u3bar_w, color='forestgreen', ls='-', label = '2.75m')
ax1.plot(t2_w, u1bar_w - u1barerr_w, color='crimson', ls='--', alpha = 0.5)
ax1.plot(t2_w, u2bar_w - u2barerr_w, color='cornflowerblue', ls='--', alpha = 0.5)
ax1.plot(t2_w, u3bar_w - u3barerr_w, color='forestgreen', ls='--', alpha = 0.5)
ax1.plot(t2_w, u1bar_w + u1barerr_w, color='crimson', ls='--', alpha = 0.5)
ax1.plot(t2_w, u2bar_w + u2barerr_w, color='cornflowerblue', ls='--', alpha = 0.5)
ax1.plot(t2_w, u3bar_w + u3barerr_w, color='forestgreen', ls='--', alpha = 0.5)
ax1.set_ylabel('Wind Speed (m/s)', fontsize=15)
ax1.tick_params(labelsize=13)
ax1.grid()
ax1.legend()

ax2.plot(t2_g, u1bar_g, color='crimson', ls='-', label = '0.75m')
ax2.plot(t2_g, u2bar_g, color='cornflowerblue', ls='-', label = '1.75m')
ax2.plot(t2_g, u3bar_g, color='forestgreen', ls='-', label = '2.75m')
ax2.plot(t2_g, u1bar_g - u1barerr_g, color='crimson', ls='--', alpha = 0.5)
ax2.plot(t2_g, u2bar_g - u2barerr_g, color='cornflowerblue', ls='--', alpha = 0.5)
ax2.plot(t2_g, u3bar_g - u3barerr_g, color='forestgreen', ls='--', alpha = 0.5)
ax2.plot(t2_g, u1bar_g + u1barerr_g, color='crimson', ls='--', alpha = 0.5)
ax2.plot(t2_g, u2bar_g + u2barerr_g, color='cornflowerblue', ls='--', alpha = 0.5)
ax2.plot(t2_g, u3bar_g + u3barerr_g, color='forestgreen', ls='--', alpha = 0.5)
ax2.tick_params(labelsize=13)
ax2.grid()
ax2.legend()

fig.text(0.47, 0.04, 'Time in Seconds', fontsize=15)
fig.show()

#%%

fig, ax = plt.subplots(1)

T = np.linspace(0, max(t_g), 100000)
spl = CubicSpline(t_g[:-1], autocorrlist_g[1])
ax.plot(T, spl(T), color='cornflowerblue', label='Autocorrelation')
#ax.plot(t_g[:-1], autocorrlist_g[1], 'o', ms=2)
ax.plot(T, exp_fit(T, decaytimes_g[1]), lw=2, ls='--', color='k', alpha=0.75, label='Exponential Approximation')
ax.set_xlim(-25,250)
ax.set_ylim(-0.1, 1.1)
ax.set_ylabel(r'$R(\tau)$', fontsize=15)
ax.set_xlabel(r'$\tau$', fontsize = 15)
ax.tick_params(labelsize=13)
ax.grid()
ax.legend(fontsize=13)
fig.show()

#%%

from scipy.stats import chi2

t2_w = np.arange(N_w - 1, N_w - 0.99 + 2 * (len(u1bar_w) - 1), 2)
t2_g = np.arange(N_g - 1, N_g - 0.99 + 2 * (len(u1bar_g) - 1), 2)

u1primemean_w = running_mean_turbulence(u1_w, u1bar_w)
u2primemean_w = running_mean_turbulence(u2_w, u2bar_w)
u3primemean_w = running_mean_turbulence(u3_w, u3bar_w)
w1primemean_w = running_mean_turbulence(w1_w, np.zeros(len(w1bar_w)))
w2primemean_w = running_mean_turbulence(w2_w, np.zeros(len(w2bar_w)))
w3primemean_w = running_mean_turbulence(w3_w, np.zeros(len(w3bar_w)))

u1primemean_g = running_mean_turbulence(u1_g, u1bar_g)
u2primemean_g = running_mean_turbulence(u2_g, u2bar_g)
u3primemean_g = running_mean_turbulence(u3_g, u3bar_g)
w1primemean_g = running_mean_turbulence(w1_g, np.zeros(len(w1bar_g)))
w2primemean_g = running_mean_turbulence(w2_g, np.zeros(len(w2bar_g)))
w3primemean_g = running_mean_turbulence(w3_g, np.zeros(len(w3bar_g)))

ustar1_w = np.sqrt(u1primemean_w * w1primemean_w)
ustar2_w = np.sqrt(u2primemean_w * w2primemean_w)
ustar3_w = np.sqrt(u3primemean_w * w3primemean_w)

ustar1_g = np.sqrt(u1primemean_g * w1primemean_g)
ustar2_g = np.sqrt(u2primemean_g * w2primemean_g)
ustar3_g = np.sqrt(u3primemean_g * w3primemean_g)

ustartot_w = np.mean([ustar1_w, ustar2_w, ustar3_w], axis = 0)
ustartoterr_w = np.std([ustar1_w, ustar2_w, ustar3_w], axis = 0)

ustartot_g = np.mean([ustar1_g, ustar2_g, ustar3_g], axis = 0)
ustartoterr_g = np.std([ustar1_g, ustar2_g, ustar3_g], axis = 0)

def log_fit(z, grad, z0):
    return grad * np.log(z / z0)

z = np.array([0.75, 1.75, 2.75])

k_w = np.zeros(len(ustartot_w))
kerr_w = np.zeros(len(ustartot_w))
z0_w = np.zeros(len(ustartot_w))
z0err_w = np.zeros(len(ustartot_w))
p_w = np.zeros(len(ustartot_w))

k_g = np.zeros(len(ustartot_g))
kerr_g = np.zeros(len(ustartot_g))
z0_g = np.zeros(len(ustartot_g))
z0err_g = np.zeros(len(ustartot_g))
p_g = np.zeros(len(ustartot_g))

for i in range(0, len(ustartot_w)):
    uz = np.array([u1bar_w[i], u2bar_w[i], u3bar_w[i]])
    uzerr = np.array([u1barerr_w[i], u2barerr_w[i], u3barerr_w[i]])
    obj = curve_fit(log_fit, z, uz, sigma = uzerr, absolute_sigma = True, p0 = [1, 0.01])
    k_w[i] = ustartot_w[i] / obj[0][0]
    kerr_w[i] = np.sqrt((ustartot_w[i] / obj[0][0] ** 2) ** 2 * obj[1][0,0] + (ustartoterr_w[i] / obj[0][0]) ** 2)
    z0_w[i] = obj[0][1]
    z0err_w[i] = np.sqrt(obj[1][1,1])
    chisq = np.sum(((uz - log_fit(z, ustartot_w[i] / k_w[i], z0_w[i])) / uzerr) ** 2)
    p_w[i] = 1 - chi2.cdf(chisq, 1)

for i in range(0, len(ustartot_g)):
    uz = np.array([u1bar_g[i], u2bar_g[i], u3bar_g[i]])
    uzerr = np.array([u1barerr_g[i], u2barerr_g[i], u3barerr_g[i]])
    obj = curve_fit(log_fit, z, uz, sigma = uzerr, absolute_sigma = True, p0 = [1, 0.01])
    k_g[i] = ustartot_g[i] / obj[0][0]
    kerr_g[i] = np.sqrt((ustartot_g[i] / obj[0][0] ** 2) ** 2 * obj[1][0,0] + (ustartoterr_g[i] / obj[0][0]) ** 2)
    z0_g[i] = obj[0][1]
    z0err_g[i] = np.sqrt(obj[1][1,1])
    chisq = np.sum(((uz - log_fit(z, ustartot_g[i] / k_g[i], z0_g[i])) / uzerr) ** 2)
    p_g[i] = 1 - chi2.cdf(chisq, 1)

def weighted_mean(x, sigma):
    weights = 1 / sigma ** 2
    xw = x * weights
    mean = np.sum(xw) / np.sum(weights)
    return mean

#%%
n = 500
z = np.array([0.75, 1.75, 2.75])
z2 = np.linspace(0.75, 2.75, 51)

uz_w = np.array([u1bar_w[n], u2bar_w[n], u3bar_w[n]])
uzerr_w = np.array([u1barerr_w[n], u2barerr_w[n], u3barerr_w[n]])

uz_g = np.array([u1bar_g[n], u2bar_g[n], u3bar_g[n]])
uzerr_g = np.array([u1barerr_g[n], u2barerr_g[n], u3barerr_g[n]])

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.errorbar(z, uz_w, yerr = uzerr_w, fmt = 'ko', label = 'Data')
ax1.plot(z2, log_fit(z2, ustartot_w[n] / k_w[n], z0_w[n]), color='crimson', ls='-', lw=2.5, label = 'Log Profile Fit')
ax1.grid()
ax1.legend(fontsize=15)
ax1.set_xlabel(r'$z$ (m)', fontsize=15)
ax1.set_ylabel(r'$\overline{u}(z)$ (m/s)', fontsize=15)
ax1.tick_params(labelsize=13)

ax2.errorbar(z, uz_g, yerr = uzerr_g, fmt = 'ko', label = 'Data')
ax2.plot(z2, log_fit(z2, ustartot_g[n] / k_g[n], z0_g[n]), color='crimson', ls='-', lw=2.5, label = 'Log Profile Fit')
ax2.grid()
ax2.legend(fontsize=15)
ax2.set_xlabel(r'$z$ (m)', fontsize=15)
ax2.set_ylabel(r'$\overline{u}(z)$ (m/s)', fontsize=15)
ax2.tick_params(labelsize=13)

fig.show()

#%%

filename = 'ResponseTime.csv'

data2 = pd.read_csv(filename)

wind = data2['348 [m/s]'].to_numpy()
winderr = wind * grads[5]
t = np.arange(0, (len(wind) - 1) * 2 + 0.01, 2)

expected = np.zeros(len(wind))
expected[10:38] = 7.207 * np.ones(28)
expected[38:56] = 8.22 * np.ones(18)
expected[56:76] = 9.486 * np.ones(20)
plt.plot(times, spl(times),color='cornflowerblue', lw=2, label='Actual Response')
#plt.errorbar(t, wind, yerr = winderr, color, label = 'Actual Response')
plt.plot(t, expected, color='k', ls = '--', label = 'Idealised Response')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Wind Speed (m/s)', fontsize =15)
plt.tick_params(labelsize=13)
plt.grid()
plt.legend(fontsize=15)
plt.show()
