import matplotlib.pyplot as plt
import numpy as np
import os, sys
from scipy.special import erf


#########
# funcsions for spatial derivatives
def drr1(qq,drr,dir):
   '''
   To calculate 1st order derivative in r direction
   '''
   
   if dir == 'up':
      i0, i1 = 1, qq.shape[0]
   elif dir == 'dw':
      i0, i1 = 0, qq.shape[0]-1
   else:
      print('Error: dir must be up or dw')
      sys.exit()   
   
   dqq = np.zeros_like(qq)
   dqq[i0:i1,:] = (qq[1:,:] - qq[:-1,:])/drr
   
   return dqq

def drr2(qq,drr):
   '''
   To calculate 2nd order derivative in r direction
   '''
   
   dqq = np.zeros_like(qq)
   dqq[1:-1,:] = (qq[2:qq.shape[0],:] - qq[0:-2,:])/drr*0.5
   
   return dqq
   
def dth1(qq,dth,dir):
   '''
   To calculate 1st order derivative in theta direction
   '''
   
   if dir == 'up':
      j0, j1= 1,qq.shape[1]
   elif dir == 'dw':
      j0, j1 = 0, qq.shape[1]-1
   else:
      print('Error: dir must be up or dw')
      sys.exit()   
   
   dqq = np.zeros_like(qq)
   dqq[:,j0:j1] = (qq[:,1:] - qq[:,:-1])/dth
   
   return dqq

def dth2(qq,dth):
   '''
   To calculate 2nd order derivative in r direction
   '''
   
   dqq = np.zeros_like(qq)
   dqq[:,1:-1] = (qq[:,2:qq.shape[1]] - qq[:,0:-2])/dth*0.5
   
   return dqq

#########
def time_marching(Bph, Aph, dt,urr, uth,RR,RRm,sinTH,sinTHm,drr,dth,et,so,omrr,omth,ibase):

   # poloidal magnetic field
   Brr = + dth2(sinTH*Aph,dth)/RR/sinTH
   Bth = - drr2(   RR*Aph,drr)/RR
      
   Bph_adrr = - drr2(RR*urr*Bph,drr)/RR #  動径方向の移流(Bph)
   Bph_adth = - dth2(   uth*Bph,dth)/RR #  緯度方向の移流(Bph)
   
   Aph_adrr = - drr2(   RR*Aph,drr)*urr/RR       # 動径方向の移流(Aph)
   Aph_adth = - dth2(sinTH*Aph,dth)*uth/RR/sinTH # 緯度方向の移流(Aph)
   
   # 磁場の微分(拡散量)
   Bphrr = drr1(Bph,drr,'up')
   Bphth = dth1(Bph,dth,'up')
   Aphrr = drr1(Aph,drr,'up')
   Aphth = dth1(Aph,dth,'up')
   
   Bph_dfrr = et*drr1(RRm**2*Bphrr,drr,'dw')/RR**2
   Bph_dfth = et*dth1(sinTHm*Bphth,dth,'dw')/RR**2/sinTH
   Aph_dfrr = et*drr1(RRm**2*Aphrr,drr,'dw')/RR**2
   Aph_dfth = et*dth1(sinTHm*Aphth,dth,'dw')/RR**2/sinTH
   
   # Omega effect
   Bph_omrr = Brr*omrr*RR*sinTH
   Bph_omth = Bth*omth*RR*sinTH
   
   # source term
   tmp, Bphso = np.meshgrid(rr,Bph[ibase,:], indexing='ij')
   Aph_sour = so*Bphso/(1 + (Bphso)**2)

   dBph = + (Bph_adrr + Bph_adth) \
          + (Bph_dfrr + Bph_dfth - et*Bph/RR**2/sinTH**2) \
          + (Bph_omrr + Bph_omth) 
          
   dAph = + (Aph_adrr + Aph_adth) \
          + (Aph_dfrr + Aph_dfth - et*Aph/RR**2/sinTH**2) \
          + Aph_sour
      
   Bphm = Bph + dt*dBph
   Aphm = Aph + dt*dAph
   
   return Bphm, Aphm, Bphso, Aph_sour

#
def boundary_condition(Aph, Bph,margin,rr,th,ixg,jxg):
   # 動径方向境界条件
   for i in range(0, margin):
      #下部境界条件(完全導体)A=B=0
      Bph[i,margin:jxg-margin] = + Bph[2*margin-i-1,margin:jxg-margin]/rr[i]*rr[2*margin-i-1]
      Aph[i,margin:jxg-margin] = - Aph[2*margin-i-1,margin:jxg-margin]

      #上部境界条件B=0,dA/dr=0
      Bph[ixg-i-1,margin:jxg-margin] = -Bph[ixg-2*margin+i,margin:jxg-margin]
      Aph[ixg-i-1,margin:jxg-margin] = +Aph[ixg-2*margin+i,margin:jxg-margin]/rr[ixg-i-1]*rr[ixg-2*margin+i]

   # 緯度方向境界条件      
   for j in range(0, margin):
      #極の境界条件A=B=0
      Bph[margin:ixg-margin,j]  = -Bph[margin:ixg-margin,2*margin-j-1]
      Aph[margin:ixg-margin,j]  = -Aph[margin:ixg-margin,2*margin-j-1]
      #赤道境界条件B=0(反対称),dA/dθ=0
      Bph[margin:ixg-margin,jxg-j-1] = -Bph[margin:ixg-margin,jxg-2*margin+j]
      Aph[margin:ixg-margin,jxg-j-1] = -Aph[margin:ixg-margin,jxg-2*margin+j]
   return Aph, Bph
# Prepare data directory
os.makedirs('data',exist_ok=True)

# 計算継続のフラグ
cont_flag = True

########################
########################
## setup prameters
#年数
year = 100

#格子点数
ix=64
jx=128


#マージンの数
margin = 1

# マージンを含む格子点数
ixg = ix + 2*margin
jxg = jx + 2*margin


# 太陽定数
rsun = 6.96e10

########################
########################
# 計算領域の設定
# 領域の定義
rrmin, rrmax = 0.65*rsun, rsun
thmin, thmax = 0, np.pi

# dr,dθの設定
drr = (rrmax - rrmin)/ix
dth = (thmax - thmin)/jx

#座標rrの設定
rr    = np.zeros(ixg)
rr[0] = rrmin + drr*(0.5 - margin)

for i in range(1, ixg):
   rr[i] = rr[i - 1] + drr
   
#座標thの設定    
th    = np.zeros(jxg)
th[0] = thmin + dth*(0.5 - margin)

for j in range(1,jxg):
   th[j] = th[j - 1] + dth

np.savez('data/geometry.npz',rr=rr,th=th)

#メッシュの作成
RR ,TH  = np.meshgrid(rr, th,indexing='ij')
RRm = np.zeros_like(RR)
THm = np.zeros_like(TH)

RRm[1:RR.shape[0],:] = 0.5*(RR[1:RR.shape[0],:] + RR[0:RR.shape[0]-1,:])
THm[:,1:RR.shape[1]] = 0.5*(TH[:,1:RR.shape[1]] + TH[:,0:RR.shape[1]-1])

sinTH = np.sin(TH)
cosTH = np.cos(TH)
sinTHm = np.sin(THm)
X, Y = RR * np.cos(TH), RR * np.sin(TH)

##########################
##########################
# differential rotation
ome = 456.e-9*2*np.pi # rotation rate at equator
omc = 0.92*ome         # rotation rate at radiative zone
rc = 0.7*rsun          # radiative zone boundary
d  = 0.02*rsun         # width of tachocline
c2 = 0.2*ome

om = omc + 0.5*(1 + erf((RR-rc)/d))*(ome - omc - c2*cosTH**2)

omrr = drr2(om, drr)
omth = dth2(om, dth)/RR


# diffusivity
etc = 1.e9
ett = 1.e11

et = etc + 0.5*(ett - etc)*(1 + erf((RR-rc)/d))

#ソース関数の定数・パラメータ
# B0  = 1.0 * 10 ** 5
# s0  = 20
# r2  = 0.95*rsun
# r3  =  1.0*rsun
# d2  = 0.01*rsun
# d3  = 0.01*rsun
# r_c =  0.7*rsun

#タコクラインの要素番号
ibase = np.argmin(abs(rr - rc))

cso = 35
so0 = cso*ett/rsun
r1 = 0.95*rsun
d1 = 0.01*rsun

so = so0*0.5*(1 + erf((RR-r1)/d1))*(1 - erf(RR-rsun)/d1)*cosTH*sinTH

# source time time-independent part
#s  = s0 / 2 * (1 + erf_2) * (1 - erf_3) * np.sin(TH) * np.cos(TH)

# 太陽表面の要素番号
isurf = np.argmin(abs(rr - rsun)) - 1


# Omg_Eq =  460.7 * 2 * np.pi * 1.e-9
# a2     = -62.69 * 2 * np.pi * 1.e-9
# a4     = -67.13 * 2 * np.pi * 1.e-9
# Omg_c  =  432.8 * 2 * np.pi * 1.e-9
# Omg_s  =  Omg_Eq + a2 * np.cos(TH) ** 2 + a4 * np.cos(TH) ** 4

# differential rotation
#Omg = Omg_c + 0.5 * (1 + erf_1) * (Omg_s - Omg_c)


# 子午面循環流の定数・パラメータ
# u0  = 2000
# m   = 0.5
# p   = 0.25
# q   = 0
# xi  = rsun/RR  - 1
# rr0 = 0.71*rsun # base of the meridional flow
# xi0 = rsun/rr0 - 1
# xi[RR > rsun] = 0
# c1  = (2*m+1)*(m+p)/(m+1)/p * (xi0**(-m))
# c2  = (2*m+p+1)*m/(m+1)/p * (xi0**(-(m+p)))

# # 誤差関数
# erf_1 = erf(2 * (RR - r_c)/d1)
# erf_2 = erf(    (RR - r2 )/d2)
# erf_3 = erf(    (RR - r3 )/d3)


# # Meridional flow
# # radial velocity
# urr = u0*(rsun/RR) \
#    *(-1/(m+1) + c1/(2*m + 1)*xi**m - c2/(2*m+p+1)*xi**(m+p)) \
#    *xi*sinTH**q*( (q+2)*cosTH**2 - sinTH**2)
# # colatitudinal velocity
# uth = u0*((rsun/RR)**3) \
#    *(-1+c1*xi**m-c2*xi**(m+p)) \
#    *np.sin(TH)**(q+1)*np.cos(TH)

u0 = 1000
rb = 0.65*rsun

urr = -u0*2*(rsun - rb)/np.pi/RR \
     *(RR-rb)**2/(rsun - rb)**2 \
     *np.sin(np.pi*(RR-rb)/(rsun-rb))*(3*cosTH**2 - 1)
     
uth = u0*((3*RR-rb)/(rsun-rb)*np.sin(np.pi*(RR-rb)/(rsun-rb)) \
      + RR*np.pi/(rsun-rb)*(RR-rb)/(rsun-rb)*np.cos(np.pi*(RR-rb)/(rsun-rb))) \
      *2*(rsun-rb)/np.pi/RR*(RR-rb)/(rsun-rb)*cosTH*sinTH


urr[RR < rb] = 0
uth[RR < rb] = 0
#θ＝０(回転軸)(対称性)
# 境界の外で子午面流の設定
for i in range(0,margin):
   # upper
   urr[i,:] = - urr[2*margin-i-1,:]
   uth[i,:] = + uth[2*margin-i-1,:]
   
   # lower
   urr[ixg-i-1,:] = - urr[ixg-2*margin+i,:]
   uth[ixg-i-1,:] = + uth[ixg-2*margin+i,:]


# latitudinal boundary
for j in range(0,margin):
   # pole
   urr[:,j] = + urr[:,2*margin - j - 1] # symmetric
   uth[:,j] = - uth[:,2*margin - j - 1] # antisymetric

   # equator
   urr[:,jxg-j-1] = + urr[:,jxg-2*margin + j] # symmetric
   uth[:,jxg-j-1] = - uth[:,jxg-2*margin + j] # antisymmetric

#CFL condition
c_cfl=0.1
dtmin = 1.e10
for i in range(margin,ixg-margin):
   for j in range(margin,jxg-margin):
      dt_adv = c_cfl * np.min([drr, rr[i]*dth])/ np.sqrt(urr[i,j]**2 + uth[i,j]**2)
      dt_dif = c_cfl * np.min([drr, rr[i]*dth])**2 /(2 * et[i,j])
      dtmin = np.min([dtmin,dt_adv,dt_dif])
      
dt  = dtmin
print(dtmin)

# 初期条件
Aph = np.zeros((ixg, jxg))
Bph = np.zeros((ixg, jxg))
#Aph = np.zeros((ixg, jxg))
#Aph = sinTH/(RR/rsun)**2
Bph = np.sin(2*TH)*0.1
Bph[0:ibase,:] = 0

tend = 30000*86400 # total calculation duration
dtout = 100*86400 # data output cadence
time = 0
n = 0
nd = 0

plt.close('all')
plt.clf()
fig = plt.figure('dynamo',figsize=(5,10))

while time < tend:
   time += dt
   n += 1
   if(time//dtout != (time-dt)//dtout):
      nd += 1
      ax = fig.add_subplot(111,aspect='equal')
      ax.pcolormesh(Y,X,Bph,vmax=1.e0,vmin=-1.e0,cmap='bwr')
      ax.contour(Y,X,RR/rsun*sinTH*Aph,colors='black',levels=np.linspace(-8.e12,8.e12,10))
      ax.set_xlim(0,rsun)
      ax.set_ylim(-rsun,rsun)
      plt.pause(0.1)
      print(time/86400,n,nd)
      Brr =  dth2(sinTH*Aph,dth)/RR/sinTH
      Bth = -drr2(   RR*Aph,drr)/RR
      np.savez(file='data/data.'+str(nd).zfill(6)+'.npz' \
            ,Aph=Aph,Bph=Bph,Brr=Brr,Bth=Bth,time=time)
                
   ####ダイナモ方程式                       
   Bphm, Aphm, Bphso, Aph_sour = time_marching(Bph , Aph ,dt, urr, uth, RR, RRm, sinTH, sinTHm, drr, dth, et, so, omrr, omth, ibase)
   
   Aphm, Bphm = boundary_condition(Aphm, Bphm,margin,rr,th,ixg,jxg)

   Bphn, Aphn, Bphso, Aph_sour = time_marching(Bphm, Aphm, dt, urr, uth, RR, RRm, sinTH, sinTHm, drr, dth, et, so, omrr, omth, ibase)
   Aphn, Bphn = boundary_condition(Aphn, Bphn,margin,rr,th,ixg,jxg)
    
   Bph = 0.5*Bph + 0.5*Bphn
   Aph = 0.5*Aph + 0.5*Aphn

   