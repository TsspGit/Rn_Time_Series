__author__ = '@Tssp'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.titlesize']=16

# Read Data:
EAMRn =  [6.0388045019033, 5.8796358303147915, 5.4355607324717, 6.291724925138513, 6.2576039372658245, 5.552841303299885, 5.72919935109664, 6.5026631647226765, 5.323442264479034, 5.87598890187789, 6.054384465120276, 4.824274530216139, 4.76371617219886, 6.610461449136539, 5.602090134912608, 5.986272578336755, 5.743023930763711, 5.509295911205058, 6.933863386815908, 5.857911907896703, 5.18718299087213, 5.3284633402921715, 6.374315145064373, 5.721179767530792, 5.731950759887695]
EAM_BCN =  [5.841507619741011, 6.494560475252112, 5.991169442935866, 5.3099448534907125, 5.5067049532520524, 5.24415584486358, 6.098319890547772, 6.074115091440629, 5.6936385096335895, 6.592449849965621, 6.658405498582489, 6.1026371157899195, 6.05432467557946, 5.744691887680365, 6.2773624731569875, 5.721792337845783, 6.95516418924137, 5.6521395079943595, 5.789252495279118, 5.490125500426, 5.645911002645687, 5.682678144805285, 5.553609731246014, 7.142091906800562, 5.705052784511021]
EAM_PMP =  [6.222283849910814, 6.088368396369779, 5.989478909239477, 6.227297140627491, 6.021387372698102, 5.906428551187321, 6.043572211752132, 5.833752184498067, 6.401992175043846, 5.773003403021365, 6.477028048768336, 5.340389407410914, 5.847680305948063, 6.187764459726762, 5.185361356151347, 5.869581183608697, 5.954702571946747, 6.5108359979123485, 6.475868770054409, 6.12918826511928, 5.2642073728600325, 5.74742484579281, 5.425031817689234, 5.628776199963628, 5.496837538115832]
EAM_ZGZ =  [6.597640368403221, 5.690122292966259, 5.685162213383888, 5.854494289476044, 5.97024392108528, 5.298223729036292, 5.6153261612872685, 5.7388374951421, 5.708856504790637, 6.186367307390485, 5.657204881006358, 6.056342494731047, 5.48471890663614, 5.840221327178332, 5.62105147692622, 5.779281032328703, 5.472495993789361, 6.129297840351961, 5.549182074410575, 5.972467500336316, 5.960614963453644, 5.451028123193858, 6.05630247933524, 6.012967401621293, 6.525303470845125]
EAM_HSC =  [7.328170698516223, 5.7496132364078445, 6.495539100802675, 5.129196478396046, 5.801801642593072, 5.6420243321632855, 6.781983356086576, 5.695901092217893, 5.734931945800781, 5.8178236825125555, 5.909621452798649, 5.937380070589026, 6.666411964260802, 5.7827730373460415, 6.5266770343391265, 5.419647995306521, 5.877196175711496, 5.067497603747309, 6.448573871534698, 6.93081030553701, 6.249518452858438, 5.697031682851363, 5.525543757847378, 5.472242277495715, 5.596459252493722]
# Boxplot:
X = [EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC]
lbl = ['Rn', 'Rn+T BCN', 'Rn+T PMP', 'Rn+T ZGZ', 'Rn+T HSC']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True, patch_artist=True,
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.grid()
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
colors = ['dimgray']*5
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
fig.savefig('../../Figures/Boxplots/MAE_Boxplot_wcolor_5Fw_v3.eps', dpi=300)

# Couples:
EAM_BCN_PMP =  [5.565392046558614, 6.360884997309471, 5.388128163863201, 5.678791552173848, 6.011999402727399, 5.780287178195253, 7.588729118814274, 5.684051202268017, 6.098654766472018, 5.904849266519352, 5.958316258021763, 5.751921128253548, 5.869905316099828, 6.1128697298011, 5.813538531867826, 6.187644102135483, 6.274161319343412, 5.434064164453623, 5.998474510348573, 5.61422433658522, 6.1375822339739114, 6.288802361001774, 5.799859572430046, 7.500206733236507, 5.753884373878946]
EAM_BCN_HSC =  [6.178411522690131, 6.207960089858697, 5.940298897879464, 5.666716205830476, 6.695698485082509, 5.694830485752651, 6.543263065571687, 5.6315736576002475, 5.226264642209423, 6.0280057556775155, 5.94934268873565, 6.155915163001236, 6.209846263029138, 6.004123220638353, 9.160981548075773, 5.3852604068055445, 5.711089620784837, 5.767856597900391, 5.70049608970175, 5.805385394972198, 5.9087278405014345, 6.114483969552176, 5.886259312532386, 6.0753642101677094, 5.930446040873625]
EAM_BCN_ZGZ =  [6.013795619108239, 5.628193719046457, 6.189528173329879, 6.431885816613022, 5.502926378834005, 5.614769410113899, 5.089723742738062, 6.240708876629265, 5.320453760575275, 5.557994336498027, 5.662222258898677, 5.4834045488007215, 7.047238836483079, 6.541026640911491, 6.011062777772242, 5.127812482872788, 5.721613436329122, 5.508403583448761, 6.487839562552316, 5.727931081032266, 5.62757869642608, 6.196613856724331, 5.768891120443539, 5.671485550549566, 5.881165407141861]
EAM_PMP_HSC =  [6.513932675731425, 6.457771495896942, 6.4499825068882535, 5.77047246816207, 6.875553286805445, 6.279323500029895, 6.205875902759786, 6.134435731537488, 5.225199796715561, 5.622586036215023, 6.651118181189712, 6.094638590909997, 7.415758677891323, 5.417508105842435, 5.483417744539222, 5.739740955586336, 6.969659688521404, 6.246341938875159, 6.303894587925503, 6.887409210205078, 6.4738024886773555, 5.6890466651138, 5.73628230970733, 6.524096430564414, 5.698808008310746]
EAM_PMP_ZGZ =  [6.00289874174157, 6.031973546865035, 5.996296123582489, 5.7688509493458024, 5.903805907891721, 7.160029781108, 5.836118231014329, 7.470587905572385, 6.393890380859375, 5.646960900754345, 5.872741348889409, 5.899850845336914, 6.263890752986986, 5.68642705800582, 5.856858234016263, 6.200988068872569, 6.205248229357661, 6.824626572278081, 5.692284214253328, 5.4826650035624604, 6.122798024391641, 6.694975755652603, 6.006079265049526, 6.540798810063576, 6.138389976657167]
EAM_HSC_ZGZ =  [5.351387218553192, 5.569351390916474, 6.468320340526347, 5.4908978209203605, 5.322523428469288, 5.900434143689214, 5.676687668780891, 6.547797689632493, 6.165187446438536, 6.069760886990294, 5.362084330344687, 5.631854232476682, 5.944553998051857, 5.512956774964625, 6.01994024004255, 5.326624033402424, 5.9880202157156805, 5.508445545118683, 5.721133524057817, 5.721196700115593, 6.750664847237723, 6.22592019061653, 5.814094971637337, 5.896604226560009, 5.467990174585459]
X = [EAMRn, EAM_BCN_PMP, EAM_BCN_HSC, EAM_BCN_ZGZ, EAM_PMP_HSC, EAM_PMP_ZGZ, EAM_HSC_ZGZ]
lbl = ['Rn', 'Rn+T \nBCN / PMP', 'Rn+T \nBCN / HSC', 'Rn+T \nBCN / ZGZ', 'Rn+T \nPMP / HSC', 'Rn+T \nPMP / ZGZ',
 'Rn+T \nHSC / ZGZ']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True, patch_artist=True,
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
plt.grid()
colors = ['dimgray']*7
for patch, color in zip(bplot['boxes'], colors):
	patch.set_facecolor(color)
fig.savefig('../../Figures/Boxplots/MAE_Boxplot_Couples_wcolor_5Fw_v3.eps', dpi=300)
#fig.savefig('../../Figures/Boxplots/MAE_Boxplot_Couples_5Fw.eps', dpi=300)
#fig.savefig('../../Figures/Boxplots/../../Figures/BoxplotsMAE_Boxplot_Couples_5Fw.png', dpi=300)

# Trios:
EAM_BCN_PMP_HSC =  [5.922449150863959, 5.958976395276128, 5.537253983166753, 7.3347556445063375, 6.115288364643953, 6.136834125129544, 6.416445050920759, 5.890540765256298, 6.923582855536013, 6.3110414621781326, 6.222946478396046, 5.293326903362663, 5.460452722043407, 6.079502884222537, 6.069324688035614, 6.160557766349948, 5.330670259436783, 5.747638507765167, 5.526802491168587, 5.753680015096859, 5.6993811081866825, 5.5810986343695195, 5.66301120057398, 5.874046987416793, 6.814912951722437]
EAM_BCN_PMP_ZGZ =  [5.800402115802376, 6.356695175170898, 5.460846725775271, 5.757665361676898, 6.636060286541374, 6.077818189348493, 6.237978643300582, 5.884609767368862, 6.918997784050143, 6.170729890161631, 6.51529413340043, 6.314283098493304, 5.282464241494938, 5.672503685464664, 5.602321118724589, 6.181574490605568, 6.524689187808913, 5.49418492219886, 5.403578349522182, 5.670997736405353, 6.099522687950913, 6.201309476579938, 5.263670162278778, 6.271881843099789, 6.351962965361926]
EAM_PMP_HSC_ZGZ =  [6.0057031670395205, 6.200787485862265, 5.593269425995496, 5.968673433576312, 6.404735565185547, 6.217424587327606, 5.937175828583387, 6.873704092843192, 5.895352071645308, 5.82849136663943, 5.3042808065609055, 6.613410638303173, 7.591408398686623, 6.239719468720105, 5.755355290004185, 6.566963429353675, 5.694697360603177, 5.964676798606406, 5.541631971086774, 5.669057106485172, 5.704337061667929, 5.6799322634327165, 5.411590692948322, 5.668584667906469, 6.130688842462034]
X = [EAMRn, EAM_BCN_PMP_HSC, EAM_BCN_PMP_ZGZ, EAM_PMP_HSC_ZGZ]
lbl = ['Rn', 'Rn+T \nBCN / PMP / HSC', 'Rn+T \nBCN / PMP / ZGZ', 'Rn+T \nPMP / HSC / ZGZ']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True, patch_artist=True,
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
plt.grid()
#plt.ylim([7.8, 9.4])
colors = ['dimgray']*4
for patch, color in zip(bplot['boxes'], colors):
	patch.set_facecolor(color)
fig.savefig('../../Figures/Boxplots/MAE_Boxplot_Trios_wcolor_5Fw_v3.eps', dpi=300)
#fig.savefig('../../Figures/Boxplots/MAE_Boxplot_Trios_5Fw.eps', dpi=300)
#fig.savefig('../../Figures/Boxplots/MAE_Boxplot_Trios_5Fw.png', dpi=300)