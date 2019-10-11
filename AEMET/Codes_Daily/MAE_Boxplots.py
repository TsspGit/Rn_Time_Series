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
EAM_Rn = [8.211480729123378, 9.208639915953292, 8.514089584350586, 8.322871674882604, 8.150424510874647, 8.526771139591299, 8.307480791781812, 8.621930791976604, 8.220381229481799, 8.403754741587537, 8.480089796350358, 8.563497624498732, 8.175465117109583, 8.29117425959161, 8.963435152743726, 8.061902472313415, 8.142905296163356, 8.604449779429334, 8.200639602985788, 8.299986981331035, 8.248483820164457, 8.127263779335834, 8.297681727307909, 8.25263940527084, 8.225357623810464]
EAM_RnT_BCN = [8.059945654361806, 8.03970312564931, 8.12277168923236, 8.053294607933532, 8.048365126264857, 8.165090114512342, 8.02081221722542, 8.061697614953873, 8.185345507682637, 8.413906503230967, 8.132826094931744, 8.1881618093937, 8.251170665659803, 8.117733082872755, 8.350974224983378, 7.956330563159699, 8.038048358673745, 8.065417675261802, 8.0369804463488, 8.129734850944356, 8.102025661062687, 8.101720079462579, 8.135596863766933, 8.027808940156977, 8.28533558135337]
EAM_RnT_PMP = [8.048338423383997, 8.061835918020694, 8.010926185770238, 7.947391794082966, 8.436416666558449, 8.059653180710812, 8.212462526686648, 8.13142910409481, 8.318235965485268, 8.11553975369068, 8.016166605847948, 8.170698084729784, 8.160849753846513, 8.226865322031873, 8.555549986819004, 8.194006899569898, 8.282557548360622, 8.095395757796917, 8.102666895440285, 8.380331404665684, 8.062647920973758, 8.130499251345372, 8.203159900421792, 8.082500985328187, 8.104652485948927]
EAM_RnT_ZGZ = [8.344254067603577, 8.07323772349256, 8.18323585834909, 8.246122157320062, 8.079329957353307, 8.133613464680124, 8.202599910979576, 8.189216451441988, 8.144079573610997, 8.356861033338182, 8.205509226372902, 8.233928761583693, 8.160559431035468, 8.25779160032881, 8.249153096625145, 8.072600141484687, 8.264818029200777, 8.10154943263277, 8.342795716955306, 8.31978436733814, 8.241368922781437, 8.261904817946414, 8.259412603175386, 8.172716952384786, 8.345493763051135]
EAM_RnT_HSC = [8.281530583158453, 8.28206115073346, 8.177005159093978, 8.052565513773168, 8.24305834668748, 8.182604972352372, 8.053872169332301, 8.083223383477394, 8.251345898242707, 8.172950541719477, 8.06568088937313, 8.290624862021588, 8.129040657205785, 8.125324898577752, 7.993029979949302, 8.35301245019791, 8.3312279315705, 8.177043387230407, 7.969878988063082, 8.030435237478702, 8.054201572499377, 8.09184658781011, 8.273422038301508, 8.114550732551738, 8.039057264936732]
# Boxplot:
X = [EAM_Rn, EAM_RnT_BCN, EAM_RnT_PMP, EAM_RnT_ZGZ, EAM_RnT_HSC]
lbl = ['Rn', 'Rn+T BCN', 'Rn+T PMP', 'Rn+T ZGZ', 'Rn+T HSC']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True, 
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.grid()
plt.ylim([7.8, 9.4])
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
# colors = ['#1f77b4', 'yellow', '#2ca02c', 'red', 'darkviolet']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
fig.savefig('../Figures/CNN/MAE_Boxplot_wcolor.eps', dpi=300)

# Couples:
EAM_RnT_BCN_PMP = [8.249645354899954, 8.109843112052754, 7.9596848589308715, 8.281683455122279, 8.047973389321186, 8.097486049570936, 8.487436740956408, 8.141598884095536, 8.12750398351791, 8.184711009898084, 8.01150131225586, 7.986387820954018, 8.025565451764045, 8.05869569169714, 8.186629599713264, 8.263832660431557, 8.35420085014181, 8.160669326782227, 7.977884941912712, 8.166457277663211, 8.132487601422248, 8.139243470861556, 8.106495268801426, 7.99684078135389, 7.974890201649767]
EAM_RnT_BCN_HSC = [8.046628586789394, 8.140524154013775, 7.981474125638921, 8.063533011903154, 7.96350475067788, 8.090603524066033, 8.00985790820832, 8.0011503747169, 8.196996364187687, 7.999926749696123, 8.382777234341235, 8.002729537639212, 8.123667615525266, 8.200908417397358, 8.204587246509309, 8.01505729999948, 8.151526674311212, 8.036815278073574, 8.211881110008727, 8.117931893531312, 8.247377842030627, 8.046907708999958, 8.49198572686378, 8.075461773162193, 8.03215818202242]
EAM_RnT_BCN_ZGZ = [8.159620731434924, 8.06577723077003, 8.029058172347698, 8.204421266596368, 8.018829223957468, 7.972556864961665, 8.144437871080763, 8.130097084857049, 8.145475752810215, 8.053051765928878, 8.638812531816198, 8.00288151680155, 8.171749601972865, 8.300101828067861, 7.971094740198014, 8.146595001220703, 8.19834376396017, 8.168464457735102, 8.066883371231404, 8.072343704548288, 7.9627782537582075, 7.919400316603641, 8.09924239300667, 8.158352263430332, 8.089309611219042]
EAM_RnT_PMP_HSC = [8.16188990816157, 8.074098587036133, 8.075809559923536, 8.127061194561897, 8.041548627488156, 8.146307519141663, 8.084034858865941, 8.258401505490566, 8.079277850211934, 7.945721971227767, 8.034838453252265, 8.441567806487388, 8.04375774302381, 8.2489593992842, 8.108335251503803, 8.081670152380111, 8.316200783912171, 7.987559338833424, 8.126349712939973, 8.131910364678566, 8.323530887035613, 8.252454798272316, 8.199543973232837, 8.12511529313757, 8.163320947200694]
EAM_RnT_PMP_ZGZ = [8.18601011722646, 8.050267645653259, 8.089551114021464, 8.029951379654255, 8.104592343594165, 8.368053152206096, 8.001419026800926, 8.193700262840759, 8.08381267304116, 8.223879590947577, 8.075669187180539, 8.003088484419154, 8.026454519718252, 7.9263492340737205, 8.18858933956065, 8.017728318559362, 8.280054579389857, 8.014361117748503, 8.105738781868144, 8.209875390884724, 7.91840808949572, 8.021451422508727, 8.031930436479284, 7.998329284343313, 8.114975706059882]
EAM_RnT_HSC_ZGZ = [8.331128343622735, 8.071960083981777, 8.019035501683012, 8.332791835703748, 8.478301920789354, 8.171973370491191, 8.0700169015438, 8.410480458685692, 8.04945434407985, 8.122066944203478, 8.047828390243206, 7.9331585092747465, 8.024175156938268, 8.282581248181932, 8.30061575706969, 8.042139864982442, 8.125247549503408, 8.272313990491503, 8.356525502306349, 8.328973810723488, 8.039176697426655, 8.072528595620014, 8.43459932854835, 8.309487566034845, 7.970771748968896]
X = [EAM_Rn, EAM_RnT_BCN_PMP, EAM_RnT_BCN_HSC, EAM_RnT_BCN_ZGZ, EAM_RnT_PMP_HSC, EAM_RnT_PMP_ZGZ, EAM_RnT_HSC_ZGZ]
lbl = ['Rn', 'Rn+T \nBCN / PMP', 'Rn+T \nBCN / HSC', 'Rn+T \nBCN / ZGZ', 'Rn+T \nPMP / HSC', 'Rn+T \nPMP / ZGZ',
 'Rn+T \nHSC / ZGZ']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True,
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
plt.grid()
plt.ylim([7.8, 9.4])
# colors = ['#1f77b4', 'yellow', '#2ca02c', 'red', 'darkviolet', 'darkgreen', 'dimgray']
# for patch, color in zip(bplot['boxes'], colors):
# 	patch.set_facecolor(color)
fig.savefig('../Figures/CNN/MAE_Boxplot_Couples_wcolor.eps', dpi=300)
#fig.savefig('../Figures/CNN/MAE_Boxplot_Couples.eps', dpi=300)
#fig.savefig('../Figures/CNN/MAE_Boxplot_Couples.png', dpi=300)

# Trios:
EAM_RnT_BCN_PMP_HSC = [8.406214734341235, 8.380001392770321, 8.120754607180332, 8.39817952095194, 8.043899820206013, 8.148107325777094, 8.195880362328063, 8.248492342360477, 8.119088558440513, 8.565575863452668, 8.296524818907393, 8.110864679864113, 8.16932264287421, 8.187162358710106, 8.210852156294154, 8.240688851539124, 8.175578705807949, 8.100183689847906, 8.283883196242313, 8.375484628880278, 8.200105707696144, 8.19714801869494, 8.059559030735747, 8.214903324208361, 8.185132128127078]
EAM_RnT_BCN_PMP_ZGZ = [8.174906020468853, 8.767834196699427, 8.226545496189848, 8.198476060907891, 8.451418288210606, 8.028160622779358, 8.429145691242624, 8.351192920765978, 8.11584095244712, 8.53303763206969, 8.238795016674285, 8.39970357367333, 8.419038245018493, 8.146580188832385, 8.132718918171335, 8.476719957716922, 8.079648159919902, 8.535899385492852, 8.223145058814515, 8.108783356686855, 8.19886743261459, 8.244209654787754, 8.150498653980012, 8.176358730234998, 8.201180072540932]
EAM_RnT_PMP_HSC_ZGZ = [8.217519394894863, 8.210397801500685, 8.147792978489653, 8.318645680204352, 8.065411669142703, 8.163245789548183, 8.029282995995056, 8.04766797004862, 8.24703906444793, 7.969151679505694, 8.125941093931807, 8.098855363561752, 8.092623690341382, 9.242507893988426, 8.35495096571902, 8.506346966357942, 8.87733017130101, 8.139275489969457, 8.234561920166016, 8.199499657813538, 8.168495949278487, 8.103587454937873, 8.468756168446642, 8.06826197847407, 8.101316695517681]
X = [EAM_Rn, EAM_RnT_BCN_PMP_HSC, EAM_RnT_BCN_PMP_ZGZ, EAM_RnT_PMP_HSC_ZGZ]
lbl = ['Rn', 'Rn+T \nBCN / PMP / HSC', 'Rn+T \nBCN / PMP / ZGZ', 'Rn+T \nPMP / HSC / ZGZ']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True,
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
plt.grid()
plt.ylim([7.8, 9.4])
# colors = ['#1f77b4', 'yellow', '#2ca02c', 'red', 'darkviolet', 'darkgreen', 'dimgray']
# for patch, color in zip(bplot['boxes'], colors):
# 	patch.set_facecolor(color)
fig.savefig('../Figures/CNN/MAE_Boxplot_Trios_wcolor.eps', dpi=300)
#fig.savefig('../Figures/CNN/MAE_Boxplot_Couples.eps', dpi=300)
#fig.savefig('../Figures/CNN/MAE_Boxplot_Couples.png', dpi=300)


# All:
EAM_RnT_BCN_PMP_HSC_ZGZ = [8.812288365465529, 8.286808338571102, 8.146108546155565, 8.341915252360891, 8.261247594305809, 8.146348912665184, 8.14116242591371, 8.170524800077398, 8.211366085295982, 8.303294770261074, 8.311806861390458, 8.209686563370076, 8.451120944733315, 8.141129311094893, 8.169250812936337, 8.301805658543364, 8.118594068161984, 8.293023738455265, 8.434221470609625, 8.069732016705451, 8.251993138739403, 8.27000597690014, 8.149551432183449, 8.626447028302131, 8.05634279454008]
X = [EAM_Rn, EAM_RnT_BCN_PMP_HSC_ZGZ]
lbl = ['Rn', 'Rn+T \nBCN / PMP / HSC / ZGZ']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
bplot = ax.boxplot(X, sym='+', labels=lbl, notch=True,
                   medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
plt.grid()
plt.ylim([7.8, 9.4])
# colors = ['#1f77b4', 'yellow', '#2ca02c', 'red', 'darkviolet', 'darkgreen', 'dimgray']
# for patch, color in zip(bplot['boxes'], colors):
# 	patch.set_facecolor(color)
fig.savefig('../Figures/CNN/MAE_Boxplot_All_wcolor.eps', dpi=300)
#fig.savefig('../Figures/CNN/MAE_Boxplot_Couples.eps', dpi=300)
#fig.savefig('../Figures/CNN/MAE_Boxplot_Couples.png', dpi=300)





