__author__ = '@Tssp'
import numpy as np
import pandas as pd

def ErrorToCSV(Rn, BCN, PMP, ZGZ, HSC, nFw):
    DF = pd.DataFrame({'Rn': Rn,
                  'BCN': BCN,
                  'PMP': PMP,
                  'ZGZ': ZGZ,
                  'HSC': HSC})
    DF.to_csv(f'~/CIEMAT/Rn_Time_Series/AEMET/logs/ErrorsCNN{nFw}Fw.csv', index=False) #Change name whenever you want
## Input the lists here
EAMRn =  [10.448024993247174, 8.535750368808179, 9.06626348292574, 8.582859607453042, 8.794983478302651, 9.775118645201339, 9.685805056957488, 8.765169103094872, 8.726329559975483, 8.943374552625292, 8.631710133654005, 10.221656555825092, 8.933170480931059, 13.121957413693691, 11.278659008918925, 8.429257616083673, 9.378626397315491, 8.463517452808137, 8.784625925916307, 10.296041610393118, 10.64049838451629, 10.649339797648977, 9.294448203228889, 9.896800345562873, 9.205774023177776]
EAM_BCN =  [9.676332676664313, 8.026367715064515, 11.628765593183802, 9.805619747080701, 8.33443686302672, 8.20570966030689, 7.747238362089116, 8.261518762466755, 8.160165218596763, 8.875131485310007, 8.124538462212746, 8.106853809762509, 9.359839743756233, 8.444055192014003, 8.259071674752743, 8.731540923422955, 9.024265086397211, 8.420445218999335, 8.223118721170628, 8.33499023762155, 8.07998304164156, 8.290973744493849, 8.152487491039519, 8.315017740777199, 8.519863088080223]
EAM_PMP =  [8.185900302643471, 7.7895577613343585, 8.06041291419496, 8.017056891258727, 8.151025325693983, 8.283938184697577, 9.409763376763527, 8.372474589246385, 8.58283777439848, 7.989712613694211, 7.768564914135223, 8.310553814502473, 8.204557986969643, 8.42631895998691, 8.005968175035841, 8.373147274585481, 8.476829447644823, 9.6221217703312, 8.293285369873047, 8.503455020011739, 8.153312804851126, 8.266073470420025, 8.17245536154889, 8.46104216068349, 8.368328622046937]
EAM_ZGZ =  [8.109713980492126, 8.834156401613926, 8.031353849045773, 8.429605808663876, 8.343758684523563, 9.398339048345038, 8.387619749028632, 8.78066683830099, 9.738214208724651, 8.990217249444191, 7.832199583662317, 8.813559998857214, 8.989021625924618, 9.480084601868974, 7.969169210880361, 9.048326451727684, 8.128134423113885, 8.459377207654589, 8.34021905127992, 8.065221867662794, 8.039453506469727, 8.324669249514317, 8.503616130098383, 8.008206712438705, 7.977022658003137]
EAM_HSC =  [9.00646672350295, 8.192386789524809, 8.187967340996925, 8.55821593264316, 8.279952475365173, 8.296975318421708, 8.146078718469498, 8.325244903564453, 8.110590265152302, 7.897552977217004, 8.473966192691885, 8.05657707376683, 9.298356644650722, 8.922736594017517, 10.062417253534845, 8.256066910764003, 8.287230836584213, 8.723089664540392, 7.7815681619847075, 8.168973679238178, 8.750206724126288, 8.685769628971181, 7.973845380417844, 9.178157806396484, 8.2551784109562]

# Apply the function:
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=1)

EAMRn =  [8.920218056126645, 9.383422048468338, 8.931918174342105, 7.669485634251645, 7.581411261307566, 8.373459263851768, 9.348885706851357, 8.588566027189556, 9.655668881064967, 9.727798381604646, 8.720153487356086, 9.970166979337993, 9.133476337633635, 9.770184567100124, 10.342127428556743, 8.936225971422697, 8.948661723889803, 9.909335367303145, 9.932627868652343, 9.185524067125822, 10.448065747712787, 9.475664239180716, 10.46269386693051, 8.327557935212788, 9.19944201017681]
EAM_BCN =  [7.027676311292146, 6.484993302194696, 6.578755910773026, 6.61791185077868, 6.1692768699244445, 6.138560525994552, 7.982969785991468, 6.404286153692948, 6.319651714124178, 7.00300272891396, 7.170600409256784, 6.053203181216591, 6.263610518606085, 5.9287983542994445, 6.09340984946803, 6.483144619590358, 6.962250799881785, 6.173590208354749, 6.137488274825246, 6.525316178171258, 6.511476255718031, 6.09895123933491, 6.908246130692332, 6.149425105044716, 6.6071423179224915]
EAM_PMP =  [6.2759449607447575, 6.022487841154399, 6.614898199784128, 7.182573017321135, 7.098061370849609, 6.442191154078434, 6.681503697445518, 6.579481305574116, 5.976498613859478, 6.2072304575066815, 6.577594235068873, 6.226707659269634, 6.771979843942742, 6.60789690519634, 5.931071873715049, 6.159496909693668, 6.1095122487921465, 6.791392517089844, 6.678152024118524, 6.561065713982833, 6.531351109554893, 5.8847099705746295, 5.938958499306127, 6.334074321546052, 5.9939524600380345]
EAM_ZGZ =  [6.052183171322471, 6.145253552888569, 5.721018941778886, 6.3296766180741155, 6.909688527960526, 6.805426467092413, 5.658365229556435, 5.817729628713507, 7.160287676359478, 6.439720193963302, 6.613845343338816, 6.565053156802529, 7.0727795651084495, 6.096656357614617, 6.240634235582854, 6.463244227359169, 6.767367031699733, 6.781108856201172, 5.985144083123458, 5.930568092747738, 6.987106724789268, 6.507741908023232, 6.2357423481188325, 7.726269210012336, 6.23885931717722]
EAM_HSC =  [6.2190399169921875, 6.612041111996299, 6.041874855443051, 6.001055265727796, 6.161822710539165, 6.907572174072266, 5.91261689035516, 7.169971425909745, 6.305123941521895, 6.285249930933902, 5.8476920278448805, 5.604023501747533, 6.362445068359375, 6.134074442010177, 7.240007862291838, 6.82279197291324, 7.206117328844573, 6.129225520083779, 5.949077606201172, 6.9294273376464846, 6.465765983179996, 6.450022848028886, 6.658551386782998, 7.00233567890368, 7.291490936279297]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=2)

EAMRn =  [9.100836197535196, 8.877384901046753, 8.234065612157186, 10.165156841278076, 8.82512633005778, 9.892727931340536, 9.074461698532104, 6.690600951512654, 10.348102728525797, 9.917052586873373, 8.51591444015503, 10.234328269958496, 8.711634000142416, 7.850077191988627, 6.449809551239014, 8.005836208661398, 9.812852303187052, 8.008658051490784, 8.39241365591685, 9.758728504180908, 9.591944456100464, 9.83163054784139, 8.551848649978638, 7.277944803237915, 8.578910946846008]
EAM_BCN =  [5.268784244855245, 7.587532917658488, 6.305444002151489, 5.719224294026692, 6.462933540344238, 5.563564737637837, 5.045621315638225, 6.1824411153793335, 6.328147649765015, 5.692033330599467, 5.547954281171163, 5.40404490629832, 4.865819851557414, 5.954749743143718, 5.267717917760213, 5.534078399340312, 5.281425913174947, 5.28737739721934, 6.152094046274821, 4.815177798271179, 5.929673631985982, 5.0840054750442505, 5.527954975763957, 5.649938861529033, 5.4537193775177]
EAM_PMP =  [4.878248572349548, 4.880806128184001, 5.1912699937820435, 5.455279548962911, 6.241715947786967, 5.051068862279256, 5.443765838940938, 5.306729833285014, 5.495205044746399, 5.64631974697113, 5.204754551251729, 5.818707982699077, 5.552535573641459, 5.713460405667623, 5.360990047454834, 5.730958183606465, 5.083736022313436, 5.556286334991455, 5.38953693707784, 4.4899477163950605, 5.82724146048228, 5.223268032073975, 6.6776150067647295, 5.373807311058044, 5.369427522023519]
EAM_ZGZ =  [5.305404901504517, 5.452130436897278, 5.964117407798767, 5.552022695541382, 5.304312825202942, 6.033575177192688, 5.169136683146159, 5.0955227216084795, 4.950789372126262, 5.087150692939758, 5.589207808176677, 5.221441109975179, 5.675882935523987, 5.10338830947876, 6.290579954783122, 5.236805160840352, 5.246491154034932, 5.032366037368774, 5.3210233847300215, 5.136250019073486, 5.52970274289449, 6.283945759137471, 5.834049264589946, 5.834804018338521, 5.083638866742452]
EAM_HSC =  [5.272233327229817, 5.904285152753194, 5.375299572944641, 5.527841528256734, 5.07549508412679, 4.98647944132487, 5.93374224503835, 4.987348635991414, 5.6956840356191, 5.874344865481059, 6.0067975123723345, 5.28115729490916, 4.9825679063797, 4.8914644320805865, 5.743874390920003, 6.02927045027415, 5.487539052963257, 5.601582169532776, 5.610217293103536, 5.4546509981155396, 5.249972224235535, 5.6708931128184, 5.2002813418706255, 5.606161952018738, 4.926159818967183]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=3)

EAMRn =  [7.383422930216052, 7.008330354985502, 8.222349579801264, 10.699423091927754, 8.108273024411545, 9.76388990264578, 7.805588004515343, 7.409234784313084, 7.678172199996476, 8.403114318847656, 7.721116764029277, 7.236557400103697, 9.143344761170063, 9.671894506080863, 10.827095346352488, 7.0301446816355915, 8.239770122410096, 7.96983624487808, 7.679597913604422, 8.373981672463957, 8.775462278385753, 6.751404083881182, 7.807109400169137, 7.814447658578145, 8.549939578341455]
EAM_BCN =  [5.944850489036324, 5.215296794458763, 5.36394547924553, 6.1697955180689235, 4.832291829217341, 6.4361229729406615, 7.4099107722646185, 5.551391522908948, 5.6948975631871175, 4.78953143247624, 6.8518195004807305, 5.408027373638349, 5.6573789144299695, 4.851821860087287, 5.849031625334749, 5.85903596386467, 5.948857218948836, 5.404987256551526, 5.919570490257027, 5.76951335631695, 5.730293116618678, 5.149875050967502, 5.5737133616024686, 5.546107734601522, 5.637033521514578]
EAM_PMP =  [5.180089301669721, 5.158153140667787, 5.275980251351583, 6.23943482231848, 5.04840909820242, 6.244193401533304, 5.979090110542848, 5.151133743758054, 5.952020979419197, 5.355395012295123, 5.156127064498429, 5.627561706857583, 5.860884400987134, 5.319421630544761, 5.088429598464179, 6.017114187024303, 5.177316685312802, 6.312439594072165, 5.629380530917768, 4.782803722263611, 5.475602926667204, 5.336153600633759, 5.598712842489026, 5.7061119866125365, 6.253697582126892]
EAM_ZGZ =  [5.158116802726824, 4.562306669569507, 5.09413787999104, 6.022229892691386, 5.565652866953427, 5.312041095851623, 5.1270923024600314, 5.082295486607502, 6.413608629678943, 5.649430107824581, 5.532536182206931, 5.910764635223703, 4.990140895253604, 5.962463968807889, 5.13728934219203, 6.2474202421522635, 5.3843191481128185, 5.815879507163136, 5.326253949981375, 5.440431929126229, 5.255575199717099, 5.87535401963696, 6.019986929352751, 5.304262102264719, 5.2153494136849625]
EAM_HSC =  [5.4313027686679485, 5.237415942949118, 5.643505332396202, 6.3055277558946115, 5.702661140677855, 5.162437360311292, 5.371933455319748, 5.900192418049291, 6.5836352712100314, 7.192434291249698, 5.523431640310386, 5.410289882384625, 4.952235664289022, 5.44400319364882, 5.279731829141833, 6.161695657317171, 4.472994774887242, 5.708543207227569, 5.875157936332152, 5.898671848257792, 5.7198710884015584, 5.616674796822145, 5.394539901890705, 5.886229092312842, 5.258955414762202]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=4)

EAMRn =  [8.129219561207051, 9.291915815703723, 9.232651457494619, 7.24677697006537, 8.192336413325096, 9.99023421929807, 11.4468453076421, 7.875728295773876, 9.938700306172274, 7.604354975174885, 9.960118196448501, 7.690531827965561, 7.47366325222716, 7.015528542654855, 7.696821173843072, 8.523174908696388, 7.369429763482541, 10.503863899075256, 7.949842180524554, 6.619432838595643, 7.061773222319934, 7.739328579026825, 7.357518449121592, 7.060841268422652, 8.946395796172473]
EAM_BCN =  [5.836270663203026, 5.2125061580113, 6.086043104833486, 6.571354729788644, 5.059065760398398, 4.791291567744041, 6.733518678314832, 5.778489949751873, 6.648460543885523, 5.937420008133869, 5.215019887807418, 5.83025581982671, 6.355946832773637, 5.0735408627257055, 6.454363569921377, 5.525402263719208, 5.360467755064672, 5.928773802153918, 5.185207795123665, 5.366259282949019, 5.956027439662388, 5.987470354352679, 5.224542345319476, 5.692251127593371, 6.446965003500179]
EAM_PMP =  [6.801223910584742, 6.120996202741351, 6.1176682297064335, 5.522751671927316, 5.381114531536491, 6.094263193558674, 6.023691333070094, 5.148112433297293, 5.045921053205218, 6.237141823282047, 6.282663695666255, 5.844255719866071, 5.0040327967429645, 5.8100380410953445, 6.155578574355768, 6.11981641029825, 5.796654097887934, 5.7944581946548155, 5.887545877573442, 5.486886121788803, 5.3419539393210895, 5.836837690703723, 5.565901970376774, 6.5560294560023715, 5.784016005846919]
EAM_ZGZ =  [5.142125927672094, 5.485275346405652, 5.154255808616171, 6.096619080524055, 5.283513477870396, 6.667123989183075, 6.389388765607562, 5.547967638288226, 7.108685318304568, 5.42737104454819, 6.180479049682617, 5.247854622042909, 5.876549312046596, 5.339367574574996, 5.427892490309112, 6.797992667373346, 6.470530646187918, 5.337268050836057, 7.229261826495735, 5.745140776342275, 6.742325763313138, 5.7044364384242465, 5.517204673922792, 5.92933464050293, 6.633615260221521]
EAM_HSC =  [5.571999608253946, 6.142632036792989, 6.874008100859973, 5.08209567167321, 5.584388149027922, 6.503062773723991, 5.726753468416175, 5.7932824504618745, 5.613761746153539, 5.688630551707988, 8.02549206480688, 5.5236859224280535, 5.6782486973976605, 5.538776280928631, 5.3525522971639825, 6.267358429577886, 5.714100584691884, 5.900104133450255, 5.831785318802814, 5.606637721159021, 6.047462580155353, 6.745572343164561, 6.624835345209862, 5.029524433369539, 5.5732458075698545]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=5)

EAMRn =  [7.199096371429135, 10.949726913914535, 7.102279971344302, 7.441390644420277, 7.7245609013721195, 7.843273894955414, 6.9303672096946025, 10.087811787923178, 7.4139529912158695, 7.037238804980962, 7.98885541973692, 7.325238853994042, 9.77881236991497, 8.419398336699516, 9.031919845426925, 8.122400996660945, 7.551524229723998, 6.7503954184175745, 8.45966015440045, 8.998330935083255, 8.657038833155777, 8.20618924227628, 7.664829485344164, 6.757492990204782, 7.72144556527186]
EAM_BCN =  [6.200277925741793, 6.416804650817254, 6.179425634519018, 6.888036477445352, 6.040578206380208, 5.6221412119239265, 8.126888621937145, 6.220502024949199, 5.710843673860184, 6.104337865656072, 6.458299925833037, 6.84414580374053, 5.682802681971078, 6.527304254396998, 6.335997764510338, 5.314191606309679, 6.254194664232658, 5.6717836784593985, 5.322226129396998, 6.5139556653571855, 7.324559953477648, 6.293901886602844, 5.711406707763672, 6.261309113165344, 6.250125307025331]
EAM_PMP =  [6.25135822488804, 5.653029239538944, 6.052485880225595, 6.213072304773813, 5.768804530904751, 6.356389864526614, 6.013093158452198, 6.366147744535196, 6.14330203123767, 5.402340975674716, 5.78957879422891, 6.626635965674814, 6.735841462106416, 6.651772470185251, 6.191725297407671, 6.04858240455088, 5.878512141680477, 5.762817768135456, 6.200726441662721, 5.952221186474116, 5.671349612149325, 6.119587869355173, 6.277855150627367, 6.7239251570268115, 6.618282549309008]
EAM_ZGZ =  [5.790242667150015, 6.307063709605824, 5.989267060250947, 5.993648683181917, 5.622263474897905, 6.250742555868746, 5.469685255879104, 6.874623809197937, 6.341244090687145, 6.008141604336825, 6.179399509622593, 6.81519309920494, 6.958572002372357, 6.941296317360618, 7.330992958762429, 5.662385150639698, 6.172636552290483, 7.419635387382122, 6.440009839607008, 7.6734107431739265, 6.390396773213088, 6.272186818749014, 7.551807673290522, 6.975240110146879, 6.5648230735701745]
EAM_HSC =  [6.473346324882122, 5.654957434143683, 6.09938727484809, 5.830078279129182, 5.502605862087673, 5.712336607653685, 5.506876126684324, 6.2202474806043835, 7.001520214658795, 6.628868411285708, 6.170096079508464, 6.607831473302359, 6.244195610585839, 5.243697195342093, 5.920332590738933, 6.397632984199909, 6.106873252175071, 6.908258611505682, 6.3791557466140905, 6.200909932454427, 6.084986253218218, 5.558804829915364, 6.920706007215712, 7.23733428030303, 6.706886445633089]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=6)

EAMRn =  [7.634746589660645, 7.3251203918457035, 7.642132148742676, 10.017069549560548, 7.526558418273925, 7.6144905853271485, 7.463948707580567, 7.781013717651367, 8.233566665649414, 7.457447204589844, 7.365788459777832, 7.662135314941406, 6.9268246841430665, 7.659941215515136, 7.904061889648437, 7.701684913635254, 9.450319519042969, 6.53350902557373, 7.7340107345581055, 7.904002952575683, 7.609865913391113, 7.206285667419434, 8.244847869873047, 7.189037399291992, 7.118951454162597]
EAM_BCN =  [6.376288909912109, 5.936005172729492, 6.425348129272461, 5.303470268249511, 5.905623474121094, 6.138012199401856, 6.217509307861328, 7.261514320373535, 5.44273609161377, 7.287875595092774, 6.984950981140137, 6.6134400558471675, 6.656462249755859, 6.0692057800292964, 6.496013450622558, 6.079669799804687, 5.6202128219604495, 6.203231430053711, 5.708571128845215, 6.264457015991211, 5.154493026733398, 6.4616310501098635, 6.417467384338379, 6.381855506896972, 6.519448356628418]
EAM_PMP =  [6.03772762298584, 5.825994491577148, 7.135335006713867, 6.692838935852051, 6.031383857727051, 6.095298461914062, 5.5719568634033205, 5.826842346191406, 6.6103487396240235, 6.503559684753418, 6.40533447265625, 5.541371383666992, 7.490607566833496, 6.483073692321778, 7.087529716491699, 5.88082260131836, 6.1028789520263675, 6.539832801818847, 6.327079124450684, 6.904300270080566, 7.113219947814941, 5.886056632995605, 6.498428726196289, 5.5759416198730465, 6.190700492858887]
EAM_ZGZ =  [7.771866149902344, 6.2091646575927735, 6.436640701293945, 5.778949241638184, 5.88134349822998, 6.72303840637207, 5.1053112030029295, 6.599088821411133, 6.913335189819336, 5.647949333190918, 7.031970825195312, 6.8168315505981445, 7.0119046783447265, 6.540394287109375, 6.687944831848145, 6.693549690246582, 6.186533851623535, 6.210292739868164, 6.455731849670411, 6.283222160339355, 6.818985290527344, 5.724887428283691, 6.061352653503418, 6.221818542480468, 7.333361663818359]
EAM_HSC =  [6.3759619140625, 6.342533340454102, 7.505606422424316, 5.923218002319336, 5.7591727828979495, 7.4564830017089845, 7.053232002258301, 7.172579307556152, 6.517566604614258, 6.400135803222656, 6.428466186523438, 6.381308403015137, 6.124355697631836, 7.0874489974975585, 6.839044380187988, 6.218432235717773, 5.3892274093627925, 5.593617897033692, 6.74344108581543, 6.5718217468261715, 6.6114241409301755, 6.3065561676025395, 6.688242301940918, 6.380804023742676, 5.9828462982177735]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=7)

EAMRn =  [7.523451701249226, 7.575774617714457, 6.877513696651648, 7.233407426588606, 6.944661470923093, 7.472314626863687, 7.684462556744566, 6.417123020285427, 10.557862461203396, 7.344552106196337, 8.662767995702158, 7.898867125558381, 8.48074990451926, 6.857475771762357, 7.054249376353651, 7.073422271426361, 7.415463853590559, 10.971930928749613, 7.680888751945873, 8.978835304184715, 7.8296966552734375, 7.885672616486502, 7.470109127535679, 7.455340092725093, 7.762498836706181]
EAM_BCN =  [7.39896559007097, 6.532154347636912, 5.778212103513208, 6.474661156682685, 6.3059525442595525, 6.138221438568418, 7.897956319374614, 6.553287203949277, 6.981728431021813, 5.862395673695177, 7.175595802835899, 6.152990208994044, 5.8187985939554645, 6.089630278030245, 6.765801118151976, 6.533870281559406, 6.007267489291654, 6.154031885732518, 5.713145983101118, 6.053199919143526, 6.187069770133141, 6.381566831381014, 7.461573723519202, 7.675532010522219, 6.314121624030689]
EAM_PMP =  [7.351970785915261, 6.694169261667988, 6.53300921751721, 6.733084990246462, 6.578093311574199, 7.253820211580484, 6.930420866106997, 6.303191383286278, 6.889874486639949, 6.850836687748975, 6.151573143383064, 5.969518680383663, 5.594288136699412, 5.737820842478535, 6.319064111992864, 7.335155638137667, 6.3701717641093945, 6.0713792555403, 5.846357100080736, 6.113558136590637, 7.235246412824877, 6.536574713074335, 6.4794802146382855, 6.940768289093924, 6.62665044199122]
EAM_ZGZ =  [6.922124881555538, 8.206456174944886, 6.767517731921507, 6.777180435633896, 7.029661310781346, 6.923155907357093, 7.537207272973391, 6.399668778523361, 5.563989620397587, 7.410240588801922, 7.240042620366163, 7.194678127175511, 6.021708309060276, 6.589297568443978, 7.115434401106127, 7.0742080612938, 6.3141670982436375, 6.519824188534576, 6.3305831002716975, 5.939432465203918, 6.3511611636322325, 6.110691750403678, 7.421193302267849, 5.959505591062036, 6.417587393581277]
EAM_HSC =  [6.923871937364635, 5.970928041061552, 6.097729635710763, 6.69537074022954, 6.670694823312287, 7.450865188447556, 7.050157112650352, 6.990697086447536, 6.903124327706818, 6.855186273555945, 6.722893667693185, 6.7442976318963686, 7.0795450871533685, 6.005106066713239, 6.541240918754351, 6.3994786857378365, 7.474131518071241, 6.424845591630086, 6.7819556812248605, 6.582725713748743, 6.503312139227839, 6.074368051963277, 6.712837974623878, 6.003633857953666, 6.44954522765509]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=8)