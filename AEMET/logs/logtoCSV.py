__author__ = '@Tssp'
import numpy as np
import pandas as pd

def ErrorToCSV(Rn, BCN, PMP, ZGZ, HSC, nFw):
    DF = pd.DataFrame({'Rn': Rn,
                  'BCN': BCN,
                  'PMP': PMP,
                  'ZGZ': ZGZ,
                  'HSC': HSC})
    DF.to_csv(f'~/CIEMAT/Rn_Time_Series/AEMET/logs/ErrorsLSTM{nFw}Fw.csv', index=False) #Change name whenever you want
## Input the lists here
EAMRn =  [8.740292041859728, 8.923653298235955, 8.120191492932909, 8.357837839329497, 8.19908304417387, 9.713374523406333, 8.98495808053524, 8.617079633347531, 8.113136981395964, 8.847826693920378, 8.704604574974548, 8.150504619517225, 8.45872842504623, 8.868662489221451, 8.280050683528819, 8.694553172334711, 8.69999374227321, 8.560444689811543, 8.810224533081055, 8.293744432165267, 8.255912334360975, 8.024007837823097, 9.106537230471348, 8.57328727397513, 8.225111251181744]
EAM_BCN =  [8.568378773141415, 8.166869264967898, 10.345800643271588, 9.297386128851707, 8.202106841067051, 8.63140008804646, 8.953370642154775, 11.692721995901554, 8.13191661428898, 8.813823497041742, 10.196903634578623, 9.481500300955265, 8.224398755012675, 7.924781434079434, 8.578013237486495, 8.572098062393513, 7.889691535462725, 8.210852075130381, 9.013468153933262, 9.743206957553296, 8.111238398450487, 8.72965082209161, 8.673402015199052, 9.140831683544402, 8.417772455418364]
EAM_PMP =  [9.490086656935672, 8.400683139232878, 8.503120828182139, 9.467893072899352, 7.884717819538523, 8.60446994862658, 8.360340970627805, 8.200742031665559, 10.708492644289707, 10.088512745309384, 9.540443258082613, 9.691635253581595, 8.684984775299721, 8.643906897686897, 11.743698241862845, 8.58407491318723, 9.065539299173558, 10.68966601757293, 8.651823977206616, 8.622044705330058, 8.364645694164519, 8.35351542209057, 9.267672072065638, 8.634873653980012, 9.667460624207841]
EAM_ZGZ =  [8.812480683022358, 8.822776469778507, 9.364415797781437, 8.26473723066614, 10.317609584077875, 9.331793440149186, 9.23577832161112, 8.405286261375915, 8.375080514461436, 9.330122927401929, 10.78006082900027, 9.882615231453105, 8.685359792506441, 8.047293480406417, 9.685985808676861, 8.29406746397627, 8.528231681661403, 9.001134953600294, 9.030725763199177, 8.574767011277219, 9.164805432583423, 9.350000787288584, 8.55878720384963, 8.334454232073846, 8.481241632015147]
EAM_HSC =  [8.50014905726656, 8.552723661382148, 8.686123300105967, 8.232391803822619, 10.052355583677901, 10.59395631830743, 8.526187531491543, 11.800713275341277, 10.863056020533785, 9.51692512187552, 8.846138203397711, 8.128977552373359, 9.694562952569191, 9.939195876425885, 11.103523984868476, 8.6140271044792, 8.203127597240691, 8.89121319385285, 8.547739799986495, 10.670206394601376, 10.234264617270611, 9.057499418867396, 11.044167538906665, 8.561771068167179, 8.44239421600991]
# Apply the function:
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=1)

EAMRn =  [4.455577488949424, 4.0284477635433795, 4.808561345150596, 4.55860700105366, 4.253813452469675, 4.241019640470806, 4.257499534205387, 5.049767303466797, 4.503654399671053, 5.384844689620168, 4.678232253225226, 4.340913190339741, 4.352413619192023, 4.457686012669614, 4.150560921116879, 4.2152660169099505, 5.329453920063219, 4.187962381463302, 4.475576782226563, 4.609924758108039, 4.605782960590563, 4.755347522936369, 4.402031908537213, 5.921355518541838, 4.307796960127981]
EAM_BCN =  [5.560730100932874, 5.643220680638364, 5.258025079024465, 5.151800095407586, 6.024271633750514, 5.942677146510074, 5.379977898848685, 5.298909076891447, 5.09706991095292, 6.01291813097502, 5.0304201628032486, 6.3744759409051195, 5.650898060045744, 6.366256071391859, 4.949563116776316, 5.483727706106086, 5.787289549175061, 5.084082954808285, 5.453283370168585, 5.968217709189967, 5.4209551761024874, 6.109376646343031, 5.193467391164679, 5.342863986366673, 5.451419187846937]
EAM_PMP =  [5.788248323139391, 5.411420601292661, 6.993508067883943, 7.146261877762644, 6.2151327835886105, 5.422031442742599, 7.142214805201481, 6.131400379381682, 6.188386174252159, 5.0679878636410365, 5.183023713764391, 5.80337347733347, 6.272190294767681, 5.778470691881682, 5.6457876908151725, 5.091684963828639, 5.732838721024363, 5.213669264943976, 5.448038402356599, 5.609979729903372, 5.184521685148541, 4.97642139635588, 6.4965676960192225, 5.458539661608245, 5.649384147242496]
EAM_ZGZ =  [5.620032300447162, 5.434848825555099, 5.558265726189864, 5.0885983517295434, 5.916837190326891, 5.748614903500205, 5.169785710384971, 6.389354344418174, 5.422781452379729, 6.516178331877056, 5.687227992007607, 5.29447330675627, 5.852668119731702, 5.614428831401624, 6.372023451955695, 6.212695673892372, 5.491360835025185, 6.716392918636925, 5.3850555821468955, 6.992427665308902, 5.205578010960629, 4.833020782470703, 6.03615578099301, 5.3134411460474915, 6.000254339920847]
EAM_HSC =  [5.253145157663446, 5.715057533665707, 5.314892417506168, 5.105509828266345, 5.956104599802118, 5.240567297684519, 6.14235819766396, 5.662800678453947, 6.171455142372533, 5.564318084716797, 5.724386476215563, 6.702545607717414, 6.05616908826326, 5.780523922568873, 5.485541775352076, 4.818582113165604, 5.726784073679071, 5.137520438746402, 5.577313272576583, 5.658423815275493, 5.173361406828228, 6.174643747430099, 5.41750797472502, 5.642092413651316, 5.433038089149877]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=2)

EAMRn =  [3.7490967909495034, 3.935135801633199, 4.099624395370483, 3.9446632464726767, 4.098543286323547, 4.097099184989929, 4.959032615025838, 4.254788517951965, 4.417349179585774, 3.963659405708313, 3.747764070828756, 4.101221084594727, 4.114190459251404, 4.350411454836528, 4.0512925783793134, 4.329038778940837, 4.0070193608601885, 4.523640791575114, 3.584690570831299, 3.8198084831237793, 4.471408883730571, 3.829350749651591, 4.571760773658752, 5.013952573140462, 3.579583923021952]
EAM_BCN =  [4.898000160853068, 5.53286349773407, 5.644812266031901, 4.921801765759786, 4.794505715370178, 4.634541114171346, 4.163359642028809, 4.427578568458557, 4.621647079785665, 4.611921588579814, 4.482564091682434, 4.549000064531962, 4.819433291753133, 4.484278162320455, 4.2953232526779175, 4.474859317143758, 5.125451922416687, 5.775393565495809, 5.219390948613484, 4.822721878687541, 4.637261470158895, 5.312360922495524, 4.452442487080892, 4.763244231541951, 5.292954484621684]
EAM_PMP =  [4.803870876630147, 4.863439480463664, 5.723843336105347, 4.499903043111165, 4.858681797981262, 4.525212685267131, 4.10196320215861, 5.484429399172465, 4.622074882189433, 4.842626611391704, 5.194111347198486, 4.440966526667277, 4.5732366641362505, 4.656860629717509, 4.393186767896016, 4.709552884101868, 4.927796522776286, 4.862605253855388, 4.890125155448914, 4.829675118128459, 5.25032099088033, 4.588352878888448, 5.215224424997966, 4.804404457410176, 4.619945168495178]
EAM_ZGZ =  [5.518187681833903, 4.387360572814941, 4.580492814381917, 5.4329514900843305, 5.093539555867513, 3.929577628771464, 4.1465797026952105, 4.647925059000651, 4.706991036732991, 5.259838898976644, 4.120768308639526, 4.990888595581055, 4.3003567059834795, 4.414419412612915, 4.965130090713501, 4.659240206082662, 4.446655909220378, 4.811709046363831, 4.751657883326213, 4.858033299446106, 5.241779327392578, 7.015673955281575, 5.223951657613118, 5.304585417111714, 4.180247227350871]
EAM_HSC =  [4.660356442133586, 4.55318820476532, 5.515566468238831, 4.819697419802348, 4.719329476356506, 5.16957147916158, 4.7789896329243975, 4.414321064949036, 4.868115981419881, 4.747110684712728, 4.968623121579488, 4.695639729499817, 4.695580283800761, 4.727117816607158, 5.081104238828023, 4.33245313167572, 4.315590739250183, 4.923653960227966, 4.607247551282247, 4.9411312739054365, 4.690238833427429, 4.280343572298686, 4.590883294741313, 4.504361589749654, 4.758095860481262]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=3)

EAMRn =  [3.7535898660876086, 3.7440922923923767, 4.268204797174513, 4.286951084726865, 4.2214724747176025, 3.94523231270387, 3.7702077098728455, 4.082586426095864, 4.065829326197044, 3.794252729907478, 4.263910863817353, 3.7828718755663058, 3.6952596841399203, 3.9912616690409553, 3.60695652126037, 4.3669776129968385, 4.012712065706548, 3.9831421449012363, 3.5688262624838916, 3.826485899305835, 3.9479416916050862, 4.246147784990134, 3.7079297488497707, 5.148702011894934, 3.9530327787104342]
EAM_BCN =  [4.713615614114349, 4.351932211020558, 4.3916899297655245, 4.088296359347314, 4.473935432040815, 5.051824392731657, 4.73205605733026, 4.616262455576474, 4.505703916254732, 4.348551248766713, 4.679461095750947, 4.39832510407438, 4.208468250392639, 4.7335250697185085, 4.515195748240678, 4.373472076101401, 4.688722512156693, 4.323883371254833, 4.563498506840971, 4.603093294753242, 4.353038551881141, 4.731103366183252, 4.562999489381141, 4.268700078590629, 4.149218608423607]
EAM_PMP =  [4.185755739506987, 4.54363364780072, 4.693391072381403, 4.877403455911224, 4.562856772511275, 4.318534536460011, 4.2444024233473945, 4.391523066255235, 4.9516135540205175, 4.205146356956246, 4.119496217707998, 4.0676866511708685, 4.492780311820433, 5.154512543039224, 4.64726457890776, 4.457690563398538, 4.522708971475818, 4.427265639157639, 4.292114375792828, 4.411620228560929, 4.318019630982704, 4.304951461320071, 4.667528880011175, 4.178702914837709, 4.946071034854221]
EAM_ZGZ =  [4.27077110526488, 4.682766983189533, 4.38070045549845, 4.6372359364303115, 4.8496634493169095, 4.118922086106133, 4.78612711011749, 4.314863696540754, 4.2079853569109416, 4.400750268365919, 4.795604902444427, 4.5839750152273275, 4.335850666478737, 3.9492351099387886, 4.3937185621753185, 4.734768346412895, 4.876228293192756, 4.663461704844052, 4.653849100329213, 4.6726976374989935, 4.690262706009383, 4.653891022672358, 4.264557435340488, 4.371663634310064, 4.361620637559399]
EAM_HSC =  [4.847478139031794, 4.394455585283103, 4.632651535506101, 4.3949215682511475, 4.651475100173164, 4.48302970964884, 4.314809032322205, 4.434040934769149, 4.668061521864429, 4.236763315102489, 4.501681534285398, 4.968398418623147, 4.742753608939574, 4.353845340689433, 4.554823531317957, 4.25064043654609, 4.4573768537069105, 4.670786513495691, 4.389137897294821, 5.050203932929285, 4.344370458543915, 4.270831550519491, 4.523073255401297, 5.0461089144048, 4.362188830818098]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=4)

EAMRn =  [3.863845552716936, 4.18499806462502, 4.511896094497369, 4.023006789538325, 3.522846221923828, 4.092941050626794, 3.6964547293526784, 3.798047669079839, 4.099916029949577, 4.099651180967992, 3.7769654332375038, 4.2938026895328445, 4.068790357940051, 3.7671948647012514, 3.7079263803910236, 4.101219838979293, 4.249411329931142, 3.8267517479098574, 4.127935837726204, 4.22561991944605, 3.8789333421356824, 3.970127533893196, 3.9653556590177574, 4.012158530099051, 4.125013740695253]
EAM_BCN =  [4.5225793098916816, 4.179971422467913, 4.074484611044125, 4.188092367989676, 4.2569592145024515, 4.206096454542511, 4.2602549183125396, 4.340472591166594, 4.545226583675462, 4.5010683487872685, 4.2348028689014665, 4.0318920758305765, 4.385552503624741, 4.450800058793049, 4.226976939610073, 4.215424362494021, 4.245093092626455, 4.332494035059092, 4.688531136026188, 4.547010032498107, 4.331177886651487, 4.269830820511799, 4.306191269232302, 4.266518573371732, 4.3465167065056]
EAM_PMP =  [4.31743049621582, 4.332794656558913, 4.115844259456712, 4.44876495672732, 4.1675368328483735, 4.609159158200634, 4.329074236811424, 4.605621221114178, 4.606719776075714, 4.628368416611029, 4.6031191300372685, 4.41373163340043, 4.396809052447884, 4.136537123699577, 4.288679706807039, 4.210488844890984, 4.582238216789401, 4.167971358007314, 4.265827178955078, 4.16513306753976, 4.04195345664511, 4.432728670081314, 4.0291367355658085, 4.369783829669563, 4.302700782308773]
EAM_ZGZ =  [4.119294769909917, 4.674192662141761, 4.4397802936787505, 4.353041512625558, 4.658250108057139, 4.241477187798948, 4.559807991494938, 4.206830978393555, 4.3562725612095425, 4.275843094806282, 4.498034963802415, 4.841319376108598, 4.877701039216956, 4.354166536915059, 4.2278289794921875, 4.690738016245317, 4.470618150672134, 4.4207234674570515, 4.177968511776048, 4.237525589612066, 4.7443544815997685, 4.394366789837273, 4.261269043902962, 4.563127478774713, 4.595968830342195]
EAM_HSC =  [4.538964018529775, 4.252817621036452, 4.436831415915976, 4.392038695666255, 4.354197054493184, 4.506837883774115, 3.946049554007394, 4.147842134748187, 4.487446454106545, 4.4353720995844625, 4.331634171155034, 4.290889778915717, 4.188948495047433, 4.3021482350874924, 4.465493649852519, 4.391399811725227, 4.869547435215542, 4.374265631850885, 4.562290386277802, 4.3173584062225965, 4.722684899154975, 4.339980611995775, 3.997980390276228, 4.280013220650809, 4.221853917958785]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=5)

EAMRn =  [3.9852788905904752, 4.145421423093237, 3.881951572919133, 4.18863350454003, 3.767420951766197, 4.659655253092448, 4.35555876144255, 4.09724568839025, 3.637829212227253, 4.190481359308416, 3.968402284564394, 3.8134551385436395, 3.89594261092369, 4.296548477326981, 4.221207358620384, 4.149276001284821, 4.327160921963778, 4.2178687278670495, 4.220445189813171, 4.444453114210957, 4.210034341523142, 3.906887670960089, 3.8449693544946535, 4.302564293447167, 4.2375373455009075]
EAM_BCN =  [4.046204846314709, 4.040172499839706, 4.685046070753926, 4.495942241013652, 4.184757117069129, 3.7179661375103574, 4.451569451226129, 4.483305690264461, 4.0918665028581716, 4.061098580408578, 4.2657039912060055, 4.3369563324282865, 4.364223595821496, 4.709327004172585, 4.392598816842744, 4.260243618127071, 4.236445263178662, 4.427123483985361, 4.063006969413372, 3.9967451769896227, 4.389470765084932, 3.7600777365944604, 4.6847651703189115, 4.430569812504932, 4.418576712560172]
EAM_PMP =  [4.522126978093928, 4.422045967795632, 3.9083717808578955, 4.15222449254508, 4.518603893241497, 4.269108473652541, 4.209798639470881, 4.117873490458787, 4.458968191435843, 4.738320437344638, 4.579587416215376, 4.208901607629024, 4.353738033410274, 4.0207787330704505, 4.2437441276781485, 4.739982566448173, 4.08612403484306, 4.250116290468158, 4.3525157889934505, 4.4905096111875595, 4.721846532340002, 4.515718864672111, 4.292978460138494, 4.268210362906408, 4.300922971783263]
EAM_ZGZ =  [4.282978096393624, 4.407190072416055, 3.967267392861723, 4.751790826970881, 4.724977666681463, 4.049876126376065, 4.35462023031832, 4.463838866262725, 3.9239118942106614, 4.623210829917831, 4.323416854396011, 4.427659314088147, 4.530854388920948, 4.293589274088542, 4.302986877133148, 4.06069695829141, 4.362655716713029, 4.755554507477115, 4.474281311035156, 4.509138974276456, 4.2860736268939394, 4.196624678794784, 4.35602087926383, 4.435392707285255, 4.80823605469983]
EAM_HSC =  [4.34828012639826, 4.123099510115806, 4.472355505432746, 4.164515331538037, 3.988585924861407, 4.154062675707268, 3.8311819596724077, 4.549815592139658, 4.316645073168205, 4.171349188294074, 4.379411523992365, 4.732625132859355, 4.532634272719875, 4.631559314149799, 4.662138120092527, 4.694140347567472, 4.354658724081637, 4.388097512601602, 3.981193696609651, 4.265063854178997, 4.266150773173631, 4.038706577185429, 4.2924296446520875, 4.172076408309166, 4.687234897806187]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=6)

EAMRn =  [4.196964492797852, 4.46706714630127, 4.05839599609375, 4.239179573059082, 4.4313004684448245, 4.406066436767578, 4.522392616271973, 4.472535018920898, 4.5317246627807615, 4.3813671875, 4.708159904479981, 4.1767950439453125, 4.268525886535644, 4.594416236877441, 4.1765971374511714, 4.321331367492676, 4.2620757293701175, 4.7704290008544925, 4.217694549560547, 4.549561882019043, 4.427413825988769, 4.212901649475097, 4.228718490600586, 4.5136754608154295, 4.577945556640625]
EAM_BCN =  [4.2608295440673825, 4.370547218322754, 4.398656349182129, 4.490412101745606, 4.368522109985352, 4.575956916809082, 4.561670837402343, 4.64585018157959, 4.594339790344239, 4.0843816757202145, 4.20224365234375, 4.498267402648926, 4.57490795135498, 4.395052909851074, 4.422470283508301, 4.6830665969848635, 4.3176026535034175, 4.06838550567627, 4.44373664855957, 4.41038257598877, 4.20467041015625, 4.3913623809814455, 4.940188674926758, 5.101396560668945, 4.253580856323242]
EAM_PMP =  [4.638762550354004, 4.269898338317871, 4.162276268005371, 4.062205696105957, 4.255702743530273, 4.227354583740234, 4.13007926940918, 4.139345817565918, 4.168551635742188, 4.4216220092773435, 4.545370483398438, 4.232826538085938, 4.413426361083984, 4.319552345275879, 4.167181243896485, 4.38985538482666, 4.292587127685547, 4.162406425476075, 4.367906379699707, 4.731299972534179, 4.534853324890137, 4.199484634399414, 4.3123962020874025, 4.503843078613281, 4.467704620361328]
EAM_ZGZ =  [4.20795970916748, 4.672216682434082, 4.46587574005127, 4.469555511474609, 4.226943702697754, 4.444907112121582, 4.324973983764648, 4.258256950378418, 4.448083114624024, 4.035605354309082, 4.628864631652832, 4.517797546386719, 4.263526840209961, 4.065397109985351, 4.598621063232422, 4.305875358581543, 3.980795021057129, 4.252764663696289, 4.345308532714844, 4.276197547912598, 4.137822608947754, 4.273823966979981, 4.3464361190795895, 4.317210464477539, 4.496767044067383]
EAM_HSC =  [4.136552238464356, 4.060865783691407, 4.328389015197754, 4.223659210205078, 4.216705741882325, 4.229723815917969, 4.80277084350586, 4.875481338500976, 3.9619440841674805, 4.206061515808106, 4.73862922668457, 4.581286277770996, 4.348853988647461, 4.4212167358398435, 4.4471408462524415, 4.48928581237793, 4.366962051391601, 4.593421936035156, 4.295030670166016, 4.628790359497071, 4.217835159301758, 4.186346778869629, 4.462295265197754, 4.719895401000977, 3.964511947631836]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=7)

EAMRn =  [4.952022099258876, 4.306361019021214, 4.26186740988552, 4.726178537500967, 4.4243533823749805, 4.51902978727133, 4.6349671996466, 4.058059352459294, 4.364924440289488, 4.5486496273833925, 3.908836742438892, 4.239707795700224, 4.4244385898703396, 4.727986968389832, 4.725974130158377, 4.833825404101079, 4.294783072896523, 4.219673194507561, 4.16641378874826, 4.716363548052193, 4.208407222634495, 4.471608379099629, 4.192955205936243, 4.6698664674664485, 3.950010620721496]
EAM_BCN =  [4.5551265301090655, 4.442658263858002, 4.754272272091101, 4.770976906955832, 4.467828316263633, 4.124462845301864, 4.451589527696666, 4.163911385111289, 4.090965686458172, 4.661146447210029, 4.601703266106029, 4.4623016886191795, 4.26703141467406, 4.278884812156753, 4.229369550648302, 4.289549534863765, 4.283918588468344, 4.240715971087465, 4.020021721868232, 4.592235338569868, 4.352377334443649, 4.938398531167814, 4.588327124567315, 4.22532049500116, 4.534412346263923]
EAM_PMP =  [4.302367975216101, 4.569309801158338, 4.286858511443185, 4.311108881884282, 4.54335890666093, 4.2171695444843555, 4.6781670598700495, 4.131775506652228, 4.737325196218963, 4.306437501812925, 4.20899766978651, 4.551973059625909, 4.283445301622447, 4.335623297360864, 4.229037483139794, 4.192019774182008, 4.475881066652808, 4.265643355870011, 4.54401155981687, 4.8452159768283956, 4.46077316586334, 4.532220594953783, 4.430184808107886, 4.281146096711112, 4.316790816807511]
EAM_ZGZ =  [4.44193426453241, 4.355932821141611, 4.400962111973526, 4.28339144262937, 4.348963444775874, 4.228848485663386, 4.037641053152557, 4.522119276594408, 3.947612422527653, 4.477335750466526, 4.649943096802966, 3.9431044701302405, 4.370834879355855, 4.487304158730082, 4.1215880743347775, 4.42228891353796, 4.798356386694578, 4.3939046198778815, 4.291640819889484, 4.460815807380299, 4.595799247817237, 4.535867634386119, 4.122596476337697, 4.552540882979289, 4.465033125169207]
EAM_HSC =  [4.436383842241646, 4.2556378581736345, 4.208272650690362, 4.170663172655766, 4.81834623129061, 4.285636826316909, 4.319722355002224, 4.300356949910079, 4.522970785008798, 4.0974226092348, 4.030206699182491, 4.378465520273341, 4.509145264578338, 4.305668179351504, 4.566404965844485, 4.426421665909267, 4.193589880914971, 5.2742003261452854, 4.813206889841816, 4.619743875937886, 4.646716165070487, 4.418379528687732, 4.36189862997225, 4.509591999620494, 4.5258571133755225]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=8)