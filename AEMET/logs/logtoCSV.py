__author__ = '@Tssp'
import numpy as np
import pandas as pd

def ErrorToCSV(Rn, BCN, PMP, ZGZ, HSC, nFw):
    DF = pd.DataFrame({'Rn': Rn,
                  'BCN': BCN,
                  'PMP': PMP,
                  'ZGZ': ZGZ,
                  'HSC': HSC})
    DF.to_csv(f'~/CIEMAT/Rn_Time_Series/AEMET/logs/ErrorsGRU{nFw}Fw.csv', index=False) #Change name whenever you want
## Input the lists here
EAMRn =  [9.000444818050303, 8.349712655899372, 8.562880860998275, 8.61004512868029, 8.431340887191448, 8.226686274751703, 8.15916808108066, 8.684225488216319, 9.624847696182576, 8.245558190853037, 8.092400976952087, 8.4308764680903, 8.60714137300532, 8.094565087176385, 8.128054355053191, 8.183357847497819, 8.608384680240713, 8.106878483549078, 8.9718534997169, 8.515732582579268, 8.787836642975503, 8.702615048022981, 8.629383046576317, 8.645570673841112, 8.159410517266457]
EAM_BCN =  [7.918629139027697, 8.729513411826275, 8.28811669856944, 8.43143682276949, 8.918506135331823, 9.00656306489985, 8.282266373329975, 8.359258002423225, 8.66384136930425, 8.174177007472261, 8.546623351726126, 9.046607605954433, 8.646341445598196, 8.134835872244327, 8.305248179334276, 9.34954253663408, 7.945531033455057, 8.353503977998773, 7.941071165368912, 9.497509043267433, 10.018753619904214, 9.114655149743912, 8.604092171851624, 9.302760225661258, 8.93390018381971]
EAM_PMP =  [9.784764918875187, 8.746080723214657, 8.59907361294361, 8.64389062435069, 8.79542862100804, 9.425362201447182, 8.212053501859625, 8.185307888274497, 8.249972079662566, 10.85758505476282, 8.97399979449333, 8.36470575535551, 8.224503699769365, 8.886394054331678, 10.837092257560567, 9.396071291984395, 8.7429513728365, 8.518495762601813, 8.142027469391518, 8.265465837843875, 8.929044682928856, 8.566862309232672, 8.14279044942653, 8.857801762033017, 9.255191802978516]
EAM_ZGZ =  [8.527840715773563, 8.32880316389368, 9.141596692673703, 9.182865467477352, 10.341135430843272, 8.434837625381794, 8.619585199558989, 8.292468700003116, 8.559270412363905, 8.51243355933656, 9.029055818598321, 8.537811360460646, 9.44547141866481, 8.02253106299867, 8.175560443959338, 8.410097122192383, 8.804744233476354, 8.364050317317881, 8.48618758992946, 8.220125685346888, 8.472667044781623, 7.951036088010098, 8.782702588020488, 8.919350197974671, 7.8020577126360955]
EAM_HSC =  [11.14376457701338, 8.92574139858814, 8.904983479925926, 8.431339142170359, 10.298480825221285, 8.817125563925885, 8.02083843312365, 9.359588663628761, 8.99174913446954, 8.309420159522523, 8.758913486561877, 7.98653570134589, 8.148237796539956, 8.934624448735663, 7.720164562793488, 8.6217364047436, 8.183876849235372, 9.205182461028404, 8.036592402356737, 8.700467292298661, 7.492284409543301, 10.029107357593293, 8.548922477884496, 8.278837163397606, 8.492715551498089]
495, 10.670206394601376, 10.234264617270611, 9.057499418867396, 11.044167538906665, 8.561771068167179, 8.44239421600991]
# Apply the function:
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=1)

EAMRn =  [4.243077127557052, 4.753308466861123, 4.175265262001439, 4.79469532213713, 4.563143238268401, 4.365763734516345, 4.334007423802426, 4.6833549097964635, 4.246612669292249, 4.171058012309827, 5.008455417030736, 4.402913063450863, 4.13618722212942, 4.813780172247636, 4.159981095163446, 4.114958391691509, 4.502826128507915, 4.816983032226562, 4.5156604967619245, 4.936781431499281, 4.46438630756579, 4.362284971538343, 4.411615351626748, 4.090620141280325, 4.580667515804893]
EAM_BCN =  [5.337232248406661, 5.383632097746196, 5.063294260125411, 6.223927668521279, 5.168038217644942, 5.324299420808491, 5.741225192421361, 5.282501622250206, 4.781378213982833, 4.813793985467208, 5.98046854922646, 4.976773593300267, 5.811867001182154, 5.633893183657998, 5.179742833187706, 5.189588285747328, 5.5618429886667355, 5.02312300832648, 5.3781882436651935, 5.035185803865131, 5.536609729967619, 4.848017883300781, 5.736052663702714, 5.519458369204872, 5.682808484529194]
EAM_PMP =  [4.75684316534745, 4.688304459421258, 5.329621927361739, 4.729292698910362, 5.44351268567537, 4.83034149973016, 5.2716113441868835, 5.23910574662058, 5.541031405800267, 5.217665180407073, 5.376066107498972, 5.06130909166838, 5.32961590415553, 5.094270445171156, 5.238864978991057, 5.218041309557463, 4.897162748637952, 4.844835702996505, 5.784702461644223, 5.410463072124281, 4.914197540283203, 7.092816603811164, 5.195248372931229, 6.821920053582442, 4.973299167030736]
EAM_ZGZ =  [5.31286516691509, 5.327782681113796, 5.319626456812808, 5.542825116609272, 6.026833825362356, 5.303130220112048, 4.764710878071032, 4.928944356817948, 5.582468735544305, 5.126517165334601, 5.89256053723787, 5.081989930805407, 5.6211047122353, 5.868737110338713, 5.5638786315917965, 5.74970060649671, 4.931042038766962, 4.909197516190378, 4.536751195004112, 5.060553661145662, 4.98071413542095, 4.796813282213713, 5.900701984606291, 5.429880644145765, 5.1435701470626025]
EAM_HSC =  [6.94412841796875, 5.2812246623792145, 6.45523039165296, 5.432317713687294, 5.293152658562911, 5.2910310042531865, 5.638648504959909, 7.093604599802117, 5.348588401392886, 4.84461151926141, 4.924547456440172, 6.107208533036081, 4.950648699308696, 5.454328878302324, 5.595828809236226, 5.865972900390625, 5.060071403101871, 4.945800741095292, 5.067465812281559, 4.705112216347143, 5.964335030003598, 5.565589503238075, 5.364225086412932, 6.041085413882607, 4.92149947317023]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=2)

EAMRn =  [3.606408635775248, 3.572542349497477, 3.8771291176478067, 4.034246166547139, 3.6635631322860718, 3.6914013624191284, 3.6434195041656494, 3.5768818855285645, 3.6583271821339927, 3.722114086151123, 3.9909435907999673, 3.8967487812042236, 4.045818130175273, 3.9025398890177407, 3.7953693866729736, 4.0168607632319135, 3.6722711324691772, 3.4152382612228394, 4.14865513642629, 3.9544774691263833, 3.951229770978292, 3.7540091276168823, 3.9758307933807373, 4.349425633748372, 4.2368520100911455]
EAM_BCN =  [4.7893234093983965, 4.290756821632385, 4.413326581319173, 4.2757556438446045, 4.538016041119893, 4.678666989008586, 4.361314574877421, 4.320657730102539, 4.94240415096283, 4.491933703422546, 4.535069982210795, 4.495493372281392, 4.445746898651123, 5.532538692156474, 4.539170106252034, 4.471248229344686, 4.824944257736206, 4.260373036066691, 4.002591609954834, 4.010493000348409, 4.344085256258647, 5.0562797387441, 4.715538303057353, 4.912005066871643, 4.166307171185811]
EAM_PMP =  [4.920416474342346, 5.135318756103516, 4.763395587603251, 4.235755920410156, 5.108957688013713, 4.580274780591329, 5.303570667902629, 4.8174781401952105, 4.248739719390869, 4.257191975911458, 4.6323684851328535, 4.327699581782023, 4.385396440823873, 4.688929319381714, 4.294856588045756, 4.216776808102925, 4.672071258227031, 4.346182942390442, 4.415519595146179, 4.977826317151387, 4.4489439725875854, 4.6879072189331055, 4.583765069643657, 5.020648121833801, 4.77842132250468]
EAM_ZGZ =  [4.62359086672465, 4.899671316146851, 4.53639288743337, 4.6147821346918745, 4.787959973017375, 4.440513491630554, 4.508014520009358, 4.546189387639363, 4.654877940813701, 4.431081533432007, 4.6847240924835205, 4.98079784711202, 4.7942808866500854, 4.604560852050781, 4.676695307095845, 4.9993990659713745, 4.458614269892375, 4.528416991233826, 4.035430828730266, 4.455737749735515, 4.832101225852966, 4.428772608439128, 5.6771490176518755, 4.316216468811035, 4.0330677429835005]
EAM_HSC =  [4.3848468859990435, 4.4811627467473345, 4.336761434872945, 4.504581252733867, 4.329248428344727, 4.750150243441264, 4.507692297299703, 4.854531327883403, 4.725682179133098, 4.877050956090291, 4.7500254313151045, 4.355985442797343, 4.272101362546285, 4.515087286631267, 4.370684623718262, 4.198924422264099, 4.404033303260803, 6.309581120808919, 4.8351512749989825, 4.405075867970784, 5.011096318562825, 4.589003165562947, 4.532977183659871, 4.81303056081136, 4.349568923314412]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=3)

EAMRn =  [4.206115761983026, 3.902941674301305, 4.027360660513652, 3.6224513889588033, 4.041706124531854, 3.847152474000282, 3.869150771308191, 3.339285545742389, 4.1672448620353775, 3.741682426216676, 3.629092501610825, 3.6326614851804124, 3.9559994727065884, 3.7773286091912652, 3.85638553580058, 3.948927259936775, 4.0355222249768445, 3.6682936284959933, 3.2714415874677836, 3.80211328722767, 3.6957286166161607, 3.8105065257278916, 3.759575322731254, 3.3851944834915635, 3.3039770224659715]
EAM_BCN =  [3.890589723881987, 4.87193034850445, 4.255074058611369, 4.577548784079011, 4.48843446711904, 4.733856201171875, 4.6874261049880195, 4.328387663536465, 4.526907301440682, 4.149771267605811, 4.602658379938185, 5.082597830860885, 4.377580072461944, 5.119124697655747, 4.310050807048365, 4.605517239914727, 4.511831263905948, 4.265030536454978, 5.045421364381141, 4.37677650844928, 4.432108613633618, 4.488682697728737, 4.618169450268303, 4.361702043985583, 4.71472399996728]
EAM_PMP =  [4.688738242867067, 4.32438270332887, 4.21500664150592, 4.379994775831085, 4.686860349989429, 4.412585425622685, 4.661683544670184, 4.253670053383739, 4.30658906759675, 4.433126233287694, 4.508240768589924, 4.604964620059299, 4.290904566184762, 4.305049188358268, 5.0135374954066325, 4.763810973806479, 4.597106343692111, 4.730332561374939, 4.383508465953709, 4.439339981865637, 4.120717274773981, 4.584621547423687, 4.373232929976945, 4.238475484946339, 4.796814082824078]
EAM_ZGZ =  [4.4186898850903065, 4.379623924334025, 4.704027431527364, 4.33904797268897, 4.454019998766713, 4.232521135782458, 4.321824575207897, 4.212261199951172, 4.859215765884242, 4.439462327465568, 4.675886331145296, 4.618944423714864, 4.132535324883215, 4.52576293158777, 4.6983107340704535, 4.654451586536525, 4.316125692780485, 4.098482151621396, 4.270543560539324, 4.58407171976935, 4.22609946654015, 4.530207211209326, 4.278754637413418, 4.440853787451675, 4.284418794297681]
EAM_HSC =  [4.643857975596005, 4.237986417160821, 4.582314245479623, 4.536294996123953, 4.449214895975959, 4.10067414745842, 4.579854276991382, 5.465910724757873, 4.307199261852147, 4.572738765441265, 4.28822126093599, 4.2492643926561495, 4.41079629327833, 4.3924035534416275, 4.446086136336179, 4.4578399658203125, 4.30788280054466, 4.466027643262725, 4.937034921547801, 4.3169421559756564, 4.588480212024806, 4.522623121123953, 4.446178239645417, 4.254926701181943, 4.641805707793875]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=4)

EAMRn =  [3.7278270332180723, 3.9020838445546677, 4.0623616199104156, 3.90975959933534, 3.726034164428711, 3.789939530041753, 3.8759122576032365, 3.663306528208207, 3.450118746076311, 3.9697759589370416, 3.833116609223035, 3.5153662233936545, 3.637443464629504, 3.661110391422194, 4.103724654839963, 3.803656870005082, 3.735092513415278, 3.4568120216836733, 4.153839111328125, 3.5031871795654297, 3.8255408150809154, 3.8800000949781768, 4.163266318184989, 3.9089907821343868, 3.939042227608817]
EAM_BCN =  [4.368818010602679, 4.6236277210469146, 4.2409799926135, 4.358165468488421, 4.24463077467315, 4.561450179742307, 4.154421086214026, 4.061948192362883, 4.416999310863261, 4.425851705122967, 4.3732057688187576, 4.348333592317542, 3.9630894563636003, 4.236096129125478, 4.258750720899933, 4.111997448668188, 4.63056790098852, 4.505112394994619, 4.9093365571936785, 4.517441263004225, 4.346680076754823, 4.372792068792849, 4.69433255098304, 4.711817877633231, 4.347834606559909]
EAM_PMP =  [4.473344218974211, 4.398435320172991, 4.111279779550981, 4.146151250722457, 4.69318860890914, 4.710637112053073, 4.591715365040059, 4.290386628131477, 4.278151492683255, 4.3860251757563375, 4.2518216736462655, 4.574068886893136, 4.1946748616744065, 4.4365730674899355, 3.8458789514035594, 4.610673281611229, 4.300678525652204, 4.331123507752711, 4.276264502077686, 4.454694864701252, 4.3633992331368585, 4.129899355829979, 4.132718572811204, 4.40801612698302, 4.315013613019671]
EAM_ZGZ =  [4.024457191934391, 4.326636762035136, 4.162015058556381, 4.115769016499422, 4.431588893034021, 4.176248083309251, 4.244204929896763, 4.437071780769193, 4.415889428586376, 4.297089596183932, 4.239844380592813, 3.795859823421556, 4.485647707569356, 4.19079574273557, 4.195340137092435, 4.31095349058813, 4.180207933698382, 4.011128016880581, 4.740314872897401, 4.794497042286153, 4.258062479447346, 4.083898193982183, 4.111202979574398, 4.305248182647082, 4.264222592723613]
EAM_HSC =  [4.208998816353934, 4.235959617459044, 4.388882695412149, 4.363296781267438, 4.610842451757314, 4.154496990904516, 4.469562102337273, 4.373315460827886, 4.278920504511619, 4.251472940250319, 3.760616497117646, 4.183422789281728, 4.064661298479352, 4.215675976811623, 3.9712338739511917, 4.463799301458865, 4.544893498323401, 4.44145899402852, 4.299233806376555, 4.197125220785336, 4.244846694323481, 3.8758961035280812, 4.3084458331672515, 4.584129839527364, 4.274439441914461]

ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=5)

EAMRn =  [3.9852788905904752, 4.145421423093237, 3.881951572919133, 4.18863350454003, 3.767420951766197, 4.659655253092448, 4.35555876144255, 4.09724568839025, 3.637829212227253, 4.190481359308416, 3.968402284564394, 3.8134551385436395, 3.89594261092369, 4.296548477326981, 4.221207358620384, 4.149276001284821, 4.327160921963778, 4.2178687278670495, 4.220445189813171, 4.444453114210957, 4.210034341523142, 3.906887670960089, 3.8449693544946535, 4.302564293447167, 4.2375373455009075]
EAM_BCN =  [4.046204846314709, 4.040172499839706, 4.685046070753926, 4.495942241013652, 4.184757117069129, 3.7179661375103574, 4.451569451226129, 4.483305690264461, 4.0918665028581716, 4.061098580408578, 4.2657039912060055, 4.3369563324282865, 4.364223595821496, 4.709327004172585, 4.392598816842744, 4.260243618127071, 4.236445263178662, 4.427123483985361, 4.063006969413372, 3.9967451769896227, 4.389470765084932, 3.7600777365944604, 4.6847651703189115, 4.430569812504932, 4.418576712560172]
EAM_PMP =  [4.522126978093928, 4.422045967795632, 3.9083717808578955, 4.15222449254508, 4.518603893241497, 4.269108473652541, 4.209798639470881, 4.117873490458787, 4.458968191435843, 4.738320437344638, 4.579587416215376, 4.208901607629024, 4.353738033410274, 4.0207787330704505, 4.2437441276781485, 4.739982566448173, 4.08612403484306, 4.250116290468158, 4.3525157889934505, 4.4905096111875595, 4.721846532340002, 4.515718864672111, 4.292978460138494, 4.268210362906408, 4.300922971783263]
EAM_ZGZ =  [4.282978096393624, 4.407190072416055, 3.967267392861723, 4.751790826970881, 4.724977666681463, 4.049876126376065, 4.35462023031832, 4.463838866262725, 3.9239118942106614, 4.623210829917831, 4.323416854396011, 4.427659314088147, 4.530854388920948, 4.293589274088542, 4.302986877133148, 4.06069695829141, 4.362655716713029, 4.755554507477115, 4.474281311035156, 4.509138974276456, 4.2860736268939394, 4.196624678794784, 4.35602087926383, 4.435392707285255, 4.80823605469983]
EAM_HSC =  [4.34828012639826, 4.123099510115806, 4.472355505432746, 4.164515331538037, 3.988585924861407, 4.154062675707268, 3.8311819596724077, 4.549815592139658, 4.316645073168205, 4.171349188294074, 4.379411523992365, 4.732625132859355, 4.532634272719875, 4.631559314149799, 4.662138120092527, 4.694140347567472, 4.354658724081637, 4.388097512601602, 3.981193696609651, 4.265063854178997, 4.266150773173631, 4.038706577185429, 4.2924296446520875, 4.172076408309166, 4.687234897806187]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=6)

EAMRn =  [3.7533526611328125, 4.241487274169922, 4.336471900939942, 4.060837936401367, 4.095240669250488, 4.003857879638672, 3.829657211303711, 3.864851951599121, 4.2682591247558594, 3.5850336837768553, 4.100414810180664, 3.801966361999512, 4.183148040771484, 3.8625107192993164, 3.8587365341186524, 4.225034255981445, 3.904011650085449, 3.9636911392211913, 3.915298728942871, 3.7352838134765625, 4.074429359436035, 3.819208755493164, 3.819045219421387, 4.30169605255127, 3.842276954650879]
EAM_BCN =  [4.021566886901855, 4.223763771057129, 4.08941837310791, 4.656446800231934, 4.116229553222656, 4.517872543334961, 4.3478679275512695, 4.436008186340332, 4.242064895629883, 4.216308631896973, 4.088694801330567, 4.3274470138549805, 4.1379563522338865, 4.46468002319336, 4.547192115783691, 4.716913948059082, 4.032287940979004, 4.342548217773437, 4.145879592895508, 4.347478828430176, 4.25864875793457, 4.278014221191406, 4.391259307861328, 4.763085250854492, 4.549625854492188]
EAM_PMP =  [4.212772331237793, 4.230829772949218, 4.122470703125, 4.4887911987304685, 4.409502601623535, 4.2256432723999025, 4.181822204589844, 3.8557263565063478, 4.5746779632568355, 4.273147430419922, 4.581426200866699, 4.298981819152832, 4.603575439453125, 4.003382530212402, 4.26109130859375, 4.558569717407226, 4.424758415222168, 4.169933204650879, 4.4110187149047855, 4.353902053833008, 4.095330543518067, 4.373003005981445, 4.344386558532715, 4.151196174621582, 4.063701438903808]
EAM_ZGZ =  [4.513472099304199, 4.384644584655762, 4.452014350891114, 4.199510650634766, 4.314134368896484, 4.013456573486328, 4.247576713562012, 4.278156013488769, 4.513744163513183, 4.267616539001465, 4.4755488204956055, 4.426329879760742, 4.523608055114746, 4.320463371276856, 4.112490730285645, 4.201494369506836, 4.196569099426269, 4.282483863830566, 4.120729713439942, 4.464745788574219, 4.207312088012696, 4.297920341491699, 4.287797164916992, 4.373654060363769, 4.2091513442993165]
EAM_HSC =  [4.6499395751953125, 4.47149169921875, 4.182150077819824, 4.2302317047119145, 4.416854209899903, 4.402963104248047, 4.009306716918945, 4.462158241271973, 4.272747116088867, 4.602614288330078, 4.308685188293457, 4.299767837524414, 4.444028778076172, 4.462868843078613, 4.4319248962402344, 3.960057716369629, 4.329862251281738, 4.186188774108887, 4.103828163146972, 4.389150238037109, 4.351916160583496, 4.478188323974609, 3.947095947265625, 4.143416290283203, 4.389231872558594]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=7)

EAMRn =  [3.894643047068379, 3.653440796502746, 3.9196946172430964, 3.7795925706920057, 3.9249635828603613, 4.194459405275855, 3.684923077573871, 4.308417405232345, 4.312130805289391, 4.160896414577371, 4.033841954599513, 3.8345425671870164, 3.848336965730875, 4.073487763357635, 3.7028076625106356, 3.882261861668955, 4.122644367784557, 4.202464604141689, 4.240354972310586, 4.369300124668839, 4.518615307194172, 3.912219396912225, 4.368034249485129, 3.7343929781772123, 4.0644911208955365]
EAM_BCN =  [4.498572170144261, 4.553658173816038, 4.350047951877707, 4.128473036360033, 4.838475784452835, 4.457661052741627, 4.3754453753480815, 4.38621184849503, 4.4379668660683205, 4.631054302253346, 4.634287050454923, 4.3611021136293315, 4.379484006673983, 4.419039093621887, 4.348880012436669, 4.651327076524791, 4.306974127741143, 4.749216551827912, 4.32424526403446, 4.339296208749904, 3.9339765416513575, 4.334653797716197, 4.251224215668027, 4.4224672223081685, 4.443586708295463]
EAM_PMP =  [4.304831514264097, 4.393754675836846, 4.237070782349841, 4.09675152467029, 4.507930717845954, 4.601851623837311, 4.266725559045772, 4.209246758187171, 4.612445113682511, 4.150819381864944, 4.146456576810024, 4.55349089367555, 4.263573523795251, 4.40432863896436, 4.235530362270846, 4.542193611069481, 4.689849324745707, 4.402301637252958, 4.415062026222153, 4.159379033759089, 4.240727755102781, 4.44289224454672, 4.334632571380918, 4.359151896863881, 4.459756152464612]
EAM_ZGZ =  [4.723765193825901, 4.19483502076404, 4.126455325891476, 4.326942103924138, 4.107972815485284, 4.422249633486908, 4.567704984457186, 4.602744036381788, 4.485884071576713, 4.493836733374265, 4.370713639967512, 4.147004269137241, 4.5426440097317835, 4.971593271387686, 4.376697880206722, 4.276563361139581, 4.559680334412225, 4.450220126916866, 4.3875065794085515, 4.261151795340057, 4.4292126268443495, 4.306041113220819, 4.207029172689608, 4.634478200780283, 4.577938079833984]
EAM_HSC =  [4.0338825943446395, 4.287449449595838, 4.2582855224609375, 4.281458259809135, 4.281348879974668, 4.312520622026803, 4.61141314364896, 4.187675324997099, 4.377041618422706, 4.1099119280824565, 4.609243449598256, 4.112040415848836, 4.3174992551898015, 4.544374786981262, 4.253210728711421, 4.876399842819365, 4.771323780022045, 4.355725316718074, 4.168266485233118, 4.257371354811262, 4.093725789891611, 4.143651603472115, 4.4055410328477915, 4.167588791044632, 4.476899817438409]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=8)