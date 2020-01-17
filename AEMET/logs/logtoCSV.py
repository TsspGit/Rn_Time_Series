__author__ = '@Tssp'
import numpy as np
import pandas as pd

def ErrorToCSV(Rn, BCN, PMP, ZGZ, HSC, nFw):
    DF = pd.DataFrame({'Rn': Rn,
                  'BCN': BCN,
                  'PMP': PMP,
                  'ZGZ': ZGZ,
                  'HSC': HSC})
    DF.to_csv(f'~/CIEMAT/Rn_Time_Series/AEMET/logs/ErrorsANN{nFw}Fw.csv', index=False) #Change name whenever you want
    
## Input the lists here
EAMRn =  [10.581104156818796, 10.597705719318796, 8.95735663556038, 8.247473777608668, 8.236245865517475, 8.450228670810132, 11.710065760511034, 9.027224155182534, 10.303569712537401, 10.208518576114736, 9.449290255282788, 10.128199678786258, 9.31849788097625, 9.087384406556474, 12.352315740382418, 8.210627819629426, 8.52417852523479, 10.217723318871032, 10.281077851640417, 8.314987710181702, 8.239511205794964, 8.171931084166182, 8.233295400091942, 9.534385640570457, 9.08535133524144]
EAM_BCN =  [7.96779385018856, 8.258521871363863, 8.067553865148666, 8.213813173009994, 8.388636933996322, 8.186554279733212, 8.23741340637207, 8.306841748826047, 8.099844181791266, 8.644968235746344, 8.25624116938165, 8.217245345420025, 8.31235926202003, 8.048737627394656, 8.241551176030585, 8.342641181134162, 8.577736306697764, 8.204775789950757, 8.487287886599278, 8.912953884043592, 8.64372675469581, 8.469143928365504, 8.43527806058843, 8.309308315845247, 8.291143863759142]
EAM_PMP =  [8.27260971069336, 8.255017787852186, 8.223483795815326, 9.185555397196019, 8.327454262591424, 8.293066511762904, 8.307491464817778, 8.229597538075549, 8.270174148234915, 8.230721047584046, 8.423907665496177, 9.197831052414914, 8.01008220429116, 10.082203154868267, 8.328439306705556, 8.540894285161444, 8.580192078935339, 8.24542110524279, 8.307662720375873, 8.347048982660821, 8.232629532509662, 8.70083999633789, 8.229715956018326, 8.857709600570354, 8.161881223638007]
EAM_ZGZ =  [8.285629272460938, 9.085222041353266, 8.456757241107049, 8.287602526076297, 8.21048606710231, 8.094553927157788, 8.123980745356134, 8.18193719742146, 8.195762918350544, 9.167774930913398, 8.308684207023457, 8.613008864382481, 8.138693261653819, 8.278737818941156, 8.224622158294029, 8.189602141684674, 8.26195380028258, 9.202496630080203, 8.390990886282413, 8.382673304131691, 8.250029259539666, 8.640642369047125, 8.185717481247922, 8.289734049046293, 8.093882946257896]
EAM_HSC =  [8.161884104951898, 8.274343409436815, 8.343612021588264, 8.088140528252785, 8.611909663423578, 8.411075389131586, 8.315024517952128, 8.290486315463452, 8.205943574296667, 8.407476059933925, 8.259061448117519, 8.741553651525619, 8.12931559948211, 8.696127384266955, 8.078928156101957, 8.22383401748982, 8.387798268744286, 8.54787883352726, 8.241009570182637, 9.200153391411964, 8.501158328766518, 8.028868411449675, 8.037233230915476, 8.160631504464657, 8.135690689086914]
# Apply the function:
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=1)

EAMRn =  [9.880740517064146, 9.998881530761718, 8.11953353881836, 9.556399335359272, 8.618888935289885, 8.592060570967824, 7.587674512361225, 8.187305249665913, 7.982001535516036, 9.697531409012644, 6.696960690146998, 9.73315554167095, 8.889953452662418, 7.655782398424651, 11.575583407753392, 7.848883056640625, 10.237398729826275, 8.341073528089021, 7.999158799020868, 7.87209183542352, 8.100619506835937, 11.164641129343133, 9.158659523411801, 8.741007915296052, 10.90646659449527]
EAM_BCN =  [6.324986307244552, 6.357231581838508, 6.8231548510099715, 6.284435874537418, 6.587933148835835, 7.519052806653474, 5.325741978695518, 6.366140345523232, 6.544312728078742, 6.470848966899671, 6.608321782162315, 5.753058704576994, 6.143629054019326, 6.025583206979852, 5.745066391794305, 5.435178094161184, 6.0157983880294, 6.277760716488487, 7.122351315146998, 6.83402408800627, 5.617164651971114, 5.616879312615645, 6.27932438097502, 5.512631225585937, 6.318729199861226]
EAM_PMP =  [6.078783015200966, 5.3496340299907486, 6.0451100801166735, 6.313876423082854, 6.007308558413857, 5.42646195261102, 6.432560729980469, 7.256528593364515, 5.2166492662931745, 7.039212517989309, 5.793521078009355, 6.18548355102539, 6.3309814453125, 6.125536908601459, 6.935459136962891, 6.124358448229338, 6.273029407701994, 6.65717897917095, 5.432272178248355, 6.223389474969161, 6.495887997275904, 5.851900442023027, 6.331579148141961, 6.1285240173339846, 6.710039801346628]
EAM_ZGZ =  [6.607335903770045, 5.999748149671053, 7.033010181627776, 6.8473864505165505, 5.987203337016859, 6.197192182038959, 6.490613435444079, 5.7973980953818876, 6.356552605879934, 6.15353176719264, 5.558351135253906, 6.175447684840152, 6.238901881167763, 6.0805502640573605, 5.013307872571443, 6.036050816586143, 6.381568226061369, 6.485184197676809, 7.0971517864026525, 5.548935980545847, 6.579517766049034, 5.61929104453639, 6.698200346294202, 6.026549329255756, 5.9935141312448605]
EAM_HSC =  [5.241947374845806, 5.571686754728619, 6.111731599506579, 7.1746885600842925, 5.439632937782689, 6.763962916324013, 6.777764491031045, 6.567471835487767, 6.327899009303043, 6.019517958791632, 5.12187230963456, 6.481628779361123, 6.066818679006476, 6.341996885600843, 7.339731878983347, 5.500122873406661, 6.615395234760485, 6.674745901007402, 6.171475300035978, 6.997959337736431, 5.57933205052426, 5.996448597155119, 5.772379865144428, 6.114144094366776, 6.168980206941304]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=2)

EAMRn =  [7.9093442757924395, 10.391881306966146, 7.415862520535787, 10.358901500701904, 9.390706857045492, 11.44815973440806, 7.314517895380656, 10.539958834648132, 6.1762421528498335, 8.982629934946695, 7.481685519218445, 10.145270427068075, 8.522885123888651, 7.384389440218608, 10.638948122660318, 12.264597853024801, 6.337928215662639, 8.304092725118002, 5.116148273150126, 8.12887692451477, 7.337352991104126, 8.14232603708903, 8.254424254099527, 8.231483896573385, 10.071879307428995]
EAM_BCN =  [5.23324187596639, 5.633693297704061, 5.522133549054463, 4.6083182493845625, 5.104920148849487, 6.363452156384786, 4.853762785593669, 4.979216297467549, 5.33648955821991, 5.223660270373027, 5.1254829565684, 4.3379448254903155, 5.360606869061788, 5.262920618057251, 4.419289271036784, 6.492438991864522, 6.171358982721965, 5.421299735705058, 5.350168466567993, 4.697426358858745, 5.10857625802358, 4.978649338086446, 5.343359708786011, 4.771495183308919, 4.664789199829102]
EAM_PMP =  [5.222173611323039, 5.31292998790741, 5.42160705725352, 4.228197534879048, 5.352372805277507, 5.027087052663167, 4.691741903622945, 4.630957325299581, 5.192378600438436, 4.834132552146912, 4.818626403808594, 5.008797367413838, 5.381787618001302, 4.967156847318013, 5.1218440135320025, 5.097584366798401, 4.917782306671143, 5.658592661221822, 4.978079915046692, 5.4360880851745605, 5.187272508939107, 4.900980154673259, 4.431302785873413, 5.1187431414922075, 5.021980444590251]
EAM_ZGZ =  [4.897401730219523, 5.079087853431702, 5.071682373682658, 4.357151985168457, 4.992110212643941, 5.34828519821167, 4.996670405069987, 4.901075800259908, 5.008656104405721, 5.772928913434346, 5.152998963991801, 5.154646674791972, 4.723044315973918, 5.0350567897160845, 5.4433969259262085, 5.543257355690002, 4.6851439873377485, 5.43924617767334, 4.849577903747559, 5.5012632211049395, 5.2053654591242475, 4.772641539573669, 5.135607719421387, 5.43820337454478, 5.158190528551738]
EAM_HSC =  [5.910768230756124, 5.162447810173035, 5.283419171969096, 5.085898796717326, 5.243786096572876, 5.2547091245651245, 4.923824111620585, 5.82889187335968, 5.012461066246033, 5.062581658363342, 5.269904772440593, 4.426479538281758, 4.430337429046631, 4.993951201438904, 5.283889333407084, 5.215342362721761, 4.588442087173462, 4.939471483230591, 5.147585868835449, 5.714196681976318, 4.877458532651265, 5.468334436416626, 4.745297392209371, 4.733845313390096, 4.341796000798543]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=3)

EAMRn =  [8.857896549185527, 6.317901572001349, 11.033739109629208, 8.362243730997301, 9.279116129137806, 6.946716583881182, 9.324571471853355, 10.971974402358851, 7.050562868413237, 7.391269447877235, 7.224249967594736, 8.625386267593226, 10.209288017036988, 6.7957456136487195, 11.429189819650551, 7.756806285110946, 7.704739914726965, 6.291193814621758, 9.485740936908526, 11.997007900906592, 6.602036505630336, 10.132343528196984, 7.479502215827863, 7.23052640305352, 8.74642484212659]
EAM_BCN =  [4.866345592380799, 5.352028364987717, 5.090855234677029, 4.891555078250846, 5.0003218110074705, 4.717446199397451, 4.500287871999839, 4.711069087392276, 4.795132981133215, 4.467667805779841, 4.608115166732945, 4.832925737518625, 4.61605791701484, 4.334631025176687, 4.969387487037895, 4.791469927915593, 4.758982982832132, 4.437518444257913, 3.95985676087055, 4.552257773802452, 3.922027941831608, 5.729909090651679, 4.845954383771444, 4.8371207247075345, 4.047490346063044]
EAM_PMP =  [5.2954569551133615, 4.551336583402968, 5.038119483239872, 4.810901445211823, 5.422992548991725, 4.949922109387584, 4.863628623411827, 4.649335094333924, 4.70398287429023, 4.883415654762504, 3.843529691401216, 5.233021451025894, 4.805643494596186, 4.3400179676173885, 4.165301765363241, 4.977466150657418, 4.420836025906592, 4.871015529042666, 4.693828858051104, 5.045259534698172, 4.29491916145246, 4.470503620265686, 4.592991740433211, 4.693936770724267, 4.835015051143685]
EAM_ZGZ =  [5.661048731853053, 5.161342935463817, 4.990675149504671, 5.312272533927996, 5.382877389180291, 4.674359311762545, 5.157318626482462, 4.77891847276196, 4.594491978281552, 5.5821345614403794, 4.461590049193077, 4.615779798055432, 5.758415772742832, 4.680844218460555, 5.557385867403955, 4.45992853223663, 4.714529529060285, 4.994204570337669, 5.175861083355146, 4.582127836561694, 4.819306521071601, 4.842753734785257, 3.9654062802029637, 5.293171794144149, 5.151058118367932]
EAM_HSC =  [4.34610551657136, 4.562690538229401, 5.800164684806902, 4.8867648016546195, 5.127414152794278, 4.579647850744503, 5.597704385973744, 4.984555037980227, 5.303108411965911, 5.080754466892517, 4.728609222726724, 4.3043103562187905, 5.291350413843529, 4.254864997470502, 4.298685211496255, 4.741587805993778, 4.90218569568752, 4.106610799573131, 4.931823848449078, 5.030060718968971, 5.892149935063627, 4.630316980106315, 5.186143894785459, 4.847231756780565, 4.401133232509967]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=4)

EAMRn =  [13.299905037393374, 9.10944973692602, 8.647231043601522, 9.300279734086017, 8.35601452418736, 8.867346043489418, 7.549337581712372, 9.468735208316724, 7.829199304386061, 7.247728775958626, 8.048210221893934, 6.995538633696887, 7.690962616278201, 13.871963539902044, 6.864202810793507, 7.660147530691964, 14.37921111437739, 12.30308486004265, 10.718727384294782, 5.164315865964306, 11.846510478428431, 8.060163342222875, 8.824912635647522, 7.196752664994221, 8.646096171164999]
EAM_BCN =  [4.708319839166135, 4.704037919336436, 4.434140964430206, 4.858494233111946, 4.972125812452667, 4.69184887165926, 4.534136090959821, 4.895461062995755, 5.279911080185248, 5.379054789640466, 4.7189969821852085, 4.143571464382872, 4.520246155407964, 4.30203001839774, 4.768755737616091, 4.432671877802635, 4.999102884409379, 4.331177302769253, 4.8943003051135, 4.254874832776128, 4.817174989349988, 4.839725805788624, 4.976691732601243, 4.313010974806183, 5.025805103535554]
EAM_PMP =  [5.077488568364357, 4.419039434316207, 4.720878017191985, 4.209895114509427, 4.674786587150729, 4.593883203000439, 4.533270310382454, 4.487852719365334, 5.536712957888233, 4.792502734125877, 4.7834029295006575, 4.794197549625319, 4.530743929804588, 5.055578737842794, 5.143230048977599, 4.59979045634367, 4.795485126728914, 4.642309149917291, 5.049805738488022, 4.209297063399334, 4.906932830810547, 4.436246288066008, 4.065397262573242, 5.680438800733917, 4.642457689557757]
EAM_ZGZ =  [5.056410731101523, 4.814097190389828, 4.117606026785714, 4.700556930230588, 4.995604495612943, 4.384593768995636, 4.559988021850586, 4.774543801132514, 4.467735640856684, 5.118238254469269, 4.439602559926558, 4.522733921907386, 4.9254540813212495, 4.439044563137755, 4.524481559286312, 4.006676771202866, 4.4219976931202165, 5.642049283397441, 4.552058161521445, 4.890947692248286, 4.585176545746473, 4.412137829527563, 4.444358280726841, 4.496264905345683, 4.952017531103017]
EAM_HSC =  [4.600640511026188, 4.4527411168935345, 3.719135673678651, 4.183275962362484, 4.380141394478934, 4.125748965204979, 4.717261256003867, 5.117828680544483, 4.724652854763732, 4.293121727145448, 5.214604591836735, 4.842701347506776, 4.504816561329122, 4.04917896037199, 4.580660644842654, 4.866889603283941, 4.402821287816884, 4.695724409453723, 4.964240716428173, 4.040937657258948, 4.864628422017, 4.726313104434889, 4.745452919784857, 4.248256099467375, 4.983369554792132]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=5)

EAMRn =  [6.9093659911492855, 7.784783758298315, 6.792947981092665, 7.227920493694267, 7.1539429558648004, 8.703483273284604, 6.103275915588996, 6.541139198072029, 9.00677455555309, 7.02453956218681, 7.632309345283893, 8.584325424348465, 10.28397014887646, 5.767750672619752, 6.986819142042989, 10.284569826993076, 8.860951780068755, 8.365350434274385, 8.041691250271267, 8.613917726458926, 5.894467902906014, 6.89205427844115, 9.729893925214055, 6.969940994725083, 8.044087727864584]
EAM_BCN =  [5.010207706027561, 4.924277218905362, 4.83906855727687, 4.829226021814828, 4.42455642391937, 4.9250409290044, 5.01671704378995, 5.204090773457229, 5.296429720791903, 4.444125551165956, 4.654554001008622, 4.259783195726799, 4.64480059074633, 4.44641147960316, 4.5565043748027145, 4.562284411806049, 5.4036121175746725, 5.048848161793718, 4.450546457309915, 4.816734044238775, 4.954210185041331, 4.653926964962121, 4.788400129838423, 4.2357774984956995, 5.0061024752530185]
EAM_PMP =  [4.467888841725359, 4.968272970180319, 4.88237770157631, 4.201896860141947, 4.495966400763001, 5.247635619808929, 4.29237288658065, 4.470610917216599, 4.569895079641631, 4.771878097996567, 5.144298630531388, 4.580532112506905, 4.45492838849925, 5.2296063586919, 4.096019937534525, 5.156663682725695, 4.380979711359197, 4.717555845626677, 4.822689557316328, 5.2867477108733825, 4.671850609056877, 4.437665226483586, 4.879912829158282, 4.7918795190676295, 3.965314999975339]
EAM_ZGZ =  [5.027976604423138, 5.142551422119141, 5.379573051375572, 4.963734790532276, 5.018765613286182, 5.284233093261719, 3.999032685250947, 4.483491724187678, 4.728011044588956, 4.927118205060863, 4.932372391825974, 4.612603659581656, 4.449963502209596, 4.2162221272786455, 4.723384317725595, 4.5957555674543284, 4.136763139204546, 4.356883771491773, 5.118611499516651, 4.673585679796007, 4.378160110627762, 4.487251667061237, 3.9862922899650806, 5.14633147885101, 5.504179713701961]
EAM_HSC =  [5.621005164252387, 4.859366638491852, 4.061545747699159, 4.788359247072779, 4.21774041532266, 5.169966341269137, 5.72622834793245, 4.100826995541351, 5.566847406252466, 5.296684611927379, 5.080468726880623, 4.876441146388198, 4.515629200020221, 5.209773169623481, 4.539570240059284, 4.934886354388612, 4.634371092825225, 4.537991225117385, 5.203629888669409, 5.258072053543245, 5.130796182035196, 4.622637392294528, 4.968420510340219, 4.818362727309719, 4.664870175448331]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=6)

EAMRn =  [6.588793296813964, 6.711134948730469, 9.369001159667969, 7.164509887695313, 9.096622886657714, 5.953938064575195, 7.506752662658691, 13.072723007202148, 7.349069900512696, 11.146910858154296, 6.0503089904785154, 11.460886917114259, 7.153837814331054, 8.222893524169923, 7.536647644042969, 7.946441879272461, 8.463862533569335, 7.970678329467773, 11.042744941711426, 7.26727523803711, 7.874296951293945, 8.381033744812012, 9.244409942626953, 4.765035133361817, 10.166494255065919]
EAM_BCN =  [4.406255645751953, 4.592775802612305, 4.213763504028321, 4.175679740905761, 4.644869689941406, 4.752929306030273, 4.325121040344238, 4.6994179916381835, 4.829271850585937, 4.218277816772461, 4.635025291442871, 3.9574481582641603, 4.20955680847168, 5.430682220458984, 4.543532829284668, 5.089001350402832, 3.8549607467651366, 5.089059219360352, 4.61947811126709, 4.099877014160156, 4.6111263656616215, 4.252127838134766, 5.12316593170166, 5.331585540771484, 4.8301569366455075]
EAM_PMP =  [4.820737609863281, 5.265278358459472, 5.2166796875, 4.546687393188477, 4.675618896484375, 4.295512390136719, 4.199739265441894, 5.031737060546875, 4.468861923217774, 4.75194881439209, 4.847290306091309, 4.393739128112793, 4.549313926696778, 4.562192611694336, 4.440248908996582, 4.814693145751953, 4.029816055297852, 4.982411117553711, 4.6346698760986325, 4.971700363159179, 4.658419075012207, 5.020409126281738, 4.860447120666504, 4.452244606018066, 4.207275238037109]
EAM_ZGZ =  [3.5666857528686524, 4.1654746627807615, 4.12468994140625, 4.003257064819336, 4.413184852600097, 4.572491302490234, 3.8655735778808595, 5.1081107711791995, 4.251926879882813, 5.1637450790405275, 4.437108993530273, 3.774048271179199, 4.921551666259766, 4.424455184936523, 5.015518226623535, 4.731182861328125, 4.576274147033692, 4.3746683883666995, 4.429668769836426, 4.35126106262207, 4.820165634155273, 4.643289527893066, 4.760447273254394, 5.445890350341797, 4.392440223693848]
EAM_HSC =  [5.532123260498047, 4.599990959167481, 4.1690504837036135, 5.3029388809204105, 4.070294075012207, 4.858217697143555, 4.738374710083008, 4.94377182006836, 4.123273315429688, 5.334890975952148, 5.075549392700196, 4.816747016906739, 4.147074737548828, 4.978269996643067, 4.609133491516113, 5.28509635925293, 4.475118827819824, 4.965553283691406, 4.817886619567871, 4.804480018615723, 5.396377830505371, 5.141982536315918, 4.659411010742187, 4.824765281677246, 4.286901130676269]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=7)

EAMRn =  [8.120295156346689, 6.254716023360149, 5.681481691870359, 6.73635203295415, 8.150774209806235, 8.968285097934231, 9.62243376628007, 7.394667823715966, 7.257008958571028, 13.062663087750426, 7.758068160255356, 6.242641524513169, 6.522305630221226, 8.906514838190361, 6.165689411729869, 8.862719394192837, 5.891409883404722, 8.388197794999227, 6.064509363457708, 6.788895899706548, 6.733558428169477, 6.224548755305828, 10.32785430757126, 7.18425433470471, 8.88884712445854]
EAM_BCN =  [4.618853389626683, 4.705136440768101, 5.313835181812249, 4.140964545825921, 4.434409717522045, 4.8526778268341975, 4.461392053283087, 4.885453139201249, 5.181136707268139, 4.112813137545444, 4.299050812674041, 4.900530182489074, 5.033467509959004, 4.127447336026938, 4.644378813186495, 4.228362659416576, 4.261967347400023, 4.381474258876083, 5.097976760108872, 4.784936961561146, 4.9659914451070355, 4.857003504687016, 4.40221374813873, 4.58672283663608, 4.245728860987295]
EAM_PMP =  [5.109984973869701, 4.97010863653504, 4.994384576778601, 4.42380701197256, 4.664584811371152, 4.2194700335512065, 4.713944652292988, 5.232679952489267, 5.261548259470723, 4.7821250764450225, 3.9466334427937424, 4.27359949244131, 4.430951146796199, 4.538791996417659, 4.959982881451597, 4.474911567008141, 5.142954344796662, 5.082199398833926, 5.40005636687326, 4.221802437659537, 4.736294245955968, 5.341043113481881, 4.675775093607383, 5.087679494725595, 4.7247657398186105]
EAM_ZGZ =  [5.145658625234471, 5.050423310534788, 4.1748180955943495, 5.297482518866511, 3.8074473012792, 5.501901796548673, 4.944393044651145, 5.013736271622157, 4.866750887124845, 5.237704928558652, 5.070188578992787, 4.494435451998569, 4.988369441268468, 4.467132417282255, 4.291065253833733, 5.104214318908087, 5.653384369198639, 5.244954175288134, 4.467028513993367, 4.7154640348831025, 4.427225018491839, 4.452941554607731, 5.1708962846510484, 4.464476670369064, 5.0482602638773395]
EAM_HSC =  [4.298953556778407, 4.647862953714805, 4.757771180407835, 4.30063795335222, 4.430192286425298, 4.295024305286974, 5.043172062033474, 4.826527661616259, 5.147121807136158, 4.745521281025197, 4.476073010133044, 4.905301764459893, 4.702297663924718, 4.5720459588683475, 4.71685806123337, 4.635050178754447, 4.436577787493715, 4.905340591279587, 4.917958174601639, 4.74046000867787, 4.500432835947169, 4.3836124155781055, 3.968545366041731, 4.78838575004351, 5.902544380414604]
ErrorToCSV(EAMRn, EAM_BCN, EAM_PMP, EAM_ZGZ, EAM_HSC, nFw=8)