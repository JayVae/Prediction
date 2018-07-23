clear;
clc;
inputNum=11
load('week_0001.csv');%load data
x=week_0001'/max(week_0001)

Len=length(x);
[C,L]=wavedec(x,4,'db6');
cA4=appcoef(C,L,'db6',4);
cD4=detcoef(C,L,4);
cD3=detcoef(C,L,3);
cD2=detcoef(C,L,2);
cD1=detcoef(C,L,1);

A4=wrcoef('a',C,L,'db6',4);
D1=wrcoef('d',C,L,'db6',1);
D2=wrcoef('d',C,L,'db6',2);
D3=wrcoef('d',C,L,'db6',3);
D4=wrcoef('d',C,L,'db6',4);

figure(1)
subplot(6,1,1);plot(x);title('ori');
subplot(6,1,2);plot(A4);title('A4');
subplot(6,1,3);plot(D4);title('D4');
subplot(6,1,4);plot(D3);title('D3');
subplot(6,1,5);plot(D2);title('D2');
subplot(6,1,6);plot(D1);title('D1');

figure(2)
low_fre=A4;
mid_fre=D3+D4;
hi_fre=x-low_fre-mid_fre;
csvwrite('week_hi_fre2.csv',hi_fre')
subplot(3,1,1);plot(low_fre);title('Low');
subplot(3,1,2);plot(mid_fre);title('Mid');
subplot(3,1,3);plot(hi_fre);title('Hi');

figure(3)
l0=iddata(low_fre')
m0=iddata(mid_fre')
lm=ar(l0,4)%
ar_result=predict(lm,low_fre',4)
plot(ar_result)
% compare(l0,lm,1)
figure(4)
mm=armax(m0,[2,4])
armax_result=predict(mm,mid_fre',4)
plot(armax_result)
% compare(m0,mm,1)

figure(5)
% multilayer=[0.06551528722047806, -0.045997440814971924, 0.0012152784038335085, -0.020840011537075043, 0.0025771199725568295, 0.04145967587828636, 0.012329130433499813, 0.003486917121335864, 0.038878265768289566, -0.004283997695893049, 0.023711705580353737, -0.030726350843906403, 0.024558313190937042, 0.012036694213747978, -0.009047856554389, 0.026949848979711533, -0.02558155730366707, 0.02819523774087429, 0.008562950417399406, 0.022723278030753136, 0.012935452163219452, -0.009570841677486897, 0.022158151492476463, -0.012938430532813072, 0.01631299778819084, -0.003925243392586708, 0.03260892257094383, -0.0017990447813645005, -0.0021011200733482838, 0.004596737213432789, 0.02204817347228527, 0.012363815680146217, -0.0065556420013308525, -0.00948251597583294, 0.011590615846216679, 0.006483523175120354, 0.011384292505681515, 0.0006604789523407817, -0.000498950423207134, -0.0013130299048498273, -0.0031965263187885284, 0.010916333645582199, -0.007006529718637466, 0.004408390261232853, -0.010584434494376183, 0.04311777278780937, -0.00509209930896759, 0.028694674372673035, 0.003887931350618601, 0.011426963843405247, -0.012902555987238884, -0.10906831920146942, -0.012377464212477207, 0.006311987992376089, 0.06630601733922958, 0.03469843417406082, 0.057114116847515106, -0.027925439178943634, -0.033551670610904694, -0.019697081297636032, -0.007014336995780468, 0.02325711026787758, 0.02411886863410473, 0.012696059420704842, -0.050624921917915344, 0.023707831278443336, -0.013323649764060974, 0.041161566972732544, -0.008899274282157421, 0.006846975535154343, -0.013873578049242496, -0.028971724212169647, -0.0035490836016833782, 0.025172159075737, 0.027692800387740135, 0.04827278479933739, 0.0073389881290495396, 0.006519760936498642, -0.15362457931041718, -0.04803111404180527, -0.009254608303308487, 0.021585090085864067, 0.07877805083990097, 0.049623046070337296, 0.09747987240552902, -0.36660274863243103, -0.05673585087060928, 0.17202456295490265, 0.0549703985452652, -0.1427600234746933, 0.10993272811174393, 0.10327595472335815, 0.06898686289787292, -0.01782985031604767, -0.007190043572336435, -0.02755557931959629, 0.028229719027876854, 0.012134015560150146, 0.04823758080601692, -0.18187233805656433, -0.0746605172753334, 0.1439628154039383, 0.023412536829710007, 0.06714186817407608, -0.026055095717310905, 0.024877775460481644, 0.06243186071515083]
% multilayer=[0.03728591278195381, -0.04527963325381279, -0.03641507774591446, 0.0014458884252235293, 0.01518990844488144, -0.009221768006682396, 0.012038243934512138, -0.007231824100017548, 0.019433673471212387, -0.003604992525652051, 0.025857876986265182, -0.07693648338317871, 0.027676070109009743, 0.008815416134893894, 0.011051679961383343, 0.017746251076459885, -0.01813892088830471, 0.03001205064356327, -0.0056890822015702724, 0.019211191684007645, 0.003214228665456176, -4.88758014398627e-05, 0.01502829696983099, 0.0037512597627937794, 0.01770644821226597, -0.0004957913770340383, 0.017370859161019325, -0.003658158704638481, 0.019651861861348152, 0.004382343031466007, 0.02629031427204609, -0.005917956121265888, 0.0063559142872691154, -0.0235182736068964, 0.015303966589272022, 0.007383092772215605, 0.011124389246106148, 0.00777526805177331, 0.00939377211034298, 0.013049216940999031, 0.008421340957283974, 0.013349693268537521, 0.0038687982596457005, 0.018236638978123665, 0.0036785434931516647, 0.02730144001543522, -0.014059809036552906, 0.03861568123102188, 0.005914618726819754, 0.019322849810123444, -0.01100209541618824, -0.07922781258821487, -0.02705974131822586, 0.009698324836790562, 0.02465999498963356, -0.00033926955075003207, 0.029116623103618622, -0.0012167685199528933, -0.028125731274485588, 0.009190300479531288, 0.011978409253060818, -0.0027384685818105936, 0.01320905052125454, 0.012115123681724072, -0.054771266877651215, 0.021728668361902237, -0.0018262842204421759, 0.04083364084362984, -0.0046906122006475925, 0.0010561939561739564, 0.0016244633588939905, 0.004932005889713764, 0.008888009004294872, 0.019200704991817474, 0.0033079262357205153, 0.024540141224861145, 0.007587405852973461, 0.017671531066298485, -0.08897672593593597, -0.005153490696102381, 0.0042918650433421135, 0.025213973596692085, 0.006816758308559656, 0.008765473030507565, 0.04549911618232727, -0.3798604905605316, -0.1168544590473175, 0.1631544828414917, 0.11659988760948181, -0.06753626465797424, -0.31228548288345337, 0.10186325758695602, 0.019409481436014175, -0.02419273555278778, -0.012653269805014133, 0.010387403890490532, 0.0019103263039141893, -0.0021700821816921234, 0.016024766489863396, -0.0836135745048523, -0.22244486212730408, 0.04571622237563133, -0.00036287299008108675, 0.022972112521529198, -0.03018808551132679, -0.041278377175331116, -0.0006254910258576274]
% multilayer=[0.10642171651124954, -0.053772203624248505, -0.04814410209655762, 0.005318888928741217, -0.021347971633076668, -0.011948253028094769, 0.00029319521854631603, 0.010234533809125423, 0.027041932567954063, -0.014257477596402168, 0.02774372510612011, -0.0840819701552391, 0.05971844866871834, 0.03435719013214111, -0.02515321597456932, 0.04319171607494354, -0.04467194154858589, 0.047304995357990265, 0.01851281337440014, 0.012251656502485275, 0.006961351726204157, -0.0041969167068600655, 0.016765691339969635, -0.018052104860544205, 0.008538395166397095, -0.007732892408967018, 0.04316423460841179, -0.012289857491850853, 0.006990139838308096, -0.0010929697891697288, 0.04137185588479042, -0.004123365972191095, -0.010045904666185379, -0.042029280215501785, 0.017676062881946564, 0.008787405677139759, 0.0027284552343189716, 0.008276928216218948, 0.008001096546649933, 0.011268614791333675, -0.0030375623609870672, 0.016460003331303596, -0.010137686505913734, 0.017577268183231354, -0.009588364511728287, 0.05279853194952011, -0.02913817949593067, 0.06145744025707245, 0.01617761142551899, 0.028890017420053482, -0.034380704164505005, -0.06309816986322403, -0.04483968764543533, 0.0018109061056748033, 0.07840511202812195, 0.008588341064751148, 0.09658808261156082, -0.012632470577955246, -0.09195614606142044, -0.03790665045380592, -0.02028871886432171, -0.012334256432950497, 0.022421235218644142, 0.04094574972987175, -0.05451376736164093, 0.033719442784786224, -0.024932103231549263, 0.06845467537641525, -0.0077370041981339455, 0.009403605945408344, 0.0013030163245275617, -0.03910072147846222, -0.005397684406489134, 0.03222213685512543, 0.015693625435233116, 0.06665102392435074, 0.023735174909234047, 0.03325885906815529, -0.09255025535821915, -0.10998063534498215, -0.031161168590188026, 0.013911004178225994, 0.07471273839473724, 0.02318347990512848, 0.2145640105009079, -0.37197598814964294, -0.096021369099617, 0.17372341454029083, 0.059639282524585724, -0.1548919975757599, -0.08148769289255142, 0.2034432739019394, 0.08275379985570908, -0.06848149001598358, 0.007272889371961355, -0.003455684520304203, -0.04503122717142105, 0.011060202494263649, 0.004200910218060017, -0.12031438946723938, -0.018584851175546646, 0.08850329369306564, -0.007282187696546316, 0.05182517692446709, -0.046399567276239395, 0.029666652902960777, -0.05005552992224693]
% multilayer=[0.09500548988580704, -0.0514349490404129, 0.036832913756370544, 0.02211814746260643, -0.029369277879595757, 0.019319482147693634, -0.01787218637764454, 0.0035743259359151125, 0.028906183317303658, 0.012473204173147678, -0.0003809629997704178, -0.12158821523189545, 0.047821223735809326, 0.012197216041386127, -0.01681547611951828, 0.03167995810508728, -0.03985024616122246, 0.04816196858882904, 0.018779246136546135, -0.008252343162894249, 0.017641829326748848, -0.010417411103844643, 0.013951795175671577, -0.001600115210749209, 0.0027861222624778748, -0.026668986305594444, 0.02495822310447693, 0.01989695057272911, -0.012819868512451649, -0.013309613801538944, 0.03258049115538597, 0.009215062484145164, -0.025222165510058403, -0.13394834101200104, 0.00900944508612156, 0.010097546502947807, -0.003299015574157238, -0.006183455232530832, 0.0025472284760326147, 0.011596308089792728, -0.008087130263447762, -0.00034162396332249045, -0.0035162719432264566, 0.00535515695810318, -0.014132421463727951, 0.0299746822565794, 0.004433453548699617, 0.017922835424542427, -0.006580227520316839, 0.021426942199468613, 0.00038817519089207053, -0.0973910540342331, -0.055747490376234055, 0.0030556523706763983, 0.05319596454501152, 0.0071577089838683605, 0.0789107158780098, 0.00140669837128371, -0.021754972636699677, -0.0014255930436775088, -0.015728337690234184, -0.007369175087660551, -0.006267614662647247, 0.029693420976400375, -0.1380721628665924, 0.020280588418245316, -0.01443962100893259, 0.04465162381529808, -0.01941821165382862, -0.0037564155645668507, 0.02397235669195652, -0.031594038009643555, -0.006951786112040281, 0.01511223241686821, 0.018347736448049545, 0.033275023102760315, 0.020530227571725845, 0.026879774406552315, -0.1370743066072464, -0.02946307510137558, -0.009288549423217773, 0.0023527396842837334, 0.05824209749698639, 0.01494597364217043, 0.20104938745498657, -0.17993813753128052, 0.042176805436611176, 0.1578388661146164, 0.051412954926490784, -0.07909119874238968, -0.052307408303022385, 0.2827873229980469, 0.10526476055383682, -0.0552370622754097, -0.01348720584064722, 0.0195864699780941, -0.013553710654377937, 0.02392696402966976, 0.008453733287751675, -0.18307426571846008, -0.11343155056238174, 0.04110649973154068, 0.02840574085712433, 0.06518521904945374, -0.05229457467794418, 0.0816740021109581, -0.09522467106580734]
load('result.csv')
multilayer=result
ar_result2=ar_result((inputNum+1):112,:)
armax_result2=armax_result((inputNum+1):112,:)
sum_all_wave=ar_result2'+armax_result2'+multilayer
plot(sum_all_wave,'r')
hold on
plot(x(:,(inputNum+1):112),'g')

for i=(inputNum+1):112
%     pec(i-6)=abs(x(i)-sum_all_wave(i-6))
    pec(i-inputNum)=abs(x(i)-sum_all_wave(i-inputNum))/x(i);
end
figure(6)
plot(pec)
mean(pec)
mean(pec(81:101))