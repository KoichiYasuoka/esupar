#! /usr/bin/python3 -i
# coding=utf-8
hangul={
  "가":("家佳街可歌加價假架暇","嘉嫁稼賈駕伽迦柯呵哥枷珂痂苛茄袈訶跏軻哿嘏舸珈坷斝榎檟笳耞葭謌泇",""),
  "각":("各角脚閣却覺刻","珏恪殼愨卻咯埆搉擱桷","愨(慤)"),
  "간":("干間看刊肝幹簡姦懇","艮侃杆玕竿揀諫墾栞奸柬澗磵稈艱癇忓矸偘慳榦秆茛衎赶迀齦","杆(桿)癇(癎)"),
  "갈":("渴","葛乫喝曷碣竭褐蝎鞨噶楬秸羯蠍",""),
  "감":("甘減感敢監鑑","勘堪瞰坎嵌憾戡柑橄疳紺邯龕玪坩埳嵁弇憨撼欿歛泔淦澉矙轗酣鹻","鑑(鑒)"),
  "갑":("甲","鉀匣岬胛閘",""),
  "강":("江降講強康剛鋼綱","杠堈岡姜橿彊慷畺疆糠絳羌腔舡薑鱇嫝跭襁玒顜茳鏹傋僵壃忼悾扛殭矼穅繈罡羫豇韁鏹","強(强)鋼(鎠)岡(崗)襁(襁)"),
  "개":("改皆個開介慨槪蓋","价凱愷漑塏愾疥芥豈鎧玠剴匃揩槩磕闓","個(箇)蓋(盖)"),
  "객":("客","喀",""),
  "갱":("更","坑粳羹硜賡鏗",""),
  "갹":("","醵",""),
  "거":("去巨居車擧距拒據","渠遽鉅炬倨据祛踞鋸駏呿昛秬筥籧胠腒苣莒蕖蘧袪裾",""),
  "건":("建乾件健","巾虔楗鍵愆腱蹇騫搴湕踺揵犍睷褰謇鞬","建(䢖)乾(漧乹)"),
  "걸":("傑乞","桀乬朅榤","傑(杰)"),
  "검":("儉劍檢","瞼鈐黔撿芡","劍(劒)"),
  "겁":("","劫怯迲刦刧",""),
  "게":("","揭偈憩",""),
  "격":("格擊激隔","檄膈覡挌毄闃骼鬲鴃",""),
  "견":("犬見堅肩絹遣牽","鵑甄繭譴狷畎筧縳繾羂蠲鰹",""),
  "결":("決結潔缺","訣抉㛃焆迼玦鍥觖闋","潔(㓗洯)鍥(䤿)"),
  "겸":("兼謙","鎌慊箝鉗嗛槏傔岒拑歉縑蒹黚鼸嵰",""),
  "경":("京景經庚耕敬輕驚慶競竟境鏡頃傾硬警徑卿","倞鯨坰耿炅更梗憬璟瓊擎儆俓涇莖勁逕熲冏勍烱璥痙磬絅脛頸鶊檠冂𠗊憼巠曔燛剄哽惸扃焭煢畊竸綆罄褧謦顈駉鯁黥","卿(卿)冏(囧)景(暻)檠(㯳)京(亰)璟(璄)"),
  "계":("癸季界計溪鷄系係戒械繼契桂啓階繫","誡烓屆悸棨稽谿堦瘈禊綮縘罽薊雞髻","界(堺)谿(磎)"),
  "고":("古故固苦高考告枯姑庫孤鼓稿顧","叩敲皐暠呱尻拷槁沽痼睾羔股膏苽菰藁蠱袴誥賈辜錮雇杲鼔估凅刳栲槀櫜牯盬瞽鷱稁箍篙糕罟羖翺胯觚詁郜酤鈷靠鴣","考(攷)皐(皋)"),
  "곡":("谷曲穀哭","斛梏鵠嚳槲縠觳轂",""),
  "곤":("困坤","昆崑琨錕梱棍滾鯤衮堃崐悃捆緄裍褌閫髡鵾鶤齫","衮(袞)"),
  "골":("骨","汨滑搰榾鶻",""),
  "공":("工功空共公孔供恭攻恐貢","珙控拱蚣鞏龔倥崆栱箜蛩蛬贛跫釭槓",""),
  "곶":("","串",""),
  "과":("果課科過誇寡","菓跨鍋顆戈瓜侉堝夥夸撾猓稞窠蝌裹踝銙騍",""),
  "곽":("郭","廓槨藿椁癨霍鞹",""),
  "관":("官觀關館管貫慣冠寬","款琯錧灌瓘梡串棺罐菅涫輨丱爟盥祼窾筦綰鑵雚顴髖鸛","館(舘)寬(寛)"),
  "괄":("","括刮恝适佸栝筈聒髺鴰",""),
  "광":("光廣鑛狂","侊洸珖桄匡曠壙筐胱恇框爌獷磺絖纊茪誆誑硄","廣(広)光(炛炚)"),
  "괘":("掛","卦罫咼挂罣詿",""),
  "괴":("塊愧怪壞","乖傀拐槐魁媿廥瑰璝蒯襘",""),
  "괵":("","馘",""),
  "굉":("","宏紘肱轟浤觥訇閎",""),
  "교":("交校橋敎郊較巧矯","僑喬嬌膠咬嶠攪狡皎絞翹蕎蛟轎餃驕鮫姣佼噭憍鄗嘐嘄嚙撟晈暞榷磽窖趫蹻鉸骹鵁齩","敎(教)"),
  "구":("九口求救究久句舊具俱區驅苟拘狗丘懼龜構球","玖矩邱銶溝購鳩軀枸仇勾咎嘔垢寇嶇柩歐毆毬灸瞿絿臼舅衢謳逑鉤駒鷗玽耇廏龜颶佝俅傴冓劬匶厹叴坸姤媾嫗屨岣彀戵扣捄搆摳昫榘漚璆甌疚痀癯窶篝糗胊蒟蚯裘覯詬遘釦韝韭鬮鷇鸜","丘(坵)耇(耈)廏(廐)"),
  "국":("國菊局","鞠鞫麴𥮗匊掬跼麯趜","國(国)"),
  "군":("君郡軍群","窘裙捃桾皸",""),
  "굴":("屈","窟堀掘倔崛淈詘",""),
  "궁":("弓宮窮","躬穹芎躳",""),
  "권":("券權勸卷拳","圈眷倦捲淃勌惓棬睠綣蜷","權(権)"),
  "궐":("厥","闕獗蕨蹶",""),
  "궤":("軌","机櫃潰詭饋佹几劂匱憒撅樻氿簋繢跪闠餽麂",""),
  "귀":("貴歸鬼","句晷䤥龜","龜(龜)"),
  "규":("叫規糾","圭奎珪揆逵窺葵槻硅竅赳閨邽嫢湀茥煃刲嬀巋暌楏樛潙睽虯跬闚頍馗騤","糾(糺)"),
  "균":("均菌","畇鈞筠勻龜覠囷麏","勻(匀)龜(龜)"),
  "귤":("","橘",""),
  "극":("極克劇","剋隙戟棘亟尅屐郄",""),
  "근":("近勤根斤僅謹","墐漌槿瑾嫤筋劤懃芹菫覲饉巹廑觔跟釿靳堇",""),
  "글":("","契㔕",""),
  "금":("金今禁錦禽琴","衾襟昑妗擒檎芩衿唫噤嶔笒黅",""),
  "급":("及給急級","汲伋扱圾岌皀礏笈芨",""),
  "긍":("肯","亘兢矜殑","亘(亙)"),
  "기":("己記起其期基氣技幾旣紀忌旗欺奇騎寄豈棄祈企畿飢器機","淇琪璂棋祺錤騏麒玘杞埼崎琦綺錡箕岐汽沂圻耆璣磯譏冀驥嗜暣伎夔妓朞畸祁祇羈耭肌饑稘榿㠌忯僛剞墍屺庋弃忮愭掎攲旂曁棊歧炁猉禨綦綥羇肵芰芪蘄𧅄蜝蟣覬跂隑頎鬐鰭","棋(碁)璣(㼄)"),
  "긴":("緊","",""),
  "길":("吉","佶桔姞拮蛣",""),
  "김":("","金",""),
  "끽":("","喫",""),
  "나":("那","奈柰娜拏儺喇懦拿𣃽䏧挐挪𡖔梛糯䛔橠",""),
  "낙":("諾","",""),
  "난":("暖難","煖偄愞赧餪",""),
  "날":("","捺捏",""),
  "남":("南男","楠湳枏喃",""),
  "납":("納","衲",""),
  "낭":("娘","囊曩",""),
  "내":("內乃奈耐","柰奶嬭迺鼐",""),
  "녀":("女","",""),
  "녁":("","惄",""),
  "년":("年","撚碾","年(秊)"),
  "념":("念","恬拈捻",""),
  "녑":("","惗",""),
  "녕":("寧","獰佞儜嚀濘","寧(寗)"),
  "노":("怒奴努","弩瑙駑䛝呶孥峱猱笯臑",""),
  "농":("農","膿濃儂噥穠醲",""),
  "뇌":("腦惱","餒",""),
  "뇨":("","尿鬧撓嫋嬲淖鐃",""),
  "누":("","耨啂檽",""),
  "눈":("","嫩",""),
  "눌":("","訥吶肭",""),
  "뉴":("","紐鈕杻𧘥忸",""),
  "뉵":("","衄",""),
  "능":("能","",""),
  "니":("泥","尼柅濔膩馜㦐呢怩祢禰妮",""),
  "닉":("","匿溺",""),
  "닐":("","昵暱",""),
  "다":("多茶","爹𥥸𣘻茤觰鄲奲","多(夛)"),
  "단":("丹但單短端旦段壇檀斷團","緞鍛亶彖湍簞蛋袒鄲煓㫜担慱椴漙癉耑胆腶蜑",""),
  "달":("達","撻澾獺疸妲怛闥靼韃",""),
  "담":("談淡擔","譚膽澹覃啖坍憺曇湛痰聃蕁錟潭倓啿埮炎儋啗噉墰壜毯禫罎薝郯黮黵惔緂",""),
  "답":("答畓踏","沓遝",""),
  "당":("堂當唐糖黨","塘鐺撞幢戇棠螳倘儻搪檔溏瑭璫瞠礑蟷襠讜鏜餳餹",""),
  "대":("大代待對帶臺貸隊","垈玳袋戴擡旲岱黛㬃㬣儓懟汏碓鐓","臺(坮)擡(抬)"),
  "댁":("","宅",""),
  "덕":("德","","德(悳徳)"),
  "도":("刀到度道島徒圖倒都桃挑跳逃渡陶途稻導盜塗","堵棹濤燾禱鍍蹈屠悼掉搗櫂淘滔睹萄覩賭韜馟祹鋾夲稌叨壔弢忉慆掏搯擣檮洮涂鼗菟酴闍鞀鞱饕","島(嶋)道(噵)"),
  "독":("讀獨毒督篤","瀆牘犢禿纛櫝黷",""),
  "돈":("豚敦","墩惇暾燉頓旽沌焞弴潡躉",""),
  "돌":("突","乭咄堗",""),
  "동":("同洞童冬東動銅凍","棟董潼垌瞳蝀憧疼胴桐朣曈彤烔橦勭侗僮哃峒涷艟苳茼蕫","同(仝)"),
  "두":("斗豆頭","杜枓兜痘竇荳讀逗阧抖斁肚脰蚪蠹陡",""),
  "둔":("鈍屯","遁臀芚遯窀迍",""),
  "둘":("","乧",""),
  "득":("得","",""),
  "등":("等登燈騰","藤謄鄧嶝橙凳墱滕磴籐縢螣鐙",""),
  "라":("羅","螺喇懶癩蘿裸邏剆覶摞蓏鑼儸砢臝倮囉曪瘰騾驘纙",""),
  "락":("落樂絡","珞酪烙駱洛嗠犖",""),
  "란":("卵亂蘭欄","瀾瓓丹欒鸞爛鑾嬾幱攔灓襴闌斕欗",""),
  "랄":("","剌辣埒辢",""),
  "람":("覽濫","嵐攬欖籃纜襤藍㛦灆婪漤爁璼惏","攬(擥𢱯)"),
  "랍":("","拉臘蠟鑞",""),
  "랑":("浪郞廊","琅瑯狼朗烺蜋㢃駺榔閬硠稂莨㫰","蜋(螂)郞(郎)"),
  "래":("來","崍萊徠淶騋唻","來(来𧼛)"),
  "랭":("冷","",""),
  "략":("略掠","畧",""),
  "량":("良兩量涼梁糧諒","亮倆樑粱輛駺俍喨悢踉魎","糧(粮)涼(凉)"),
  "려":("旅麗慮勵","呂侶閭黎儷廬戾櫚濾礪藜蠣驢驪曞儢厲唳梠癘糲膂臚蠡邌鑢",""),
  "력":("力歷曆","瀝礫轢靂攊櫟櫪癧轣酈",""),
  "련":("連練鍊憐聯戀蓮","煉璉攣漣輦孌𨏶楝湅臠鏈鰊鰱奱",""),
  "렬":("列烈裂劣","洌冽挒捩颲",""),
  "렴":("廉","濂簾斂殮瀲磏",""),
  "렵":("獵","躐鬣",""),
  "령":("令領嶺零靈","伶玲姈昤鈴齡怜囹笭羚翎聆逞泠澪岭呤另欞鹷秢苓蛉軨鴒朎","岭(岺)"),
  "례":("例禮隷","澧醴隸鱧","禮(礼)"),
  "로":("路露老勞爐","魯盧鷺撈擄櫓潞瀘蘆輅鹵嚧虜璐櫨蕗潦瓐澇壚滷玈癆窂鸕艪艫轤鐪鑪顱髗鱸㯝鏴","虜(虜)"),
  "록":("綠祿錄鹿","彔碌菉麓淥漉簏轆鵦",""),
  "론":("論","",""),
  "롱":("弄","瀧瓏籠壟朧聾儱攏曨礱蘢隴",""),
  "뢰":("雷賴","瀨儡牢磊賂賚耒攂礌礧籟纇罍蕾誄酹","賴(頼)"),
  "료":("料了僚","遼寮廖燎療瞭聊蓼嘹嫽撩暸潦獠繚膋醪鐐飂飉",""),
  "룡":("龍","龒","龍(竜)"),
  "루":("屢樓累淚漏","壘婁瘻縷蔞褸鏤陋慺嶁耬熡僂嘍螻髏漊謱漯",""),
  "류":("柳留流類","琉劉硫瘤旒榴溜瀏謬橊縲纍遛鶹","琉(瑠)"),
  "륙":("六陸","戮勠",""),
  "륜":("倫輪","侖崙綸淪錀圇掄","崙(崘)"),
  "률":("律栗率","慄嵂𥠲瑮溧",""),
  "륭":("隆","癃窿㦕",""),
  "륵":("","勒肋泐",""),
  "름":("","廩凜菻澟","凜(凛)"),
  "릉":("陵","綾菱稜凌楞倰蔆","楞(楞)"),
  "리":("里理利梨李吏離裏履","俚莉璃俐唎浬狸痢籬罹羸釐鯉涖𢻠犂摛剺哩嫠莅蜊螭貍邐魑黐漓灕","裏(裡)離(离)俐(悧)釐(厘)犂(犁)"),
  "린":("鄰","潾璘麟吝燐藺躪鱗撛鏻獜橉粦粼䗲繗嶙悋磷驎躙轔斴焛暽瞵","麟(𪊭)鄰(隣)"),
  "림":("林臨","琳霖淋棽碄晽玪痳",""),
  "립":("立","笠粒砬岦",""),
  "마":("馬麻磨","瑪摩痲碼魔媽劘螞蟇麽",""),
  "막":("莫幕漠","寞膜邈瞙鏌",""),
  "만":("萬晚滿慢漫","曼蔓鏋卍娩巒彎挽灣瞞輓饅鰻蠻墁嫚幔縵謾蹣鏝鬘","萬(万)"),
  "말":("末","茉唜抹沫襪靺帕秣",""),
  "망":("亡忙忘望茫妄罔","網芒輞邙莽惘汒漭魍","莽(莽)望(朢)"),
  "매":("每買賣妹梅埋媒","寐昧枚煤罵邁魅苺呆楳沬玫眛莓酶霉",""),
  "맥":("麥脈","貊陌驀貃貘",""),
  "맹":("孟猛盟盲","萌氓甍甿虻",""),
  "멱":("","冪覓幎",""),
  "면":("免勉面眠綿","冕棉沔眄緬麪俛湎緜","麪(麵)"),
  "멸":("滅","蔑篾衊",""),
  "명":("名命明鳴銘冥","溟暝椧皿瞑茗蓂螟酩慏洺眀䳟",""),
  "몌":("","袂",""),
  "모":("母毛暮某謀模貌募慕冒侮","摸牟謨姆帽摹牡瑁眸耗芼茅矛橅軞慔侔姥媢嫫恈旄皃眊耄蝥蟊髦",""),
  "목":("木目牧睦","穆鶩沐苜",""),
  "몰":("沒","歿",""),
  "몽":("夢蒙","朦幪懞曚溕濛瞢矇艨雺鸏",""),
  "묘":("卯妙苗廟墓","描錨畝昴杳渺猫淼眇藐貓","妙(竗)"),
  "무":("戊茂武務無舞貿霧","拇珷畝撫懋巫憮楙毋繆蕪誣鵡橅儛嘸廡膴騖堥","無(无)"),
  "묵":("墨默","嘿",""),
  "문":("門問聞文","汶炆紋們刎吻紊蚊雯抆悗懣捫璊玧",""),
  "물":("勿物","沕",""),
  "미":("米未味美尾迷微眉","渼薇彌嵄媄媚嵋梶楣湄謎靡黴躾媺濔煝娓洣侎瑂𥹄㵟冞蘪媺亹弭敉麋瀰獼糜縻苿蘼","彌(弥)"),
  "민":("民敏憫","玟旻旼閔珉岷忞慜敃愍潣暋䪸泯悶緡𩔉鈱脗閩盿罠琝琘緍苠鰵黽眠鍲","珉(瑉砇䃉)忞(忟)"),
  "밀":("密蜜","謐樒滵",""),
  "박":("泊拍迫朴博薄","珀撲璞鉑舶剝樸箔粕縛膊雹駁亳欂牔鎛駮髆",""),
  "반":("反飯半般盤班返叛伴","畔頒潘磐拌搬攀斑槃泮瘢盼磻礬絆蟠豳攽媻扳搫朌胖頖螌",""),
  "발":("發拔髮","潑鉢渤勃撥跋醱魃炦哱浡脖鈸鵓",""),
  "방":("方房防放訪芳傍妨倣邦","坊彷昉龐榜尨旁枋滂磅紡肪膀舫蒡蚌謗幫仿厖徬搒旊梆牓舽螃鎊髣魴","幫(幇)"),
  "배":("拜杯倍培配排輩背","陪裴湃俳徘焙胚褙賠北䔒貝坏扒琲蓓俖","杯(盃)裴(裵)"),
  "백":("白百伯","佰帛魄柏苩䞟珀","柏(栢)"),
  "번":("番煩繁飜","蕃幡樊燔磻藩繙膰蘩袢","飜(翻)"),
  "벌":("伐罰","閥筏橃罸",""),
  "범":("凡犯範","帆杋氾范梵泛汎釩渢滼笵訉颿",""),
  "법":("法","琺",""),
  "벽":("壁碧","璧闢僻劈擘檗癖霹辟擗甓疈襞鷿鼊","檗(蘗)"),
  "변":("變辯辨邊","卞弁便釆忭抃籩腁賆辮駢骿鴘",""),
  "별":("別","瞥鼈襒𩠻莂鷩䭱㔡炦彆","鼈(鱉)"),
  "병":("丙病兵竝屛","幷倂甁輧炳柄昞秉餠騈鉼抦絣缾迸鈵","竝(並)幷(并)昞(昺)柄(棅)鉼(鉼)"),
  "보":("保步報普補譜寶","堡甫輔菩潽洑湺褓俌𤣰䀯盙簠葆靌鴇黼溥鋪","寶(宝珤㻄)步(歩)"),
  "복":("福伏服復腹複卜覆","馥鍑僕匐宓茯蔔輹輻鰒墣幞扑濮箙菔蝠蝮鵩",""),
  "본":("本","",""),
  "볼":("","乶",""),
  "봉":("奉逢峯蜂封鳳","俸捧琫烽棒蓬鋒熢縫漨芃丰夆篷綘菶鴌","峯(峰)漨(浲)"),
  "부":("夫扶父富部婦否浮付符附府腐負副簿赴賦","孚芙傅溥敷復不俯剖咐埠孵斧缶腑艀莩訃賻趺釜阜駙鳧膚俘媍抔拊掊桴榑涪玞祔筟罘罦胕芣苻蔀蚨蜉袝裒跗鈇頫鮒麩荴",""),
  "북":("北","",""),
  "분":("分紛粉奔墳憤奮","汾芬盆吩噴忿扮昐焚糞賁雰体坌帉枌棼棻氛湓濆犇畚砏笨肦膹蕡轒黺鼢",""),
  "불":("不佛拂","彿弗岪祓紱艴茀韍髴黻",""),
  "붕":("朋崩","鵬棚硼繃堋鬅漰",""),
  "비":("比非悲飛鼻備批卑婢碑妃肥祕費","庇枇琵扉譬丕匕匪憊斐榧毖毗沸泌痺砒秕粃緋翡脾臂菲蜚裨誹鄙棐庀奜霏俾馡伾仳剕圮埤妣屁庳悱椑沘淝淠濞狒狉痞痹睥篦紕羆腓芘芾萆蓖蚍貔贔轡邳郫閟陴鞴騑騛髀鼙","祕(秘)毗(毘)"),
  "빈":("貧賓頻","彬斌濱嬪穦儐璸玭嚬檳殯浜瀕牝邠繽豳霦贇鑌擯馪矉臏蘋顰鬢蠙","彬(份)"),
  "빙":("氷聘","憑騁凭娉",""),
  "사":("四巳士仕寺史使舍射謝師死私絲思事司詞蛇捨邪賜斜詐社沙似査寫辭斯祀","泗砂糸紗娑徙奢嗣赦乍些伺俟僿唆柶梭渣瀉獅祠肆莎蓑裟飼駟麝篩傞剚卸咋姒楂榭汜痧皶竢笥蜡覗駛魦鯊鰤涘禠",""),
  "삭":("削朔","數索爍鑠搠槊蒴",""),
  "산":("山産散算","珊傘刪汕疝蒜霰酸產祘㦃剷姍孿橵澘潸狻繖訕鏟簅",""),
  "살":("殺","薩乷撒煞",""),
  "삼":("三","參蔘杉衫滲芟森糝釤鬖",""),
  "삽":("","插澁鈒颯卅唼歃翣鍤霅霎","插(揷)"),
  "상":("上尙常賞商相霜想傷喪嘗裳詳祥象像床桑狀償","庠湘箱翔爽塽孀峠廂橡觴樣牀慡潒徜晌殤甞緗鎟顙鬺",""),
  "새":("塞","璽賽鰓愢㘔",""),
  "색":("色索","嗇穡塞槭濇瀒",""),
  "생":("生","牲甥省笙眚鉎",""),
  "서":("西序書署敍徐庶恕暑緖誓逝","抒舒瑞棲曙壻㥠諝墅嶼犀筮絮胥薯鋤黍鼠藇揟悆湑偦稰㷂遾噬撕澨紓耡芧鉏豫嫬","敍(叙敘)棲(栖捿)壻(婿)嶼(㠘)恕(㣽)胥(縃)諝(𧩑)"),
  "석":("石夕昔惜席析釋","碩奭汐淅晳䄷鉐錫潟蓆舃鼫褯矽腊蜥磶","晳(晰)"),
  "선":("先仙線鮮善船選宣旋禪","扇渲瑄愃墡膳繕琁璿璇羨嬋銑珗嫙僊敾煽癬腺蘚蟬詵跣鐥洒亘譔䁢㻽洗尟屳歚筅綫譱鏇鱻騸鱓秈烍暶","膳(饍)"),
  "설":("雪說設舌","薛楔屑泄洩渫褻齧禼蔎契偰㨹媟揲暬爇碟稧紲枻","禼(卨)"),
  "섬":("","纖暹蟾剡殲贍閃陝孅憸摻睒譫銛韱",""),
  "섭":("涉攝","燮葉㰔𦁗躞躡囁懾灄聶鑷顳",""),
  "성":("姓性成城誠盛省聖聲星","珹娍瑆惺醒宬猩筬腥貹胜成城誠盛晟𦖤騂","晟(晟晠)聖(聖)"),
  "세":("世洗稅細勢歲","貰笹說忕洒涗𡜧銴彗帨繐蛻",""),
  "소":("小少所消素笑召昭蘇騷燒訴掃疏蔬","沼炤紹邵韶巢遡柖玿嘯塑宵搔梳瀟瘙篠簫蕭逍銷愫穌卲霄劭䘘璅傃䴛佋嗉埽塐愬捎樔泝筱箾繅翛膆艘蛸酥魈鮹釗焇","疏(疎)穌(甦)霄(䨭)遡(溯)笑(咲)"),
  "속":("俗速續束粟屬","涑謖贖洬遬",""),
  "손":("孫損","遜巽蓀飧","飧(飡)"),
  "솔":("","率帥乺䢦𧗿窣蟀",""),
  "송":("松送頌訟誦","宋淞悚竦憽鬆",""),
  "쇄":("刷鎖","殺灑碎曬瑣","鎖(鎻)"),
  "쇠":("衰","釗",""),
  "수":("水手受授首守收誰須雖愁樹壽數修秀囚需帥殊隨輸獸睡遂垂搜","洙琇銖粹穗繡隋髓袖嗽嫂岫戍漱燧狩璲瘦綏綬羞茱蒐蓚藪邃酬銹隧鬚䳠賥豎讎睢睟瓍宿汓㻽叟售廋晬殳泅溲瞍祟籔脺膄膸陲颼饈","壽(寿)修(脩)穗(穂)岫(峀)豎(竪)讎(讐)睢(濉)"),
  "숙":("叔淑宿孰熟肅","塾琡璹橚夙潚菽倏俶儵婌驌鷫",""),
  "순":("順純旬殉循脣瞬巡","洵珣荀筍舜淳焞諄錞醇徇恂栒楯橓蓴蕣詢馴盾峋姰畃侚盹眴紃肫駨鬊鶉",""),
  "술":("戌述術","鉥𡊍絉",""),
  "숭":("崇","嵩崧菘",""),
  "쉬":("","倅淬焠",""),
  "슬":("","瑟膝璱蝨㻭𩇣虱",""),
  "습":("習拾濕襲","褶慴槢隰",""),
  "승":("乘承勝昇僧","丞陞繩蠅升榺氶塍㞼陹鬙㴍","陞(阩)"),
  "시":("市示是時詩施試始矢侍視","柴恃匙嘶媤尸屎屍弒猜翅蒔蓍諡豕豺偲毸諟媞柹愢禔絁沶諰眂漦兕厮啻塒廝枲澌緦翤豉釃鍉顋眎旹","柹(柿枾)"),
  "식":("食式植識息飾","栻埴殖湜軾寔拭熄篒蝕媳",""),
  "신":("身申神臣信辛新伸晨愼","紳莘薪迅訊侁呻娠宸燼腎藎蜃辰璶哂囟姺汛矧脤贐頣駪",""),
  "실":("失室實","悉蟋","實(実)"),
  "심":("心甚深尋審","沁沈瀋芯諶潯燖葚鐔鱏",""),
  "십":("十","什拾",""),
  "쌍":("雙","","雙(双)"),
  "씨":("氏","",""),
  "아":("兒我牙芽雅亞餓","娥峨衙妸俄啞莪蛾訝鴉鵝阿婀哦硪皒砑婭椏啊妿猗枒丫疴笌迓錏鵞","兒(児)亞(亜)峨(峩)婀(娿)"),
  "악":("惡岳","樂堊嶽幄愕握渥鄂鍔顎鰐齷偓鄂咢喔噩腭萼覨諤鶚齶",""),
  "안":("安案顔眼岸雁","晏按鞍鮟𤎝姲婩矸侒䭓犴錌妟","雁(鴈)案(桉)"),
  "알":("謁","斡軋閼嘎揠穵訐遏頞鴶",""),
  "암":("暗巖","庵菴唵癌闇啽媕嵓晻腤葊蓭諳頷馣黯","巖(岩)"),
  "압":("壓押","鴨狎",""),
  "앙":("仰央殃","鴦怏秧昂卬坱盎鞅泱","昂(昻)"),
  "애":("愛哀涯","厓崖艾埃曖隘靄䝽礙㶼唉僾啀噯娭崕挨捱欸漄獃皚睚瞹磑薆藹靉騃","礙(碍)"),
  "액":("厄額","液扼掖縊腋呝戹搤阨",""),
  "앵":("","鶯櫻罌鸚嚶嫈罃",""),
  "야":("也夜野耶","冶倻惹椰爺若捓","野(埜)捓(揶)"),
  "약":("弱若約藥躍","葯蒻爚禴篛籥鑰鶸龠",""),
  "양":("羊洋養揚陽讓壤樣楊","襄孃漾佯恙攘暘瀁煬痒瘍禳穰釀椋徉瀼烊癢眻蘘輰鑲颺驤","陽(昜)揚(敭)"),
  "어":("魚漁於語御","圄瘀禦馭齬唹䘘圉敔淤飫",""),
  "억":("億憶抑","檍臆繶",""),
  "언":("言焉","諺彥偃堰嫣傿匽讞鄢鼴鼹嘕鶠","彥(彦)"),
  "얼":("","孼蘖糱乻臬","糱(糵)"),
  "엄":("嚴","奄俺掩儼淹龑崦曮罨醃閹广","嚴(厳)"),
  "업":("業","嶪嶫鄴",""),
  "에":("","恚曀",""),
  "엔":("","円",""),
  "여":("余餘如汝與予輿","歟璵礖艅茹轝妤悆舁伃侞",""),
  "역":("亦易逆譯驛役疫域","晹繹嶧懌淢閾",""),
  "연":("然煙硏延燃燕沿鉛宴軟演緣","衍淵姸娟涓沇筵瑌娫嚥堧捐挻椽涎縯鳶硯曣㜣醼兗嬿莚瓀均戭囦埏悁掾櫞渷臙蜵蠕讌","煙(烟)淵(渊)兗(兖)姸(妍)娟(姢)軟(輭)硯(䂩)"),
  "열":("熱悅閱","說咽潱噎",""),
  "염":("炎染鹽","琰艷厭焰苒閻髥冉懕扊檿灩饜魘黶","艷(艶)"),
  "엽":("葉","燁曄熀曅爗靨枼",""),
  "영":("永英迎榮泳詠營影映","渶煐瑛瑩瀯盈楹鍈嬰穎瓔咏塋嶸潁瀛纓霙嬴𢥏蠑朠浧䀴栐濴癭韺碤縈贏郢旲","榮(栄荣)映(暎)瀯(濚)"),
  "예":("藝豫譽銳","叡預芮乂倪刈曳汭濊猊穢裔詣霓堄橤玴嫕蓺蕊𣫙艾㙯羿瘱郳䢃帠淣兒囈嫛拽掜枘獩睨瞖繄翳薉蚋蜺鯢鷖麑枻医","叡(睿䜭壡)藝(埶芸)蕊(蘂)"),
  "오":("五吾悟午誤烏汚嗚娛傲","伍吳旿珸晤奧俉塢墺寤惡懊敖熬澳獒筽蜈鼇梧浯燠䫨仵俣唔嗷噁圬嫯忤慠捂汙窹聱茣襖謷迃迕遨鏊鏖隩驁鼯","鼇(鰲)"),
  "옥":("玉屋獄","沃鈺",""),
  "온":("溫","瑥媼穩瘟縕蘊𥠺𥁕榲馧䭓媪慍氳熅轀醞韞薀㒚","穩(稳)𥁕(昷)"),
  "올":("","兀杌嗢膃",""),
  "옹":("翁擁","雍壅瓮甕癰邕饔喁廱滃癕禺罋蓊雝顒",""),
  "와":("瓦臥","渦窩窪蛙蝸訛哇囮婐枙洼猧窊萵譌娃",""),
  "완":("完緩","玩垸浣莞琓琬婠婉宛梡椀碗翫脘腕豌阮頑妧岏鋺抏杬刓忨惋涴盌輐",""),
  "왈":("曰","",""),
  "왕":("王往","旺汪枉瀇迬",""),
  "왜":("","倭娃歪矮媧",""),
  "외":("外畏","嵬巍猥偎嵔崴渨煨碨磈聵隗",""),
  "요":("要腰搖遙謠","夭堯饒曜耀瑤樂姚僥凹妖嶢拗擾橈燿窈窯繇繞蟯邀暚偠喓坳墝嬈幺徭徼殀澆祅穾窅蕘遶鷂約",""),
  "욕":("欲浴慾辱","縟褥溽蓐",""),
  "용":("用勇容庸","溶鎔瑢榕蓉涌埇踊鏞茸墉甬俑傭慂聳傛槦宂㦶嵱慵憃硧舂蛹踴","鎔(熔)涌(湧)宂(冗)"),
  "우":("于宇右牛友雨憂又尤遇羽郵愚偶優","佑祐禹瑀寓堣隅玗釪迂䨒旴盂禑紆芋藕虞雩扜圩慪燠㥥俁邘盓亴偊吁嵎庽杅疣盱竽耦耰謣踽鍝麀麌齲訧訏优","雨(㲾)"),
  "욱":("","旭昱煜郁頊彧勖栯燠稢𢒰","稢(稶)"),
  "운":("云雲運韻","沄澐耘賱夽暈橒殞熉芸蕓隕篔䆬員鄖䫟惲紜霣韵妘","篔(䉙)"),
  "울":("","蔚鬱𠃗菀䵥",""),
  "웅":("雄","熊",""),
  "원":("元原願遠園怨圓員源援院","袁垣洹沅瑗媛嫄愿苑轅婉湲爰猿阮鴛褑朊杬鋺冤笎邍倇楥芫薗蜿謜騵鵷黿猨溒","冤(寃)員(貟)"),
  "월":("月越","鉞刖粤",""),
  "위":("位危爲偉威胃謂圍衛違委慰僞緯","尉韋瑋暐渭魏萎葦蔿蝟褘衞韡喟幃熨痿葳諉逶闈韙餧骪煒",""),
  "유":("由油酉有猶唯遊柔遺幼幽惟維乳儒裕誘愈悠","侑洧宥庾喩兪楡瑜猷濡愉秞攸柚琟釉孺揄楢游癒臾萸諛諭踰蹂逾鍮曘婑囿牖逌姷聈蕤甤湵瑈需揉帷冘呦壝泑鼬龥瘉瘐窬窳籲糅緌腴莠蕕蚴蚰蝤褕黝讉鞣鮪燸瀢","兪(俞)濡(㴗)"),
  "육":("肉育","堉毓儥",""),
  "윤":("閏潤","尹允玧鈗胤阭奫贇昀荺鋆橍沇匀","閏(䦞閠)胤(㣧)"),
  "율":("","聿燏汩䢖潏鴥矞䫻霱",""),
  "융":("","融戎瀜絨狨",""),
  "은":("恩銀隱","垠殷誾溵珢慇濦㒚听𤨒圻蘟檼檃訢蒑泿蒽憖圁嶾𪙤𰜩嚚垽狺癮訔鄞齗","誾(𨶡)"),
  "을":("乙","圪鳦",""),
  "음":("音吟飮陰淫","蔭愔韾喑崟廕霪",""),
  "읍":("邑泣","揖悒挹浥",""),
  "응":("應凝","膺鷹𥌾",""),
  "의":("衣依義議矣醫意宜儀疑","倚誼毅擬懿椅艤薏蟻㛄猗儗凒劓嶷欹漪礒饐螘",""),
  "이":("二以已耳而異移夷","珥伊易弛怡爾彝頤姨痍肄苡荑貽邇飴貳媐杝䏪㛅珆鴯羡巸佴廙咿尔栮洟訑迤隶聏貤薾","彝(彛)"),
  "익":("益翼","翊瀷謚翌熤弋鷁",""),
  "인":("人引仁因忍認寅印姻","咽湮絪茵蚓靷刃芢㲽牣璌韌𣍃氤𦟘儿諲濥秵戭仞堙夤婣洇禋裀","韌(靭)𣍃(朄)仁(忈忎)禋(䄄)"),
  "일":("一日逸","溢鎰馹佾佚壹劮泆軼欥","逸(𨓜)"),
  "임":("壬任賃","妊稔恁荏䚾䛘絍衽銋飪","妊(姙)"),
  "입":("入","廿","廿(卄)"),
  "잉":("","剩仍孕芿媵",""),
  "자":("子字自者姊慈茲紫資姿恣刺","仔滋磁藉瓷咨孜炙煮疵茨蔗諮雌秄褯呰嬨孖孶柘泚牸眦眥耔胾茈莿虸觜訾貲赭鎡頿髭鮓鶿鷓粢","姊(姉)茲(玆)"),
  "작":("作昨酌爵","灼芍雀鵲勺嚼斫炸綽舃岝怍斱柞汋焯犳碏",""),
  "잔":("殘","孱棧潺盞剗驏",""),
  "잠":("潛暫","箴岑簪蠶涔","潛(潜)"),
  "잡":("雜","卡囃眨磼襍",""),
  "장":("長章場將壯丈張帳莊裝奬墻葬粧掌藏臟障腸","匠杖奘漳樟璋暲薔蔣仗檣欌漿狀獐臧贓醬傽妝嬙嶂廧戕牂瘴糚羘萇鄣鏘餦麞","將(将)壯(壮)莊(庄)墻(牆)奬(獎)"),
  "재":("才材財在栽再哉災裁載宰","梓縡齋渽滓齎捚賳溨夈崽扗榟灾纔",""),
  "쟁":("爭","錚箏諍崢猙琤鎗",""),
  "저":("著貯低底抵","苧邸楮沮佇儲咀姐杵樗渚狙猪疽箸紵菹藷詛躇這雎齟宁岨杼柢氐潴瀦牴罝羝苴蛆袛褚觝詆陼",""),
  "적":("的赤適敵滴摘寂籍賊跡積績","迪勣吊嫡狄炙翟荻謫迹鏑笛蹟樀磧糴菂覿逖馰",""),
  "전":("田全典前展戰電錢傳專轉殿","佺栓詮銓琠甸塡奠荃雋顚佃剪塼廛悛氈澱煎畑癲筌箋箭篆纏輾鈿鐫顫餞吮囀嫥屇巓戩揃旃栴湔澶牋甎畋痊癜磚籛羶翦腆膞躔輇邅鄽鋑錪靛靦顓飦餰鬋鱣鸇賟",""),
  "절":("節絕切折竊","晢截浙癤岊","絕(絶)"),
  "점":("店占點漸","岾粘霑鮎佔墊玷笘簟苫蔪蛅覘颭黏","點(点奌)"),
  "접":("接蝶","摺椄楪蜨跕蹀鰈",""),
  "정":("丁頂停井正政定貞精情靜淨庭亭訂廷程征整","汀玎町呈桯珵姃偵湞幀楨禎珽挺綎鼎晶晸柾鉦淀錠鋌鄭靖靚鋥炡渟釘涏𩓞婷旌檉瀞睛碇穽艇諪酊霆𩇕埩佂妌梃胜灯眐靘朾侹掟頲叮婧怔棖疔筳莛証酲遉","靜(静)"),
  "제":("弟第祭帝題除諸製提堤制際齊濟","悌梯瑅劑啼臍薺蹄醍霽媞儕禔偙姼晢娣擠猘睇稊緹踶蹏躋鍗隄韲鮧鯷隮","濟(済)"),
  "조":("兆早造鳥調朝助弔燥操照條潮租組祖","彫措晁窕祚趙肇詔釣曹遭眺俎凋嘲棗槽漕爪璪稠粗糟繰藻蚤躁阻雕昭嶆佻傮刁厝嘈噪嬥徂懆找殂澡琱皁祧竈笊糙糶絩絛胙臊艚蔦蜩誂譟鈟銚鋽鯛鵰鼂炤","曹(曺)棗(𠄬)"),
  "족":("足族","簇鏃瘯",""),
  "존":("存尊","拵",""),
  "졸":("卒拙","猝",""),
  "종":("宗種鐘終從縱","倧琮淙悰綜瑽鍾慫腫踵椶柊蹤伀慒樅瘇螽","椶(棕)蹤(踪)"),
  "좌":("左坐佐座","挫剉痤莝髽",""),
  "죄":("罪","",""),
  "주":("主注住朱宙走酒晝舟周株州洲柱奏珠鑄","冑湊炷註疇週遒駐妵澍姝侏做呪嗾廚籌紂紬綢蛛誅躊輳酎燽鉒拄皗邾䎻絑䝬椆㫶珘紸調晭丟侜儔尌幬硃籒鼄胕腠蔟蛀裯詋賙趎輈霌霔胄","遒(逎)"),
  "죽":("竹","粥",""),
  "준":("準俊遵","峻浚晙焌竣畯駿准濬雋儁埻隼寯樽蠢逡純葰竴僔陖睃餕迿惷𨶊懏鐏𢓭皴墫撙綧罇鱒踆蹲鵔僎偆","準(凖)濬(䜭)陖(埈)"),
  "줄":("","茁乼",""),
  "중":("中重衆仲","眾",""),
  "즉":("卽","喞","卽(即)"),
  "즐":("","櫛騭",""),
  "즙":("","汁楫葺檝蕺",""),
  "증":("曾增證憎贈症蒸","烝甑拯繒嶒矰罾",""),
  "지":("只支枝止之知地指志至紙持池誌智遲","旨沚址祉趾祗芝摯鋕脂咫枳漬砥肢芷蜘識贄洔厎汦吱馶劧忯坁搘禔觗坻墀榰泜痣秪篪舐踟躓軹阯鮨鷙扺實潪","知(𥎵)智(𥏾)"),
  "직":("直職織","稙稷禝",""),
  "진":("辰眞進盡振鎭陣陳珍震","晉瑨瑱津璡秦軫塵禛診縝塡賑溱抮唇嗔搢桭榛殄畛疹瞋縉臻蔯袗䑐蓁昣枃槇稹儘靕敒眹侲珒螴趁鬒愼滇誫轃","眞(真)晉(晋)瑨(𤨁)珍(鉁)盡(尽)"),
  "질":("質秩疾姪","瓆侄叱嫉帙桎窒膣蛭跌迭垤絰蒺郅鑕",""),
  "짐":("","斟朕鴆",""),
  "집":("集執","什潗輯楫鏶緝咠戢","潗(𬄕)"),
  "징":("徵懲","澄澂瀓癥瞪",""),
  "차":("且次此借差","車叉瑳侘嗟嵯磋箚茶蹉遮硨奲姹鹺佽岔徣槎",""),
  "착":("着錯捉","搾窄鑿齪戳擉斲",""),
  "찬":("贊讚","撰纂粲澯燦璨瓚纘鑽竄餐饌攢巑儹篡欑㜺劗爨趲攛","贊(賛)讚(讃)儹(儧)篡(簒)"),
  "찰":("察","札刹擦紮扎",""),
  "참":("參慘慙","僭塹懺斬站讒讖儳嶄巉憯攙槧欃毚譖鏨鑱饞驂黲","慙(慚)"),
  "창":("昌唱窓倉創蒼暢","菖昶彰敞廠倡娼愴槍漲猖瘡脹艙滄淐晿淌倀傖凔刱悵惝戧搶氅瑲窗蹌鋹閶鬯鶬",""),
  "채":("菜採彩債","采埰寀蔡綵寨砦釵琗責棌婇睬茝",""),
  "책":("責冊策","柵嘖幘磔笧簀蚱","冊(册)"),
  "처":("妻處","凄悽淒萋覷郪",""),
  "척":("尺斥拓戚","陟倜刺剔擲滌瘠脊蹠隻墌慼塉惕捗摭蜴跖躑","墌(坧)慼(慽)"),
  "천":("天千川泉淺賤踐遷薦","仟阡喘擅玔穿舛釧闡韆茜俴倩僢儃洊濺祆臶芊荐蒨蕆辿靝",""),
  "철":("鐵哲徹","澈撤轍綴凸輟悊瞮剟啜埑惙掇歠銕錣飻餮","哲(喆)鐵(鉄)"),
  "첨":("尖添","僉瞻沾簽籤詹諂甜幨忝惉檐櫼瀸簷襜","甜(甛)"),
  "첩":("妾","帖捷堞牒疊睫諜貼輒倢呫喋怗褺",""),
  "청":("靑淸晴請廳聽","菁鯖凊圊蜻鶄婧","靑(青)淸(清)晴(晴)請(請)"),
  "체":("體替遞滯逮","締諦切剃涕諟玼棣彘殢砌蒂髰蔕靆",""),
  "초":("初草招肖超抄礎秒","樵焦蕉楚剿哨憔梢椒炒硝礁稍苕貂酢醋醮岧釥俏𪓐偢僬勦噍嫶峭嶕怊悄愀杪燋綃耖誚譙趠軺迢鈔鍬鍫鞘顦髫鷦齠妱","草(艸)"),
  "촉":("促燭觸","囑矗蜀曯爥矚薥躅髑",""),
  "촌":("寸村","忖吋","村(邨)"),
  "총":("銃總聰","寵叢悤憁摠蔥冢葱蓯鏦驄","聰(聡)冢(塚)總(総)"),
  "촬":("","撮",""),
  "최":("最催","崔嘬摧榱漼璀磪縗脧",""),
  "추":("秋追推抽醜","楸樞鄒錐錘墜椎湫皺芻萩諏趨酋鎚雛騶鰌僦啾娵帚惆捶揫搥甃瘳箠簉縋縐蒭陬隹鞦騅魋鵻鶖鶵麤龝","鰌(鰍)"),
  "축":("丑祝蓄畜築逐縮","軸竺筑蹙蹴妯舳豖蹜鼀",""),
  "춘":("春","椿瑃賰",""),
  "출":("出","朮黜秫",""),
  "충":("充忠蟲衝","珫沖衷忡","蟲(虫)沖(冲)"),
  "췌":("","萃悴膵贅惴揣瘁顇",""),
  "취":("取吹就臭醉趣","翠聚嘴娶炊脆驟鷲冣橇毳",""),
  "측":("側測","仄惻廁昃","廁(厠)"),
  "층":("層","",""),
  "치":("治致齒値置恥","熾峙雉馳侈嗤幟梔淄痔癡緇緻蚩輜稚卮哆寘畤痓絺菑薙褫豸跱錙阤鯔鴟鴙鵄","癡(痴)稚(穉)"),
  "칙":("則","勅飭敕",""),
  "친":("親","櫬襯",""),
  "칠":("七漆","柒",""),
  "침":("針侵浸寢沈枕","琛砧鍼棽寖忱椹郴鋟駸",""),
  "칩":("","蟄",""),
  "칭":("稱","秤",""),
  "쾌":("快","夬噲",""),
  "타":("他打妥墮","咤唾惰拖朶舵陀馱駝橢佗坨拕柁沱詑詫跎躱駞鮀鴕鼉","橢(楕)"),
  "탁":("濁托濯卓","度倬琸晫託擢鐸拓啄坼柝琢踔橐拆沰涿矺籜蘀逴","橐(槖)"),
  "탄":("炭歎彈誕","呑坦灘嘆憚綻暺憻攤殫癱驒",""),
  "탈":("脫奪","侻",""),
  "탐":("探貪","耽眈嗿忐酖",""),
  "탑":("塔","榻傝塌搨",""),
  "탕":("湯","宕帑糖蕩燙盪碭蘯",""),
  "태":("太泰怠殆態","汰兌台胎邰笞苔跆颱鈦珆鮐脫娧迨埭孡駘",""),
  "택":("宅澤擇","垞",""),
  "탱":("","撑撐牚",""),
  "터":("","攄",""),
  "토":("土吐討","兔","兔(兎)"),
  "톤":("","噋",""),
  "통":("通統痛","桶慟洞筒恫樋筩",""),
  "퇴":("退","堆槌腿褪頹隤",""),
  "투":("投透鬪","偸套妬妒渝骰",""),
  "퉁":("","佟",""),
  "특":("特","慝忒",""),
  "틈":("","闖",""),
  "파":("破波派播罷頗把","巴芭琶坡杷婆擺爬跛叵妑岥怕灞爸玻皤笆簸耙菠葩鄱",""),
  "판":("判板販版","阪坂瓣辦鈑",""),
  "팔":("八","叭捌朳汃",""),
  "패":("貝敗","浿佩牌唄悖沛狽稗霸孛旆珮霈","霸(覇)"),
  "팽":("","彭澎烹膨砰祊蟚蟛",""),
  "퍅":("","愎",""),
  "편":("片便篇編遍偏","扁翩鞭騙匾徧惼緶艑萹蝙褊諞",""),
  "폄":("","貶砭窆",""),
  "평":("平評","坪枰泙萍怦抨苹蓱鮃",""),
  "폐":("閉肺廢弊蔽幣","陛吠嬖斃敝狴獘癈",""),
  "포":("布抱包胞飽浦捕","葡褒砲鋪佈匍匏咆哺圃怖暴泡疱脯苞蒲袍逋鮑拋儤庖晡曓炮炰誧鉋鞄餔鯆","拋(抛)"),
  "폭":("暴爆幅","曝瀑輻",""),
  "표":("表票標漂","杓豹彪驃俵剽慓瓢飄飆䏇僄勡嘌嫖摽殍熛縹裱鏢鑣髟鰾","飆(飇)"),
  "품":("品","稟",""),
  "풍":("風豐","諷馮楓瘋","豐(豊)"),
  "피":("皮彼疲被避","披陂詖鞁髲",""),
  "픽":("","腷",""),
  "필":("必匹筆畢","弼泌珌苾馝鉍佖疋㳼㪤咇滭篳罼蓽觱蹕鞸韠鵯駜",""),
  "핍":("","乏逼偪",""),
  "하":("下夏賀何河荷","廈霞瑕蝦遐鰕呀嘏碬閜嚇赮𧬂煆蕸㰤抲㗿岈懗瘕罅鍜㦆","廈(厦)夏(昰)"),
  "학":("學鶴","壑虐謔嗃狢瘧皬确郝鷽","學(学)"),
  "한":("閑寒恨限韓漢旱汗","澣瀚翰閒悍罕澖𡽜僩嫺橌䦥扞忓邗嫻捍暵閈駻鷳鼾邯雗",""),
  "할":("割","轄瞎",""),
  "함":("咸含陷","函涵艦喊檻緘銜鹹菡莟諴轞闞","銜(啣)"),
  "합":("合","哈盒蛤閤闔陜匌嗑柙榼溘盍郃",""),
  "항":("恒巷港項抗航","亢沆姮伉杭桁缸肛行降夯炕缿頏","恒(恆)姮(嫦)"),
  "해":("害海亥解奚該","偕楷諧咳垓孩懈瀣蟹邂駭骸咍瑎澥祄晐嶰廨欬獬痎薤醢頦鮭䀭絯姟陔","海(海)"),
  "핵":("核","劾翮覈",""),
  "행":("行幸","杏倖荇涬悻",""),
  "향":("向香鄕響享","珦嚮餉饗麘晑薌",""),
  "허":("虛許","墟噓歔",""),
  "헌":("軒憲獻","櫶䡣㦥昍巚幰攇",""),
  "헐":("","歇",""),
  "험":("險驗","嶮獫玁",""),
  "혁":("革","赫爀奕焱侐焃𧹽嚇弈洫鬩",""),
  "현":("現賢玄絃縣懸顯","見峴晛泫炫玹鉉眩昡絢呟俔睍舷衒弦儇譞怰䧋鋗㢺琄嬛娊妶灦㭹駽痃繯翾蜆誢讂鋧","顯(顕)"),
  "혈":("血穴","孑頁絜趐",""),
  "혐":("嫌","",""),
  "협":("協脅","俠挾峽浹夾狹莢鋏頰冾匧叶埉恊悏愜篋","脅(脇)"),
  "형":("兄刑形亨螢衡","型邢珩泂炯瑩瀅馨熒滎灐荊鎣迥侀夐娙詗陘","迥(逈)"),
  "혜":("惠慧兮","蕙彗譿寭憓暳蹊醯鞋譓鏸匸䚷傒嘒徯槥盻謑橞潓","惠(恵)"),
  "호":("戶乎呼好虎號湖互胡浩毫豪護","晧皓昊淏濠灝祜琥瑚頀顥扈鎬壕壺濩滸岵弧狐瓠糊縞葫蒿蝴皞婋芐犒鄗熩嫭怙瓳蔰儫冱嘷鬍嫮沍滈滬猢皜餬聕醐杲昈虍","芐(芦)浩(澔)號(号)"),
  "혹":("或惑","酷熇",""),
  "혼":("婚混昏魂","渾琿俒䫟圂湣溷焜閽",""),
  "홀":("忽","惚笏囫",""),
  "홍":("紅洪弘鴻","泓烘虹鉷哄汞訌晎澒篊鬨",""),
  "화":("火化花貨和話畫華禾禍","嬅樺譁靴澕俰嘩驊龢","畫(畵)"),
  "확":("確穫擴","廓攫矍矡礭鑊","確(碻)"),
  "환":("歡患丸換環還","喚奐渙煥晥幻桓鐶驩宦紈鰥圜皖洹寰懽擐瓛睆絙豢轘鍰鬟瑍",""),
  "활":("活","闊滑猾豁蛞","闊(濶)"),
  "황":("黃皇況荒","凰堭媓晃滉榥煌璜熀幌徨恍惶愰慌湟潢篁簧蝗遑隍楻喤怳瑝肓貺鎤","晃(晄)"),
  "회":("回會悔懷","廻恢晦檜澮繪誨匯徊淮獪膾茴蛔賄灰佪洄盔詼迴頮鱠","繪(絵)會(会)"),
  "획":("獲劃","画嚄",""),
  "횡":("橫","鐄宖澋鈜黌",""),
  "효":("孝效曉","涍爻驍斅哮嚆梟淆肴酵皛歊窙謼傚洨庨虓熇烋婋囂崤殽餚恔侾","效(効)"),
  "후":("後厚侯候","后逅吼嗅帿朽煦珝喉堠㰭姁芋吽喣垕猴篌詡譃酗餱矦","厚(垕)"),
  "훈":("訓","勳焄熏薰壎燻鑂暈纁煇薫曛獯葷","勳(勛勲)薰(𬟓)壎(塤)熏(𤋱)"),
  "훌":("","欻",""),
  "훙":("","薨",""),
  "훤":("","喧暄萱煊愃昍烜諠諼",""),
  "훼":("毁","喙毀𠦄燬芔虺","𠦄(卉)"),
  "휘":("揮輝","彙徽暉煇諱麾煒撝翬",""),
  "휴":("休携","烋畦虧庥咻隳髹鵂",""),
  "휼":("","恤譎鷸卹遹鐍霱",""),
  "흉":("凶胸","兇匈洶恟胷",""),
  "흑":("黑","",""),
  "흔":("","欣炘昕痕忻很掀惞釁",""),
  "흘":("","屹吃紇訖仡汔疙迄齕",""),
  "흠":("","欽欠歆鑫廞",""),
  "흡":("吸","洽恰翕噏歙潝翖",""),
  "흥":("興","",""),
  "희":("希喜稀戲","姬晞僖橲禧嬉憙熹凞羲爔曦俙𡅕憘犧噫煕烯暿譆㜯咥唏嘻悕欷燹豨餼巸","煕(熙熈)熹(熺)戲(戱)姬(姫)"),
  "히":("","屎",""),
  "힐":("","詰犵纈襭頡黠","")
}