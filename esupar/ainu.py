#! /usr/bin/python3 -i
# coding=utf-8
katakana={
  "ア":"a",
  "イ":"i",
  "ウ":"u",
  "エ":"e",
  "オ":"o",
  "ガ":"ga",
  "ギ":"gi",
  "グ":"gu",
  "ゲ":"ge",
  "ゴ":"go",
  "カ":"ka",
  "キ":"ki",
  "ク":"ku",
  "ケ":"ke",
  "コ":"ko",
  "ザ":"za",
  "ジ":"zi",
  "ズ":"zu",
  "ゼ":"ze",
  "ゾ":"zo",
  "サ":"sa",
  "シ":"si",
  "ス":"su",
  "セ":"se",
  "ソ":"so",
  "ダ":"da",
  "ヂ":"ji",
  "ヅ":"du",
  "デ":"de",
  "ド":"do",
  "タ":"ta",
  "チ":"ci",
  "ツ":"tu",
  "テ":"te",
  "ト":"to",
  "ナ":"na",
  "ニ":"ni",
  "ヌ":"nu",
  "ネ":"ne",
  "ノ":"no",
  "パ":"pa",
  "ピ":"pi",
  "プ":"pu",
  "ペ":"pe",
  "ポ":"po",
  "バ":"ba",
  "ビ":"bi",
  "ブ":"bu",
  "ベ":"be",
  "ボ":"bo",
  "ハ":"ha",
  "ヒ":"hi",
  "フ":"hu",
  "ヘ":"he",
  "ホ":"ho",
  "マ":"ma",
  "ミ":"mi",
  "ム":"mu",
  "メ":"me",
  "モ":"mo",
  "ヤ":"ya",
  "ユ":"yu",
  "ヨ":"yo",
  "ラ":"ra",
  "リ":"ri",
  "ル":"ru",
  "レ":"re",
  "ロ":"ro",
  "ワ":"wa",
  "ヰ":"wi",
  "ヱ":"we",
  "ヲ":"wo",
  "ン":"n",
  "ァ":"a",
  "ィ":"y",
  "ゥ":"w",
  "ェ":"e",
  "ォ":"o",
  "ㇰ":"k",
  "ㇱ":"s",
  "ㇲ":"s",
  "ッ":"t",
  "ㇴ":"n",
  "ㇷ゚":"p",
  "ㇵ":"h",
  "ㇶ":"h",
  "ㇷ":"h",
  "ㇸ":"h",
  "ㇹ":"h",
  "ㇺ":"m",
  "ャ":"a",
  "ュ":"u",
  "ョ":"o",
  "ㇻ":"r",
  "ㇼ":"r",
  "ㇽ":"r",
  "ㇾ":"r",
  "ㇿ":"r",
  "ー":"-",
  "。":".",
  "、":",",
  "．":".",
  "，":",",
  "「":'"',
  "」":'"'
  "『":'"',
  "』":'"'
}
cyrillic={
  "а":"a",
  "б":"b",
  "в":"v",
  "г":"g",
  "д":"d",
  "е":"ye",
  "ё":"yo",
  "ж":"j",
  "з":"z",
  "и":"i",
  "й":"y",
  "к":"k",
  "л":"l",
  "м":"m",
  "н":"n",
  "о":"o",
  "п":"p",
  "р":"r",
  "с":"s",
  "т":"t",
  "у":"u",
  "ф":"f",
  "х":"h",
  "ц":"c",
  "ч":"c",
  "ш":"s",
  "щ":"s",
  "ъ":"'",
  "ы":"wi",
  "ь":"'",
  "э":"e",
  "ю":"yu",
  "я":"ya",
  "і":"i"
}
class Lemmatize(object):
  def __init__(self):
    self.dict={**katakana,**cyrillic}
    self.rev_katakana={v:k for k,v in reversed(katakana.items())}
    self.rev_cyrillic={v:k for k,v in reversed(cyrillic.items())}
  def __call__(self,text):
    return "".join(self.dict[c] if c in self.dict else c for c in text.replace("ㇷ゚","p").lower())
  def divide(self,text):
    if len(text)==0:
      return "_","_"
    elif len(text)>1:
      return text[0:-1],text[-1]
    import unicodedata
    c=unicodedata.name(text)
    if c.startswith("KATAKANA"):
      if text in katakana:
        t=katakana[text]
        if len(t)==2:
          return [self.rev_katakana[x] if x in self.rev_katakana else "ッ" for x in t]
    elif c.startswith("CYRILLIC"):
      if text.lower() in cyrillic:
        t=cyrillic[text.lower()]
        if len(t)==2:
          return [self.rev_cyrillic[x] if x in self.rev_cyrillic else "ь" for x in t]
    return text,text
