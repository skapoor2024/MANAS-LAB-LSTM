import numpy as np 
import pandas as pd 

#get_cavg code mentioned in lines 82-117 of https://github.com/Snowdar/asv-subtools

def get_cavg(pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
  ''' Compute Cavg, using several threshhold bins in [min_score, max_score].
  '''
  cavgs = [0.0] * (bins + 1)
  precision = (max_score - min_score) / bins
  for section in range(bins + 1):
    threshold = min_score + section * precision
    # Cavg for each lang: p_target * p_miss + sum(p_nontarget*p_fa)
    target_cavg = [0.0] * lang_num
    for lang in range(lang_num):
      p_miss = 0.0 # prob of missing target pairs
      LTa = 0.0 # num of all target pairs
      LTm = 0.0 # num of missing pairs
      p_fa = [0.0] * lang_num # prob of false alarm, respect to all other langs
      LNa = [0.0] * lang_num # num of all nontarget pairs, respect to all other langs
      LNf = [0.0] * lang_num # num of false alarm pairs, respect to all other langs
      for line in pairs:
        if line[0] == lang:
          if line[1] == lang:
            LTa += 1
            if line[2] < threshold:
              LTm += 1
          else:
            LNa[line[1]] += 1
            if line[2] >= threshold:
              LNf[line[1]] += 1
      if LTa != 0.0:
        p_miss = LTm / LTa
      for i in range(lang_num):
        if LNa[i] != 0.0:
          p_fa[i] = LNf[i] / LNa[i]
      p_nontarget = (1 - p_target) / (lang_num - 1)
      target_cavg[lang] = p_target * p_miss + p_nontarget*sum(p_fa)
    cavgs[section] = sum(target_cavg) / lang_num

  return cavgs, min(cavgs)

#here use the location of AP20_task1_4.csv file for df

df = pd.read_csv('/home/skapr-13/Desktop/wssl/OLR_2020/AP20_task1_4.csv',encoding='utf-8',usecols=list(range(1,7)))

dt = df.astype(np.float32)

maxx = dt.max().max() #maximum score present in the dataframe dt
minn = dt.min().min() #minimum score present in the dataframe dt

print(maxx,minn)

#here use the location of pair.csv file  for df_p

df_p = pd.read_csv('/home/skapr-13/Desktop/wssl/OLR_2020/pair.csv',encoding='utf-8',usecols=list(range(1,8)))

#to calculate pairs required to compute cavg

pairs = []
for r in range(11848):#11848 denote the number of utterances present
    for i in range(6):#6 number of languages present in the challenege for testing.
        s = str(i)
        pairs.append([s,df_p['6'].values[r],df_p[s].values[r]])
        #df_p['6'] denote the utt2lang_id i.e ground truth for the utternace in lang_ids

lang_num = 6
cavg, min_cavg = get_cavg(pairs,lang_num,minn,maxx)
print(cavg)
print(min_cavg)
