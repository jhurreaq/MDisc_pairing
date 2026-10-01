"""Treloar data; original numerical and presentation choices."""

import numpy as np
import pandas as pd
import os


def load_treloar_data(include_data=True, *, context):
    if not include_data: print("Skipping Treloar data."); return pd.DataFrame()
    base_path = "."; ut_file_path = os.path.join(base_path, "input/TreloarDataUT.csv"); ps_file_path = os.path.join(base_path, "input/TreloarDataPS.csv"); ebt_file_path = os.path.join(base_path, "input/TreloarDataEBT.csv")
    all_data = []; loaded_files = 0
    if os.path.exists(ut_file_path):
        try: ut_df=pd.read_csv(ut_file_path,sep=';',decimal='.',skiprows=1,header=None,names=['lambda','P11']); [all_data.append({'lambda1':r['lambda'],'lambda2':1./np.sqrt(max(r['lambda'],context.EPS)),'lambda3':1./np.sqrt(max(r['lambda'],context.EPS)),'P11':r['P11'],'P22':0.0,'mode':'UT','dataset':'Treloar'}) for _,r in ut_df.iterrows()]; print(f'Loaded Treloar UT: {ut_file_path}'); loaded_files+=1
        except Exception as e: print(f'Error UT: {e}')
    else: print(f'UT file not found: {ut_file_path}')
    if os.path.exists(ps_file_path):
        try: ps_df=pd.read_csv(ps_file_path,sep=';',decimal='.',skiprows=1,header=None,names=['lambda','P11']); [all_data.append({'lambda1':r['lambda'],'lambda2':1.0,'lambda3':1./max(r['lambda'],context.EPS),'P11':r['P11'],'P22':np.nan,'mode':'PS','dataset':'Treloar'}) for _,r in ps_df.iterrows()]; print(f'Loaded Treloar PS: {ps_file_path}'); loaded_files+=1
        except Exception as e: print(f'Error PS: {e}')
    else: print(f'PS file not found: {ps_file_path}')
    if os.path.exists(ebt_file_path):
        try: ebt_df=pd.read_csv(ebt_file_path,sep=';',decimal='.',skiprows=1,header=None,names=['lambda','P11']); [all_data.append({'lambda1':r['lambda'],'lambda2':r['lambda'],'lambda3':1./max(r['lambda']**2,context.EPS),'P11':r['P11'],'P22':r['P11'],'mode':'EBT','dataset':'Treloar'}) for _,r in ebt_df.iterrows()]; print(f'Loaded Treloar EBT: {ebt_file_path}'); loaded_files+=1
        except Exception as e: print(f'Error EBT: {e}')
    else: print(f'EBT file not found: {ebt_file_path}')
    if not all_data: print("WARN: No Treloar data loaded."); return pd.DataFrame()
    treloar_df = pd.DataFrame(all_data); treloar_df['strain_pct'] = (treloar_df['lambda1'] - 1.0) * 100.0; return treloar_df

def load_Kawabata_data(csv_path="input/Kawabata.csv", include_data=True, *, context):
    if not include_data: print("Skipping Kawabata data."); return pd.DataFrame()
    if not os.path.exists(csv_path): print(f'Kawabata file not found: {csv_path}'); return pd.DataFrame()
    df = None
    try: df=pd.read_csv(csv_path,sep=';',decimal='.',skiprows=1); df.columns=['lambda1','lambda2','P11','P22'][:len(df.columns)]; print(f"Loaded Kawabata (';')")
    except Exception: print(f"Info: Kawabata with ';' failed. Trying ','."); df=None
    if df is None:
        try: df=pd.read_csv(csv_path,sep=',',decimal='.',skiprows=1); df.columns=['lambda1','lambda2','P11','P22'][:len(df.columns)]; print(f"Loaded Kawabata (',')")
        except Exception as e2: print(f'Error: Failed Kawabata {csv_path}: {e2}'); return pd.DataFrame()
    if len(df.columns)<4: print(f'Error: Kawabata needs at least 4 cols.'); return pd.DataFrame()
    df['lambda3']=1./(df['lambda1'].apply(lambda x:max(x,context.EPS))*df['lambda2'].apply(lambda x:max(x,context.EPS))); df['lambda3']=df['lambda3'].apply(lambda x:max(x,context.EPS))
    df['strain_pct']=(df['lambda1']-1.)*100.; df['mode']='BIAX'; df['dataset']='Kawabata'; print(f'Processed {len(df)} Kawabata points.')
    for col in ['P11','P22']: df[col]=pd.to_numeric(df[col], errors='coerce') # Ensure numeric
    return df[['lambda1','lambda2','lambda3','P11','P22','mode','dataset','strain_pct']]

def combine_datasets(include_treloar=True, include_Kawabata=True, Kawabata_path="input/Kawabata.csv", *, context):
    print("\n--- Loading Data ---"); df_treloar=load_treloar_data(include_data=include_treloar, context=context); df_kawabata=load_Kawabata_data(csv_path=Kawabata_path,include_data=include_Kawabata, context=context)
    combined_df=pd.concat([df_treloar,df_kawabata],ignore_index=True)
    if len(combined_df)==0: raise ValueError("No data loaded. Check file paths and include flags.")
    print(f'\n--- Dataset Summary ---'); print(f'Total data points combined: {len(combined_df)}'); print(f"Sources: {combined_df['dataset'].unique()}"); print(f"Modes: {combined_df['mode'].unique()}")
    return combined_df.reset_index(drop=True)
