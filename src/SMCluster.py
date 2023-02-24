# preprocessing, EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_palette('hls')
plt.rcParams['font.family'] = 'Malgun Gothic'
import plotly.express as px
import chart_studio.plotly as py

import warnings
warnings.filterwarnings('ignore')

# clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class SMC:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path).drop(['ID', 'ID_fam'], axis = 1)
        self.feature = ['year', 'region', 'town_t', 'sex', 'age', 'incm', 'ho_incm', 'incm5', 'ho_incm5', 'edu', 'occp', 'cfam', 'genertn', 'allownc', 'marri_1', 'marri_2', 'fam_rela', 'tins', 'D_1_1', 'educ', 'EC1_1', 'EC_wht_23', 'EC_wht_5', 'EC_pedu_1', 'EC_pedu_2', 'BD1_11', 'BD2_1', 'BD2_31', 'dr_month', 'BP6_10', 'BP7', 'mh_stress', 'BS3_1', 'BE3_31', 'BE5_1', 'LW_mt', 'LW_mt_a1', 'LW_br', 'HE_fst', 'HE_HPdr', 'HE_DMdr', 'HE_mens', 'HE_prg', 'HE_HPfh1', 'HE_HPfh2', 'HE_HPfh3', 'HE_HLfh1', 'HE_HLfh2', 'HE_HLfh3', 'HE_IHDfh1', 'HE_IHDfh2', 'HE_IHDfh3', 'HE_STRfh1', 'HE_STRfh2', 'HE_STRfh3', 'HE_DMfh1', 'HE_DMfh2', 'HE_DMfh3', 'HE_rPLS', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG', 'HE_ast', 'HE_alt', 'HE_hepaB', 'HE_HB', 'HE_HCT', 'HE_BUN', 'HE_crea', 'HE_WBC', 'HE_RBC', 'HE_Bplt', 'HE_Uph', 'HE_Unitr', 'HE_Usg', 'HE_Upro', 'HE_Uglu', 'HE_Uket', 'HE_Ubil', 'HE_Ubld', 'HE_Uro', 'HE_Ucrea', 'N_INTK', 'N_EN', 'N_WATER', 'N_PROT', 'N_FAT', 'N_CHO', 'N_CA', 'N_PHOS', 'N_FE', 'N_NA', 'N_K', 'N_CAROT', 'N_RETIN', 'N_B1', 'N_B2', 'N_NIAC', 'N_VITC']
        self.target = ['비만', '고혈압', '당뇨병', '고콜레스테롤혈증', '고중성지방혈증', 'B형간염', '빈혈', '뇌졸중', '협심증또는심근경색증', '천식', '아토피피부염', '골관절염', '우울증']
        self.num_col = ['EC_wht_23', 'HE_fst', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG', 'HE_ast', 'HE_alt', 'HE_HB', 'HE_HCT', 'HE_BUN', 'HE_crea', 'HE_WBC', 'HE_RBC', 'HE_Bplt', 'HE_Uph', 'HE_Usg', 'HE_Ucrea', 'N_INTK', 'N_EN', 'N_WATER', 'N_PROT', 'N_FAT', 'N_CHO', 'N_CA', 'N_PHOS', 'N_FE', 'N_NA', 'N_K', 'N_CAROT', 'N_RETIN', 'N_B1', 'N_B2', 'N_NIAC', 'N_VITC']
        self.cat_col = ['region', 'town_t', 'sex', 'incm', 'ho_incm', 'incm5', 'ho_incm5', 'edu', 'occp', 'cfam', 'genertn', 'allownc', 'marri_1', 'marri_2', 'fam_rela', 'tins', 'D_1_1', 'educ', 'EC1_1', 'EC_wht_5', 'EC_pedu_1', 'EC_pedu_2', 'BD1_11', 'BD2_1', 'BD2_31', 'dr_month', 'BP6_10', 'BP7', 'mh_stress', 'BS3_1', 'BE3_31', 'BE5_1', 'LW_mt', 'LW_br', 'HE_HPdr', 'HE_DMdr', 'HE_mens', 'HE_prg', 'HE_HPfh1', 'HE_HPfh2', 'HE_HPfh3', 'HE_HLfh1', 'HE_HLfh2', 'HE_HLfh3', 'HE_IHDfh1', 'HE_IHDfh2', 'HE_IHDfh3', 'HE_STRfh1', 'HE_STRfh2', 'HE_STRfh3', 'HE_DMfh1', 'HE_DMfh2', 'HE_DMfh3', 'HE_rPLS', 'HE_hepaB', 'HE_Unitr', 'HE_Upro', 'HE_Uglu', 'HE_Uket', 'HE_Ubil', 'HE_Ubld', 'HE_Uro']
        
    
    def vis_target(self):  
    # 각 질병의 유무 비율 시각화
    
        for col in self.target:
            print('='*80)
            print(col)
            print(self.data[col].value_counts() / self.data.shape[0])
            print('='*80)

            plt.figure(figsize=(12, 5))
            sns.countplot(self.data[col], palette='Set2')
            plt.xticks([0, 1], ['1', '0'])
            plt.show()

    
    def PCA(self, data, num_pca = 2):
        x = data.drop(self.target, axis = 1).values
        y = data[self.target].values
        
        x = StandardScaler().fit_transform(x)
        
        pca = PCA(n_components=num_pca)
        printcipalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=printcipalComponents)
        return principalDf
    
    
    def tsne(self, data, num_tsne = 2):
        # class target 정보 제외
        train_df = data[self.feature]

        # 2차원 t-SNE 임베딩
        tsne_np = TSNE(n_components = num_tsne).fit_transform(train_df)

        # numpy array -> DataFrame 변환
        tsne_df = pd.DataFrame(tsne_np)
        return tsne_df
    
    
    def cluster(self, df, kind = 'k-means', num_clus=4, test_k = False):
        if kind == 'k-means':
            if test_k:
                ks = range(1,20)   # 1~19개의 k로 클러스터링하기 위함
                inertias = []    # 응집도 결과 저장을 위한 빈 리스트 만들어 놓기
                for k in ks :
                    model = KMeans(n_clusters = k, n_init = 5)  # n_init 숫자 클수록 계산량 증가하니까 grid search 단계에서는 작게 설정
                    model.fit(df)    # 'df'라는 이름의 dataset을 사용하여 모델 학습 & 학습에 소요되는 시간 측정  
                    inertias.append(model.inertia_)    # 응집도 결과를 inertias 리스트에 계속 저장(추가)
                    print('n_cluster : {}, inertia : {}'.format(k, model.inertia_))    # k 설정에 따른 결과 출력

                # Visualization
                plt.figure(figsize = (15, 6))   
                plt.plot(ks, inertias, '-o')    
                plt.xlabel('number of clusters, k')    
                plt.ylabel('inertia')    
                plt.xticks(ks)    
                plt.show()

            model = KMeans(init = 'k-means++', n_clusters = num_clus, random_state = 0)
            model.fit(df)
            df['cluster'] = model.labels_
            return df

    
    
    
#     def cluster(self, kinds = 'k-means', size = (10,10), num_clus = 4, num_pca = 2, target = False, visualize = True):
#     # kinds : 클러스터링 종류('k-means') / size : 시각화 플롯 사이즈(tuple) / num_clus : 군집 개수 / num_pca : pca 차원 개수
#     # target : target변수 추가 유무(True, False) / visualize : 시각화 유무(True, False)
        
#         self.num_clus = num_clus  # 군집 개수
        
#         if kinds == 'k-means':  # 사용 클러스터 : k-means
            
#             # target 변수 미포함
#             if not target: 
#                 # 각 feature 표준화
#                 standard_scaler = StandardScaler()
#                 df_scaled = pd.DataFrame(standard_scaler.fit_transform(self.data[self.feature]), columns=self.feature)
                
#                 # kmeans 불러오기
#                 kmeans = KMeans(init = 'k-means++', n_clusters = num_clus, random_state = 0)
#                 clusters = kmeans.fit(df_scaled)
                
#                 # 본 데이터프레임에 cluster 피처 추가
#                 self.data['cluster'] = clusters.labels_
                
#                 # PCA
#                 X = df_scaled.copy()
#                 pca = PCA(n_components=num_pca)
#                 pca.fit(X)
#                 x_pca = pca.transform(X)
                
#                 pca_df = pd.DataFrame(x_pca)
#                 pca_df['cluster'] = self.data['cluster']
                
#                 # visualize pca
#                 if visualize:
#                     axs = plt.subplots(figsize = size)
#                     axs = sns.scatterplot(0, 1, hue='cluster', data=pca_df)
            
            
#             # target 변수 포함
#             if target: 
#                 # 각 feature 표준화
#                 standard_scaler = StandardScaler()
#                 df_scaled = pd.DataFrame(standard_scaler.fit_transform(self.data), columns=self.feature + self.target)
                
#                 # kmeans 불러오기
#                 kmeans = KMeans(init = 'k-means++', n_clusters = num_clus, random_state = 0)
#                 clusters = kmeans.fit(df_scaled)
                
#                 # 본 데이터프레임에 cluster 피처 추가
#                 self.data['cluster'] = clusters.labels_
                
#                 # PCA
#                 X = df_scaled.copy()
#                 pca = PCA(n_components=num_pca)
#                 pca.fit(X)
#                 x_pca = pca.transform(X)
                
#                 pca_df = pd.DataFrame(x_pca)
#                 pca_df['cluster'] = self.data['cluster']
                
#                 # visualize pca
#                 if visualize:
#                     axs = plt.subplots(figsize = size)
#                     axs = sns.scatterplot(0, 1, hue='cluster', data=pca_df)


    def two_dimension_cluster_visualize(self, data):
        fig = px.scatter(data, x=0, y=1, color='cluster')
        fig.show()

    
    def cluster_target_visualize(self, clustering_data, ylim_list = [0,0.6]):
        df = self.data
        df['cluster'] = clustering_data['cluster']

        for i in range(len(clustering_data['cluster'].unique())):
            print(f"군집 : {i}")
            i_df = df[df['cluster']==i]

            target_num = []
            for ii in range(len(self.target)):
                target = sum(i_df[self.target[ii]] == 1) / len(i_df)
                target_num.append(target)

            print('='*80)
            print(f"{i}군집의 각 target 피처의 확률 : \n{target_num}")
            print('='*80)
            print(f"{i}군집의 각 target 피처 시각화")
            fig = plt.figure(figsize = (10,5), dpi=100)
            ax = fig.subplots()

            data = target_num
            X = range(len(target_num))
            ax.bar(X, data)
            ax.set_xticks(X)
            ax.set_xticklabels(self.target, rotation=80, fontsize = 16)
            plt.ylim(ylim_list)
            plt.show()
            print('='*80)
            print('='*80)
    
    
    
    
    
#     def analyze_cluster(self, size = (10,5), ylim = [0, 0.6], rate_extact = True, visualize = True):
#         # size : 시각화 플롯 사이즈(tuple) / ylim : y축 범위 설정(list) / rate_extract : 각 비율 추출 유무(True, False)
#         # visualize : 시각화 유무(True, False)
        
#         for col in self.target:
#             if visualize:
#                 fig, ax = plt.subplots(1,1, figsize = size, constrained_layout=True)
#                 fig.suptitle(f'질병 : {col}', fontsize = 20)
            
#             li = []
#             for i in range(self.num_clus):
#                 if rate_extact:
#                     print('='*80)
#                     print(f"column : {col}  /  Cluster : {i}")
                    
#                 df_disease = self.data[self.data[col] == 0]
#                 df_clus = self.data[self.data['cluster'] == i]
                
#                 if rate_extact:
#                     print(f'rate : {len(df_clus)/len(df_disease)}')
#                 li.append(len(df_clus)/len(df_disease))
            
#             if visualize:
#                 plt.ylim(ylim)
#                 plt.bar(range(self.num_clus), li)
#                 plt.show()
#                 print('\n\n\n\n')
     
    
    

if __name__=="__main__":
    sm = SMC('./national_only20_health_2010to2021.csv')
    df = sm.tsne(sm.data, num_tsne=2)
    clu_df = sm.cluster(df, num_clus = 8)
    sm.two_dimension_cluster_visualize(data = clu_df)
    sm.cluster_target_visualize(clu_df)