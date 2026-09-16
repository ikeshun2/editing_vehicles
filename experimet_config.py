import torch


class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # training deepsdf
        self.train_test_ratio = 0.8
        self.hidden_dim = 512
        self.latent_code_dim = 128
        self.xyz_dim = 3
        self.xyz_pos_enc_dim = 3
        self.dropout_prob = 0.001
        self.sample_per_scene = 30000
        self.batch_size = 4
        self.clamping_distance = 1.0
        self.latent_code_regularization = 1e-4
        #self.n_epochs = 4002    
        self.n_epochs = 1000
        self.deepsdf_initial_lr = 1e-3
        self.latent_code_inital_lr = 1e-4
        self.warmup_epoch = 3   
        self.surface_threshold = 0.01  # 表面付近とみなすしきい値

        # output training 3d model
        self.train_latent_output_resolution = 1000

        # training cd edit
        self.cd_reg_epoch = 3000
        self.cd_reg_initial_lr = 1e-4
        self.cd_reg_warmup_epoch = 3
        self.cd_walker_epoch = 3000
        self.cd_walker_initial_lr = 1e-4

        self.cd_walker_reg_lambda_ = 1
        self.cd_walker_content_lambda_ = 5
        self.cd_walker_batch_size = 5
        self.cd_walker_warmup_epoch = 3
        self.cd_attribute = [
            "Cd値",
            "Cl値",
        ]

        # training keyword edit
        self.keyword_reg_epoch = 3000
        self.keyword_reg_initial_lr = 1e-4
        self.keyword_reg_warmup_epoch = 3
        # self.keyword_walker_epoch = 3000
        self.keyword_walker_epoch = 1300
        self.keyword_walker_initial_lr = 1e-4

        self.keyword_walker_reg_lambda_ = 1
        self.keyword_walker_content_lambda_ = 100
        self.keyword_walker_batch_size = 5
        self.keyword_walker_warmup_epoch = 3
        # self.keyword_attribute = [
        #     "Voluminous_Smart",
        #     "Powerful_Delicate",
        #     "Linear_Curvy",
        #     "Functional_Decorative",
        #     "Robust_Flexible",
        #     "Calm_Dynamic",
        #     "Realistic_Romantic",
        #     "Elegant_Cute",
        #     "Sophisticated_Youthful",
        #     "Luxurious_Approachable",
        #     "Formal_Everyday",
        #     "Strict_Friendly",
        #     "Uniform_Free",
        #     "Special_Everyday",
        # ]
        self.keyword_attribute = [
            "Cd",
        ]


        # training geometry edit
        self.geometry_reg_epoch = 3000
        self.geometry_reg_initial_lr = 1e-4
        self.geometry_reg_warmup_epoch = 3
        self.geometry_walker_epoch = 3000
        self.geometry_walker_initial_lr = 1e-4

        self.geometry_walker_reg_lambda_ = 1
        self.geometry_walker_content_lambda_ = 5
        self.geometry_walker_batch_size = 5
        self.geometry_walker_warmup_epoch = 3
        self.geometry_attribute = [
            "ルーフ長さ",
            "キャビン長さ",
            "フード長さ",
            "L2",
            "ホイールベース",
            "基調ライン基点長さ",
            "前輪-Lの長さ",
            "フロントオーバーハング",
            "リアオーバーハング",
            "RR-OH/FR-OH",
            "H1",
            "全高",
            "ノーズ高さ",
            "ノーズスラント量",
            "基調ライン基点高さ",
            "ベルトライン高さ",
            "フロントバンパー下端高さ",
            "リアバンパー下端高さ",
            "サイドシル下端高さ",
            "基調ライン角",
            "FRウィンドウ傾斜角",
            "RRウィンドウ傾斜角",
            "全幅",
            "キャビン幅",
            "ルーフ幅",
            "トレッド幅",
            "バンパー下端幅",
            "バンパー上端幅",
            "ルーフ厚み",
            "キャビン厚み",
            "ショルダー厚み",
            "フード厚み",
            "バンパー厚み",
        ]

        # noise data
        self.noise_data = [
            "028_KI_Niro_e_2019",
            "017_FO_Evos_2021",
            "010_CI_C4_Cactus_2015",
            "010_CI_C4_Cactus_2015",
            "056_TO_RAV4_Limited_2018",
            "014_FI_500X_2015",
            "023_HY_Nexo_2019",
            "001_AR_Stelvio_Q4_2017",
            "018_FO_Explorer_ST_2020",
            "002_AR_Tonale_CPT_2019",
            "050_RE_Captur_concept_2020",
            "026_JP_Renegade_Latitude_2014",
            "026_JP_Renegade_Latitude_2014",
            "069_BM_X5_2019",
            "063_VO_XC90_T8_2015",
            "052_TE_Model_X_2016",
            "031_LR_RangeRover_SC_2009",
            "009_CH_Tahoe_RST_2020",
            "088_GC_Yukon_Denali_2021",
            "068_VZ_LadaNiva_Urban_2019",
            "071_BM_iX3_CPT_2018",
            "098_LX_UX_2018",
            "086_GC_Acadia_Denali_2020",
            "099_MR_Levante_2017",
            "113_RE_Arkana_2020",
            "119_TO_bZ4X_2021",
            "037_MZ_CX5_USspec_2012",
        ]
