# data settings
DATASET_ROOT_DIR = '../../data/komonjo/'
DATASET_DIR_LIST = [DATASET_ROOT_DIR + '100241706/',  # 0
                    DATASET_ROOT_DIR + '100249371/',  # 1
                    DATASET_ROOT_DIR + '100249376/',  # 2
                    DATASET_ROOT_DIR + '100249416/',  # 3
                    DATASET_ROOT_DIR + '100249476/',  # 4
                    DATASET_ROOT_DIR + '100249537/',  # 5
                    DATASET_ROOT_DIR + '200003076/',  # 6
                    DATASET_ROOT_DIR + '200003967/',  # 7
                    DATASET_ROOT_DIR + '200004148/',  # 8
                    DATASET_ROOT_DIR + '200005598/',  # 9
                    DATASET_ROOT_DIR + '200006663/',  # 10
                    DATASET_ROOT_DIR + '200014685/',  # 11
                    DATASET_ROOT_DIR + '200014740/',  # 12
                    DATASET_ROOT_DIR + '200015779/',  # 13
                    DATASET_ROOT_DIR + '200021637/',  # 14
                    DATASET_ROOT_DIR + '200021644/',  # 15
                    DATASET_ROOT_DIR + '200021660/',  # 16
                    DATASET_ROOT_DIR + '200021712/',  # 17
                    DATASET_ROOT_DIR + '200021763/',  # 18
                    DATASET_ROOT_DIR + '200021802/',  # 19
                    DATASET_ROOT_DIR + '200021851/',  # 20
                    DATASET_ROOT_DIR + '200021853/',  # 21
                    DATASET_ROOT_DIR + '200021869/',  # 22
                    DATASET_ROOT_DIR + '200021925/',  # 23
                    DATASET_ROOT_DIR + '200022050/',  # 24
                    DATASET_ROOT_DIR + 'brsk00000/',  # 25
                    DATASET_ROOT_DIR + 'hnsd00000/',  # 26
                    DATASET_ROOT_DIR + 'umgy00000/',  # 27
                    ]

# model settings
FE_STRIDE = 32
LAMBDA_RESP = 1.0
LAMBDA_COOR = 2.5
LAMBDA_SIZE = 5.0

# experiment settings
N_KINDS_OF_CHARACTERS = 50
RESIZE_IMAGE_SIZE_FOR_TEST = 512
RESIZE_IMAGE_SIZE_CANDIDATES = [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
RANDOM_SEED = 0
