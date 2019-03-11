# data settings
DATASET_ROOT_DIR = '../../data/komonjo/'
DATASET_DIR_LIST = [DATASET_ROOT_DIR + '100241706/',
                    DATASET_ROOT_DIR + '100249371/',
                    DATASET_ROOT_DIR + '100249376/',
                    DATASET_ROOT_DIR + '100249416/',
                    DATASET_ROOT_DIR + '100249476/',
                    DATASET_ROOT_DIR + '100249537/',
                    DATASET_ROOT_DIR + '200003076/',
                    DATASET_ROOT_DIR + '200003967/',
                    DATASET_ROOT_DIR + '200004148/',
                    DATASET_ROOT_DIR + '200005598/',
                    DATASET_ROOT_DIR + '200006663/',
                    DATASET_ROOT_DIR + '200014685/',
                    DATASET_ROOT_DIR + '200014740/',
                    DATASET_ROOT_DIR + '200015779/',
                    DATASET_ROOT_DIR + '200021637/',
                    DATASET_ROOT_DIR + '200021644/',
                    DATASET_ROOT_DIR + '200021660/',
                    DATASET_ROOT_DIR + '200021712/',
                    DATASET_ROOT_DIR + '200021763/',
                    DATASET_ROOT_DIR + '200021802/',
                    DATASET_ROOT_DIR + '200021851/',
                    DATASET_ROOT_DIR + '200021853/',
                    DATASET_ROOT_DIR + '200021869/',
                    DATASET_ROOT_DIR + '200021925/',
                    DATASET_ROOT_DIR + '200022050/',
                    DATASET_ROOT_DIR + 'brsk00000/',
                    DATASET_ROOT_DIR + 'hnsd00000/',
                    DATASET_ROOT_DIR + 'umgy00000/'
                    ]


N_KINDS_OF_CHARACTERS = 10
RESIZE_IMAGE_SIZE = 512
RESIZE_IMAGE_SIZE_CANDIDATES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704]

# model settings
FE_STRIDE = 32
LAMBDA_RESP = 1.0
LAMBDA_COOR = 5.0
LAMBDA_SIZE = 2.5

# experiment settings
RANDOM_SEED = 0
