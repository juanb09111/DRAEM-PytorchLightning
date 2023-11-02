from detectron2.config import CfgNode

def add_custom_params(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MAX_EPOCHS= 700
    #Dataset
    cfg.ANOMALY_DATASET = CfgNode()
    cfg.ANOMALY_DATASET.MAX_SAMPLES = 100
    cfg.ANOMALY_DATASET.PREVIEW_DATASET = True
    cfg.ANOMALY_DATASET.SHUFFLE = True
    cfg.ANOMALY_DATASET.MAX_SIZE = 1500
    cfg.ANOMALY_DATASET.MAX_SIZES = (256, 384, 512)
    cfg.ANOMALY_DATASET.ANOMALY_FOLDER = ""  
    cfg.ANOMALY_DATASET.DATASET_PATH = CfgNode()
    cfg.ANOMALY_DATASET.DATASET_PATH.ROOT = "datasets"
    cfg.ANOMALY_DATASET.DATASET_PATH.ANOMALY_SOURCE = ""
    cfg.ANOMALY_DATASET.DATASET_PATH.MVTEC = ""
    
    # Solver
    cfg.SOLVER.NAME = "SGD"
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    cfg.SOLVER.FAST_DEV_RUN = None
    cfg.SOLVER.BASE_LR = 0.0013182567385564075
    cfg.SOLVER.N_EPOCHS = 700
    cfg.SOLVER.INIT_WEIGHTS = True
    # Runner
    cfg.BATCH_SIZE = 2
    cfg.CHECKPOINT_PATH_TRAINING = ""
    cfg.CHECKPOINT_PATH_INFERENCE = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
