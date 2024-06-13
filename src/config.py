from dataclasses import dataclass, asdict, field


@dataclass
class FSeiConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    batch_accumulation: bool = False
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_FSei"
    data_path: str = "data"
    use_reverse_complement: bool = False
    targets_for_reverse_complement: list = field(default_factory=lambda: [1])

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class ConvNetConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    batch_accumulation: bool = False
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_ConvNet"
    data_path: str = "data"
    use_reverse_complement: bool = False
    targets_for_reverse_complement: list = field(default_factory=lambda: [1])

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class AttentionConvConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    batch_accumulation: bool = False
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_ConvNet"
    data_path: str = "data"
    use_reverse_complement: bool = False
    targets_for_reverse_complement: list = field(default_factory=lambda: [1])

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class BassetConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    batch_accumulation: bool = False
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_Basset"
    data_path: str = "data"
    use_reverse_complement: bool = False
    targets_for_reverse_complement: list = field(default_factory=lambda: [1])
    model_architecture: dict = field(default_factory=lambda: {})

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class LRConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    imbalanced_data: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    batch_accumulation: bool = False
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_LR/only_TFs"
    train_data: str = "data/numpy_no_reverse/data_TF_train.npy"
    train_target: str = "data/numpy_no_reverse/target_TF_train.npy"
    valid_data: str = "data/numpy_no_reverse/data_TF_valid.npy"
    valid_target: str = "data/numpy_no_reverse/target_TF_valid.npy"
    test_data: str = "data/numpy_no_reverse/data_TF_test.npy"
    test_target: str = "data/numpy_no_reverse/target_TF_test.npy"

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class LGBMConfig:
    n_estimators: int = (1000,)
    n_iters: int = (1500,)
    early_stopping_round: int = (50,)
    imbalanced_data: bool = (False,)
    results_path: str = "results/results_LR/only_TFs"
    train_data: str = "data/numpy_no_reverse/data_TF_train.npy"
    train_target: str = "data/numpy_no_reverse/target_TF_train.npy"
    valid_data: str = "data/numpy_no_reverse/data_TF_valid.npy"
    valid_target: str = "data/numpy_no_reverse/target_TF_valid.npy"
    test_data: str = "data/numpy_no_reverse/data_TF_test.npy"
    test_target: str = "data/numpy_no_reverse/target_TF_test.npy"

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class DNABertConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    batch_accumulation: bool = False
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_FSei"
    data_path: str = "data"
    use_reverse_complement: bool = False
    targets_for_reverse_complement: list = field(default_factory=lambda: [1])

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
