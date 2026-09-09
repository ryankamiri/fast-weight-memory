import math

from transformers import Qwen3Config


class FWQwen3Config(Qwen3Config):
    """Qwen config plus serializable fast-weight settings."""

    model_type = "fw_qwen3"

    def __init__(
        self,
        fast_weight_layers: list[int] | None = None,
        teacher_window_size: int = 8192,
        student_window_size: int = 4096,
        max_persistent_tokens: int = 512,
        chunk_size: int = 4096,
        lr: float = 0.3,
        use_projection: bool = True,
        use_conv: bool = True,
        conv_kernel_size: int = 5,
        dynamic_beta: bool = True,
        normalize_student_features: bool = False,
        fast_weight_read_scale: float = 1.0,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        if fast_weight_layers is None:
            fast_weight_layers = [i for i in (0, 7, 14, 21) if i < self.num_hidden_layers]
        if any(type(i) is not int or not 0 <= i < self.num_hidden_layers for i in fast_weight_layers):
            raise ValueError("fast_weight_layers must contain valid zero-based decoder indices")
        if len(set(fast_weight_layers)) != len(fast_weight_layers):
            raise ValueError("fast_weight_layers must not contain duplicates")
        for window in (teacher_window_size, student_window_size):
            if type(window) is not int or window <= 0:
                raise ValueError("Teacher and student windows must be positive integers")
        if student_window_size > teacher_window_size:
            raise ValueError("Student window must not exceed teacher window")
        if type(max_persistent_tokens) is not int or max_persistent_tokens < 0:
            raise ValueError("max_persistent_tokens must be a nonnegative integer")
        self.max_persistent_tokens = max_persistent_tokens
        self.fast_weight_layers = list(fast_weight_layers)
        self.teacher_window_size = teacher_window_size
        self.student_window_size = student_window_size
        self.chunk_size = chunk_size
        self.lr = lr
        self.use_projection = use_projection
        self.use_conv = use_conv
        self.conv_kernel_size = conv_kernel_size
        self.dynamic_beta = dynamic_beta
        self.normalize_student_features = normalize_student_features
        if type(fast_weight_read_scale) not in (int, float) or not math.isfinite(fast_weight_read_scale) or fast_weight_read_scale < 0:
            raise ValueError("fast_weight_read_scale must be a finite nonnegative number")
        self.fast_weight_read_scale = float(fast_weight_read_scale)
