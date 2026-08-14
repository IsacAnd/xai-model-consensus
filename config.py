"""
Config central do pipeline ADD (Audio Deepfake Detection) + XAI.
Etapa 1 (organização dos dados), Etapa 2 (treinamento) e Etapa 4 (mapas SHAP)
usam este arquivo.

Variáveis de ambiente aceitas (opcionais, sobrescrevem os defaults abaixo):
  ASVSPOOF5_ROOT       - raiz do dataset (real ou o dataset de teste gerado por
                          scripts/generate_toy_dataset.py)
  ADD_EPOCHS           - sobrescreve TrainConfig.epochs (ex.: "5" para testes rápidos)
  ADD_BATCH_SIZE       - sobrescreve TrainConfig.batch_size
  ADD_NUM_WORKERS      - sobrescreve TrainConfig.num_workers (use "0" fora de cluster)
  ADD_DEVICE           - sobrescreve TrainConfig.device ("cpu" ou "cuda")
  ADD_XAI_N_SAMPLES    - sobrescreve XAIConfig.n_shap_samples
  ADD_XAI_N_BACKGROUND - sobrescreve XAIConfig.n_background (Etapa 4)
  ADD_XAI_GRADIENT_NSAMPLES - sobrescreve XAIConfig.gradient_nsamples (Etapa 4)
  ADD_XAI_BATCH_SIZE   - sobrescreve XAIConfig.shap_batch_size (Etapa 4)
  ADD_SIM_EPS          - sobrescreve SimilarityConfig.eps (Etapa 5)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class PathConfig:
    # Raiz onde o dataset foi descompactado/criado
    asvspoof5_root: Path = field(
        default_factory=lambda: _env_path("ASVSPOOF5_ROOT", "toy_data/ASVSpoof5")
    )

    train_audio_dir: Path = None
    dev_audio_dir: Path = None
    eval_audio_dir: Path = None
    train_protocol: Path = None
    dev_protocol: Path = None
    eval_protocol: Path = None

    # Saídas do pipeline
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    manifest_dir: Path = None
    cache_dir: Path = None
    results_dir: Path = None
    checkpoint_dir: Path = None
    log_dir: Path = None

    # Etapa 4: saídas do SHAP (matrizes + heatmaps)
    shap_dir: Path = None
    shap_matrix_dir: Path = None
    shap_heatmap_dir: Path = None
    shap_manifest_path: Path = None

    # Etapa 5: saídas das métricas de comparação entre mapas SHAP
    similarity_dir: Path = None
    similarity_pairwise_path: Path = None
    similarity_summary_path: Path = None
    similarity_consensus_path: Path = None

    def __post_init__(self):
        if self.train_audio_dir is None:
            self.train_audio_dir = self.asvspoof5_root / "flac_T"
        if self.dev_audio_dir is None:
            self.dev_audio_dir = self.asvspoof5_root / "flac_D"
        if self.eval_audio_dir is None:
            self.eval_audio_dir = self.asvspoof5_root / "flac_E"
        if self.train_protocol is None:
            self.train_protocol = self.asvspoof5_root / "ASVspoof5.train.tsv"
        if self.dev_protocol is None:
            self.dev_protocol = self.asvspoof5_root / "ASVspoof5.dev.trial.tsv"
        if self.eval_protocol is None:
            self.eval_protocol = self.asvspoof5_root / "ASVspoof5.eval.trial.tsv"
        if self.manifest_dir is None:
            self.manifest_dir = self.project_root / "data"
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "cache" / "logmel"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "results" / "checkpoints"
        if self.log_dir is None:
            self.log_dir = self.project_root / "results" / "logs"
        if self.shap_dir is None:
            self.shap_dir = self.project_root / "results" / "shap"
        if self.shap_matrix_dir is None:
            self.shap_matrix_dir = self.shap_dir / "matrices"
        if self.shap_heatmap_dir is None:
            self.shap_heatmap_dir = self.shap_dir / "heatmaps"
        if self.shap_manifest_path is None:
            self.shap_manifest_path = self.shap_dir / "shap_manifest.csv"
        if self.similarity_dir is None:
            self.similarity_dir = self.project_root / "results" / "similarity"
        if self.similarity_pairwise_path is None:
            self.similarity_pairwise_path = self.similarity_dir / "pairwise_metrics.csv"
        if self.similarity_summary_path is None:
            self.similarity_summary_path = self.similarity_dir / "summary_by_pair.csv"
        if self.similarity_consensus_path is None:
            self.similarity_consensus_path = self.similarity_dir / "consensus_per_sample.csv"

    def ensure_dirs(self):
        for d in [self.manifest_dir, self.cache_dir, self.results_dir,
                  self.checkpoint_dir, self.log_dir,
                  self.shap_dir, self.shap_matrix_dir, self.shap_heatmap_dir,
                  self.similarity_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    sample_rate: int = 16000          # ASVspoof5 é distribuído em 16 kHz
    crop_seconds: float = 2.0         # random crop padronizado
    n_mels: int = 128                 # mel bins
    win_length_ms: float = 25.0       # janela
    hop_length_ms: float = 10.0       # hop
    fmin: float = 20.0
    fmax: float = None                # None -> sample_rate / 2
    top_db: float = 80.0              # clamp do log-scaling

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.win_length_ms / 1000))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_length_ms / 1000))

    @property
    def n_fft(self) -> int:
        # Próxima potência de 2 >= 2 * win_length. Zero-padding além do
        # win_length (janela continua sendo de 25 ms) dá mais bins de
        # frequência à FFT, evitando filterbanks mel vazios em bins altos
        # quando n_mels é relativamente alto (128) para a taxa de amostragem.
        target = self.win_length * 2
        n = 1
        while n < target:
            n *= 2
        return n

    @property
    def crop_samples(self) -> int:
        return int(round(self.sample_rate * self.crop_seconds))

    @property
    def expected_frames(self) -> int:
        # número de frames temporais aproximado do logmel para crop_seconds
        return self.crop_samples // self.hop_length + 1


@dataclass
class TrainConfig:
    batch_size: int = field(default_factory=lambda: _env_int("ADD_BATCH_SIZE", 32))
    epochs: int = field(default_factory=lambda: _env_int("ADD_EPOCHS", 5))
    lr: float = 3e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    scheduler: str = "plateau"        # ReduceLROnPlateau padronizado p/ todos os modelos
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    early_stopping_patience: int = 8
    early_stopping_metric: str = "eer"      # critério de seleção padronizado
    early_stopping_mode: str = "min"        # eer: menor é melhor
    grad_clip_norm: float = 5.0
    num_workers: int = field(default_factory=lambda: _env_int("ADD_NUM_WORKERS", 0))
    seed: int = 42
    device: str = field(default_factory=lambda: _env_str("ADD_DEVICE", "cuda"))
    label_smoothing: float = 0.0
    n_classes: int = 2                # 0 = bonafide, 1 = spoof
    mixed_precision: bool = True


@dataclass
class XAIConfig:
    n_shap_samples: int = field(default_factory=lambda: _env_int("ADD_XAI_N_SAMPLES", 20))
    seed: int = 123                   # seed separada, só para a amostragem do subset XAI

    # --- Etapa 4: geração dos mapas SHAP ---
    # nº de exemplos do split de treino usados como distribuição de
    # referência ("background") pelo shap.GradientExplainer
    n_background: int = field(default_factory=lambda: _env_int("ADD_XAI_N_BACKGROUND", 50))
    # nº de amostras de ruído usadas internamente pelo GradientExplainer para
    # aproximar o gradiente esperado (expected gradients) por exemplo explicado
    gradient_nsamples: int = field(default_factory=lambda: _env_int("ADD_XAI_GRADIENT_NSAMPLES", 50))
    # batch usado ao rodar os 1000 exemplares do subset pelo explainer
    shap_batch_size: int = field(default_factory=lambda: _env_int("ADD_XAI_BATCH_SIZE", 8))
    # colormap do heatmap sobreposto ao espectrograma
    cmap: str = "inferno"


@dataclass
class SimilarityConfig:
    """Etapa 5: métricas de comparação entre mapas SHAP de arquiteturas diferentes."""

    # epsilon numérico usado para evitar divisão por zero na normalização
    # min-max (cosseno/SSIM/Spearman/euclidiana) e na normalização por soma
    # (distribuição de probabilidade usada na JSD)
    eps: float = field(default_factory=lambda: float(os.environ.get("ADD_SIM_EPS", 1e-12)))

    # data_range esperado pelo SSIM: as matrizes são normalizadas para [0, 1]
    # antes de entrar no SSIM, então o range é sempre 1.0
    ssim_data_range: float = 1.0

    # base do log usada no cálculo da distância de Jensen-Shannon (scipy
    # jensenshannon); base=2 limita a distância ao intervalo [0, 1]
    jsd_log_base: float = 2.0


MODEL_NAMES = ["cnn", "bigru", "efficientnet", "vit", "resnet", "mobilenet", "bilstm", "swin", "ast", "conformer"]

PATHS = PathConfig()
AUDIO = AudioConfig()
TRAIN = TrainConfig()
XAI = XAIConfig()
SIMILARITY = SimilarityConfig()