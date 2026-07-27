# XAI MODEL CONSENSUS

Este projeto cobre: organização dos dados do **ASVspoof 5** (Etapa 1),
treinamento padronizado de 6 arquiteturas — CNN, CRNN, EfficientNet-b0, ViT
pequeno, ResNet18, MobileNetV3-Small — para detecção de deepfake de áudio
(Etapa 2), avaliação quantitativa completa no split de eval oficial
(Etapa 3: accuracy, precision, recall, f1, roc-auc, eer), geração dos
mapas SHAP (matrizes + heatmaps) sobre os 1000 exemplares do subset de XAI
(Etapa 4), e as métricas de comparação entre os mapas SHAP das 6
arquiteturas — cosseno, SSIM, Spearman, distância euclidiana e
Jensen-Shannon (Etapa 5). A etapa seguinte (métrica de consenso agregada
combinando as 5 métricas acima) não está implementada aqui.

## Estrutura

```
xai-model-consensus/
├── config.py                 # todos os hiperparâmetros e paths (editar aqui)
├── main.py                   # entry point único (organize / train / evaluate / xai / all)
├── requirements.txt
├── src/
│   ├── data_organization.py  # Etapa 1: parse dos protocolos + manifests + subset XAI
│   ├── audio_processing.py   # random crop (2s) + extração log-mel + cache em disco
│   ├── dataset.py            # torch.utils.data.Dataset + collate_fn
│   ├── metrics.py            # accuracy, precision, recall, f1, roc-auc, EER
│   ├── utils.py              # seed, logging, EarlyStopping
│   ├── train.py              # Etapa 2: loop de treino padronizado
│   ├── evaluate.py           # Etapa 3: avaliação no eval set oficial completo
│   ├── xai.py                # Etapa 4: geração das matrizes e heatmaps SHAP
│   ├── similarity.py         # Etapa 5: métricas de comparação entre mapas SHAP
│   └── models/
│       ├── cnn.py            # CNN (4 blocos conv)
│       ├── crnn.py           # CNN + GRU bidirecional
│       ├── effnet.py         # EfficientNet-b0 (from scratch, 1 canal)
│       ├── vit.py            # ViT pequeno (from scratch, patch embed adaptável)
│       ├── resnet.py         # ResNet18 (from scratch, 1 canal)
│       └── mobilenet.py      # MobileNetV3-Small (from scratch, 1 canal)
├── scripts/
│   ├── run_slurm.sh          # job array (1 por modelo) para o CENAPAD - treino
│   ├── run_slurm_organize.sh # job único para a Etapa 1
│   ├── run_slurm_xai.sh      # job array (1 por modelo) para o CENAPAD - SHAP (Etapa 4)
│   └── run_slurm_similarity.sh # job único para a Etapa 5 (depende de todos os modelos)
├── data/                     # manifests gerados (CSV), incluindo xai_subset_1000.csv
├── cache/logmel/              # cache dos tensores log-mel (train/dev/eval)
└── results/
    ├── checkpoints/          # melhor checkpoint de cada modelo
    ├── logs/                 # logs + histórico de treino (JSON) + métricas de avaliação por modelo
    └── shap/                 # saídas da Etapa 4
        ├── matrices/<modelo>/<file_id>.npy   # matriz SHAP bruta (mesmo shape do log-mel)
        ├── heatmaps/<modelo>/<file_id>.png   # log-mel + heatmap SHAP sobreposto
        └── shap_manifest.csv                 # modelo, file_id, label, pred, paths
    └── similarity/               # saídas da Etapa 5
        ├── pairwise_metrics.csv      # file_id, label, model_a, model_b, cosine, ssim, spearman, euclidean, jsd
        ├── summary_by_pair.csv       # média/desvio/n de cada métrica, agregado por par de modelos
        └── consensus_per_sample.csv  # média das 5 métricas sobre todos os pares, por exemplar
```

## 1. Antes de rodar

Edite `config.py` → `PathConfig` com os caminhos reais de ASVspoof 5. Especificamente:

- `asvspoof5_root`, `train_audio_dir`, `dev_audio_dir`, `eval_audio_dir`
- `train_protocol`, `dev_protocol`, `eval_protocol`

> **Atenção**: o parser de protocolo em `src/data_organization.py` é
> _auto-detectável_ (ele descobre sozinho qual coluna é o `file_id` batendo
> com os arquivos de áudio em disco, e qual coluna é o rótulo
> `bonafide`/`spoof`), porque o layout exato de colunas pode variar entre
> os arquivos de protocolo train/dev/eval do ASVspoof5. Ainda assim, **confira
> o `manifest_*.csv` gerado** antes de treinar (Etapa 1) para garantir que os
> paths e rótulos ficaram corretos — principalmente o `eval`, cujo protocolo
> de trial pode não trazer rótulo público dependendo da fase do challenge que
> você baixou.

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 2. Etapa 1 — organização dos dados

```bash
python main.py --stage organize
```

Isso gera:

- `data/manifest_train.csv`, `data/manifest_dev.csv`, `data/manifest_eval.csv`
- `data/xai_subset_1000.csv` — 1000 exemplares sorteados (seed fixa,
  `config.XAI.seed`) do eval, estratificados por classe, reservados para a
  Etapa 4 (SHAP) e Etapa 5 (métricas de consenso).

## 3. Etapa 2 — treinamento dos modelos

```bash
# um modelo específico
python main.py --stage train --model cnn

# todos os 6 modelos em sequência
python main.py --stage train --model all
```

Hiperparâmetros padronizados entre todos os modelos (`config.py` → `TrainConfig`):

| Item            | Valor                                                                |
| --------------- | -------------------------------------------------------------------- |
| Optimizer       | AdamW                                                                |
| Batch size      | 32                                                                   |
| Epochs (máx.)   | 50                                                                   |
| Learning rate   | 3e-4                                                                 |
| Weight decay    | 1e-2                                                                 |
| Scheduler       | ReduceLROnPlateau (fator 0.5, paciência 3, monitora val_loss)        |
| Early stopping  | paciência 8 épocas, critério = **EER de validação** (menor é melhor) |
| Grad clipping   | norm 5.0                                                             |
| Mixed precision | ligado (se CUDA disponível)                                          |

Processamento de áudio (`config.py` → `AudioConfig`):

| Item        | Valor                                        |
| ----------- | -------------------------------------------- |
| Sample rate | 16 kHz                                       |
| Random crop | 2 s (pad circular se o áudio for mais curto) |
| Mel bins    | 128                                          |
| Janela      | 25 ms                                        |
| Hop         | 10 ms                                        |
| Log scaling | `AmplitudeToDB` (top_db=80)                  |

O log-mel de cada arquivo é cacheado em `cache/logmel/<split>/*.pt` na
primeira vez que é computado (o recorte aleatório usado é fixado por
`idx` do dataset, então o cache é determinístico entre épocas — ajuste
`force_recompute=True` em `ADDDataset` se quiser um crop novo a cada época
como data augmentation).

### Saídas por modelo

- `results/checkpoints/<modelo>_best.pt` — melhor checkpoint (state_dict do
  modelo + optimizer + métricas de validação do momento salvo)
- `results/logs/history_<modelo>.json` — métricas por época (train_loss,
  val_loss, val_eer, val_acc, val_precision, val_recall, val_f1, val_roc_auc, lr)
- `results/logs/train_<modelo>.log` — log textual do treino

## 4. Etapa 3 — avaliação quantitativa (eval set oficial completo)

```bash
# um modelo específico
python main.py --stage evaluate --model cnn

# todos os modelos treinados + tabela comparativa
python main.py --stage evaluate --model all
```

Carrega o melhor checkpoint de cada modelo (salvo na Etapa 2) e roda
inferência sobre o split `eval` oficial completo do ASVspoof5, calculando
accuracy, precision, recall, f1, roc-auc e EER. Gera:

- `results/logs/eval_<modelo>.json` — métricas detalhadas por modelo
- `results/logs/eval_summary.csv` — tabela comparativa entre os 6 modelos,
  ordenada por EER (menor = melhor)

Também é possível avaliar sobre `train` ou `dev` com `--eval-split`, útil
para checar overfitting comparando as métricas entre splits.

> Nota: esta é a avaliação no eval set **completo**, usada para reportar os
> resultados finais dos modelos. O subset de 1000 exemplares
> (`data/xai_subset_1000.csv`, gerado na Etapa 1) é separado e usado a
> partir da Etapa 4.

## 5. Etapa 4 — geração dos mapas SHAP

```bash
# um modelo específico
python main.py --stage xai --model cnn

# todos os 6 modelos
python main.py --stage xai --model all
```

Para cada modelo, carrega o melhor checkpoint (Etapa 2) e roda
`shap.GradientExplainer` sobre os 1000 exemplares de
`data/xai_subset_1000.csv` (Etapa 1), reaproveitando `ADDDataset` e
`collate_fn` de `src/dataset.py` — o mesmo cache de log-mel e o mesmo
padding/crop para `AUDIO.expected_frames` usados no treino/avaliação, sem
nenhuma lógica de áudio duplicada. O `GradientExplainer` foi escolhido por
funcionar de forma uniforme nas 6 arquiteturas (incluindo CRNN, com camada
recorrente, e ViT, com atenção), diferente do `DeepExplainer`, que é mais
sensível a camadas não convolucionais.

Para cada exemplar:

1. A **distribuição de referência** ("background") do explainer é uma
   amostra aleatória (seed fixa, `config.XAI.seed`) de
   `config.XAI.n_background` exemplos do split de **treino**.
2. O SHAP é calculado em relação à **classe predita** pelo modelo naquele
   exemplar (a explicação mais direta da decisão do modelo, não do rótulo
   verdadeiro).
3. A matriz SHAP bruta (mesmo shape do log-mel, `1 × n_mels × frames`) é
   salva em `.npy` — é o insumo para a Etapa 5 (normalização + métricas de
   similaridade entre modelos). Não é normalizada aqui.
4. Um heatmap `.png` (log-mel + `|SHAP|` normalizado só para visualização,
   sobreposto) é salvo para inspeção visual/figuras do artigo.

Saídas:

- `results/shap/matrices/<modelo>/<file_id>.npy`
- `results/shap/heatmaps/<modelo>/<file_id>.png`
- `results/shap/shap_manifest.csv` — `model, file_id, label, pred, matrix_path, heatmap_path`

Parâmetros ajustáveis em `config.py` → `XAIConfig`:

| Item                | Default     | Descrição                                                                                                    |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
| `n_background`      | 50          | tamanho da amostra de referência do explainer                                                                |
| `gradient_nsamples` | 50          | amostras de ruído internas do `GradientExplainer` (expected gradients)                                       |
| `shap_batch_size`   | 8           | batch ao processar os 1000 exemplares (reduza se faltar memória de GPU, especialmente para ViT/EfficientNet) |
| `cmap`              | `"inferno"` | colormap do heatmap sobreposto                                                                               |

> **Custo computacional**: SHAP é bem mais caro que uma inferência simples
> (cada exemplar dispara múltiplos forward+backward internos). Rodar os 6
> modelos × 1000 exemplares pode levar horas dependendo da GPU — use o
> `scripts/run_slurm_xai.sh` (job array, 1 modelo por task) no CENAPAD em
> vez de rodar tudo sequencialmente numa sessão interativa.

## 6. Etapa 5 — métricas de comparação entre mapas SHAP

```bash
python main.py --stage similarity
# equivalente a:
python -m src.similarity
```

Não recebe `--model`: a Etapa 5 sempre olha para **todos** os modelos que já
têm entradas em `results/shap/shap_manifest.csv` (Etapa 4) e compara pares
de modelos — não é preciso ter rodado a Etapa 4 para os 6 de uma vez; ela
compara o que estiver disponível e ignora exemplares em que só 1 modelo
processou aquele `file_id`.

Para cada `file_id` do subset XAI com pelo menos 2 modelos disponíveis,
calcula 5 métricas entre CADA par de modelos (`itertools.combinations`,
até 15 pares para os 6 modelos):

| Métrica        | O que mede                                                          | Normalização usada                             |
| -------------- | ------------------------------------------------------------------- | ---------------------------------------------- |
| Cosseno        | alinhamento global do vetor de importância                          | min-max `[0, 1]` (achatado em vetor)           |
| SSIM           | similaridade estrutural espacial (padrão tempo-frequência)          | min-max `[0, 1]` (mapa 2D)                     |
| Spearman       | correlação de ranking, robusta a diferenças de escala entre modelos | min-max `[0, 1]` (achatado em vetor)           |
| Euclidiana     | magnitude ponto a ponto da diferença entre os mapas                 | min-max `[0, 1]` (achatado em vetor)           |
| Jensen-Shannon | diferença entre as distribuições de importância                     | soma = 1 (tratado como massa de probabilidade) |

Em todos os casos a comparação é feita sobre `|SHAP|` (magnitude), não o
valor com sinal — o sinal indica se o pixel empurrou a decisão a
favor/contra a classe predita, algo que não é diretamente comparável entre
arquiteturas distintas; o que se compara aqui é _onde_ cada modelo colocou
importância. A distância de Jensen-Shannon retornada pelo `scipy` é elevada
ao quadrado para reportar a divergência (não a distância/raiz).

Saídas:

- `results/similarity/pairwise_metrics.csv` — uma linha por
  `(file_id, model_a, model_b)`, útil para inspecionar exemplares
  específicos ou plotar distribuições por par.
- `results/similarity/summary_by_pair.csv` — média/desvio-padrão/n de cada
  métrica agregados por par de modelos (a "matriz" de consenso
  cross-architecture central da Etapa 5 do artigo).
- `results/similarity/consensus_per_sample.csv` — média das 5 métricas
  sobre todos os pares, por exemplar; serve tanto para identificar quais
  exemplares geram mais divergência entre arquiteturas quanto como insumo
  cru para uma eventual Etapa 6 (métrica de consenso agregada).

> **Dependências novas**: esta etapa usa `scipy` (Spearman, Jensen-Shannon)
> e `scikit-image` (SSIM), já incluídas em `requirements.txt`.

## 7. Rodando o pipeline completo de uma vez

```bash
python main.py --stage all --model all
```

Roda organização → treino dos 6 modelos → avaliação dos 6 modelos → mapas
SHAP dos 6 modelos → métricas de comparação (Etapa 5), em sequência. Dado o
custo da Etapa 4 (ver nota acima), para rodadas exploratórias pode ser mais
prático rodar `--stage all` só até a avaliação e disparar `--stage xai` e
`--stage similarity` separadamente via Slurm.

## Decisões de design (para revisar/ajustar conforme a necessidade do artigo)

- **Modelos "from scratch"**: EfficientNet, ResNet e MobileNet usam as
  arquiteturas do `torchvision`, mas com pesos aleatórios (sem pré-treino
  ImageNet) e a primeira camada conv adaptada para 1 canal — isso mantém a
  comparação justa entre as 6 arquiteturas treinadas apenas com os dados de
  áudio, e evita problemas de compor pré-treino de imagem natural com
  espectrogramas.
- **ViT customizado**: não uso `torchvision.models.vit_*` porque essas
  implementações fixam input em 224×224 (por causa dos positional
  embeddings), o que forçaria redimensionar/distorcer o espectrograma. O ViT
  aqui usa patch embedding via `Conv2d`, então funciona nativamente com o
  formato (128 mels × ~201 frames) e tem positional embedding interpolável.
- **CRNN**: reaproveita os mesmos blocos convolucionais da CNN simples (3
  blocos em vez de 4, para preservar mais resolução temporal) e adiciona uma
  GRU bidirecional sobre o eixo do tempo, com pooling médio antes do
  classificador.
- **Critério de seleção = EER de validação**: é a métrica padrão em
  anti-spoofing/ADD (mais informativa que accuracy sob desbalanceamento de
  classes, que é o caso do ASVspoof5 — veja proporções bonafide/spoof no
  paper oficial).
- **Cache de log-mel em disco**: evita recomputar FFT/mel a cada época,
  importante dado o volume do ASVspoof5 (centenas de milhares de
  utterances). Trade-off: como o crop é fixado no cache, o "random crop" na
  prática vira fixo após a primeira época — mencionei acima como reativar
  crop novo por época se for importante para o experimento.
- **GradientExplainer em vez de DeepExplainer/KernelExplainer (Etapa 4)**:
  `GradientExplainer` funciona de forma consistente em arquiteturas
  heterogêneas (conv puro, conv+recorrente, atenção), o que é essencial
  para comparar mapas SHAP _entre_ as 6 arquiteturas na Etapa 5 —
  `KernelExplainer` seria proibitivo em custo para espectrogramas
  (128 × ~201 features), e `DeepExplainer` tem suporte mais frágil a
  camadas recorrentes/atenção.
- **SHAP relativo à classe predita, não ao rótulo verdadeiro**: garante que
  o mapa explique _por que o modelo decidiu o que decidiu_, inclusive nos
  casos de erro — o que é o dado relevante para comparar o comportamento
  interno dos modelos entre si na Etapa 5.
- **Duas normalizações diferentes na Etapa 5**: min-max `[0, 1]` para
  cosseno/SSIM/Spearman/euclidiana (preserva o padrão espacial relativo, só
  equaliza a escala entre arquiteturas) e soma-1 só para a JSD (que exige,
  por definição, duas distribuições de probabilidade). Comparar direto a
  matriz SHAP bruta faria as métricas de escala (cosseno, euclidiana)
  ficarem dominadas pela diferença de magnitude de gradiente entre
  arquiteturas, em vez da diferença de padrão espacial de fato.
- **Comparação sobre `|SHAP|`, não sobre o valor com sinal**: o sinal indica
  se aquele pixel empurrou a decisão a favor ou contra a classe predita,
  mas essa convenção de sinal não é diretamente comparável entre
  arquiteturas heterogêneas; o que importa para a Etapa 5 é _onde_ cada
  modelo colocou importância.
- **Pareamento pelo `shap_manifest.csv`, não por índice de arquivo**: a
  Etapa 5 casa os mapas pelo `file_id` presente no manifest da Etapa 4 (não
  assume que os 6 modelos processaram os mesmos exemplares na mesma ordem),
  o que permite rodar a Etapa 4 de forma incremental (ex.: 1 modelo por vez
  via job array) sem quebrar a Etapa 5.

## O que falta (próximas etapas, não implementadas aqui)

- Etapa 6 (métrica de consenso agregada combinando cosseno, SSIM, Spearman,
  euclidiana e JSD — opcional, ver `results/similarity/consensus_per_sample.csv`
  como ponto de partida)
- Etapa 7 (conclusões)
