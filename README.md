# YOLO-censored-xp

## Visão Geral

Este repositório contém um framework para comparar o desempenho de detecção de objetos do YOLOv8 em imagens censuradas versus não censuradas. O experimento treina modelos idênticos em ambos os conjuntos de dados e avalia seu desempenho para determinar como a censura facial afeta as capacidades de detecção.

## Requisitos

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV

Instalação de dependências:

```bash
source env/bin/activate
pip install -r requirements.txt
```

## Dataset

O experimento utiliza o dataset CrowdHuman com duas variantes:

- Uncensored: Imagens originais
- Censored: Imagens com regiões faciais borradas

O dataset deve ser organizado na seguinte estrutura:

```
./crowdface/
├── train_uncensored/     # Imagens de treinamento originais
├── train_censored/       # Imagens de treinamento censuradas
├── validation/           # Imagens de validação
├── annotation_train.odgt # Anotações de treinamento
└── annotation_val.odgt   # Anotações de validação
```

## Uso

Execute o script de treinamento com parâmetros padrão:

```bash
python train.py
```

### Parâmetros de Linha de Comando

| Parameter      | Description                    | Default       |
| -------------- | ------------------------------ | ------------- |
| `--data-path`  | Path to dataset directory      | `./crowdface` |
| `--fraction`   | Dataset fraction to use (0-1)  | `0.01`        |
| `--epochs`     | Number of training epochs      | `10`          |
| `--batch-size` | Batch size for training        | `8`           |
| `--img-size`   | Image size for training        | `640`         |
| `--model`      | YOLOv8 model variant           | `yolov8n.pt`  |
| `--workers`    | Number of data loading workers | `4`           |
| `--verbose`    | Enable detailed logging        | `False`       |

Exemplo para treinamento com o dataset completo:

```bash
python train.py --fraction 1.0 --epochs 50 --batch-size 16
```

## Design do Experimento

O script:

1. Processa anotações do dataset CrowdHuman
2. Cria arquivos de labels compatíveis com YOLO
3. Treina dois modelos YOLOv8 idênticos:
   - Um nas imagens não censuradas
   - Um nas imagens censuradas (idênticas exceto pelo borramento facial)
4. Ambos os modelos usam o mesmo conjunto de validação e parâmetros de treinamento

O parâmetro de fração permite execuções rápidas de prova de conceito em um subconjunto dos dados, facilitando o escalonamento para o conjunto de dados completo quando necessário.

## Saídas

Os resultados do treinamento são salvos em:

```
runs/train/
├── uncensored_frac_*_*/  # Resultados do modelo não censurado
└── censored_frac_*_*/    # Resultados do modelo censurado
```

Cada diretório contém:

- `weights/best.pt`: Melhores pesos do modelo
- Métricas de treinamento e visualizações
- Resultados de validação
