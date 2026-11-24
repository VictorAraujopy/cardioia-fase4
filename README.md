# CardioIA Fase 4: Visão Computacional na Clínica

Classifica raios-X de tórax em **cardiomegalia** (coração aumentado) ou **sem achado**.

> Trabalho acadêmico. Não serve para diagnóstico.

## Estado do projeto

| Parte | Situação |
|---|---|
| 1. Pré-processamento | pronta |
| 2a. CNN do zero | pronta |
| 2b. Transfer Learning | a fazer |
| Protótipo | a fazer |
| Ir Além 1: fairness | pronta |
| Ir Além 2: app mobile | opcional |

- **Por que cada escolha foi feita:** [DECISOES.md](DECISOES.md)
- **O que falta e como continuar:** [PARA_PESSOA_B.md](PARA_PESSOA_B.md)

## Resultado da CNN do zero (336 imagens de teste)

| Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|
| 0,679 | 0,708 | 0,607 | 0,654 | 0,779 |

Prints em [parte2_cnn_do_zero/prints](parte2_cnn_do_zero/prints).

## Estrutura

```
parte1_preprocessamento/   notebook, relatório e exemplos
parte2_cnn_do_zero/        notebook, métricas e prints
ir_alem1_fairness/         notebook, relatório e tabelas
dados/  modelos/           fora do git (ver release parte-a)
```

Os `.py` são a fonte; os `.ipynb` são gerados a partir deles e já estão executados.

## Como rodar

**No computador** (Python 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd parte1_preprocessamento && python preprocessamento.py && cd ..
cd parte2_cnn_do_zero && python cnn_do_zero.py && cd ..
cd ir_alem1_fairness && python fairness.py && cd ..
```

A Parte 1 baixa cerca de 0,9 GB.

**No Colab:** abra o notebook da Parte 1 e rode tudo. Para abrir só o da Parte 2a, envie antes o
`dados_processados.zip` (release `parte-a`) e rode `!unzip -q dados_processados.zip -d /content`.

## Dados

[NIH Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC), fornecido pelo NIH Clinical Center.
Wang et al. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised
Classification and Localization of Common Thorax Diseases*. CVPR 2017.
