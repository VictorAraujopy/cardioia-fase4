# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CardioIA Fase 4 | Parte 2a: CNN treinada do zero
#
# Classifica raios-X em **cardiomegalia** (1) ou **sem achado** (0).
#
# Precisa das imagens geradas pela Parte 1 em `dados/processados/`.
#
# > Trabalho acadêmico. Não serve para diagnóstico.

# %%
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEMENTE = 42
TAMANHO_DA_IMAGEM = (224, 224)
IMAGENS_POR_LOTE = 32
MAXIMO_DE_EPOCAS = 60
LIMIAR_DE_DECISAO = 0.5
CLASSES = ["sem_achado", "cardiomegalia"]

esta_no_colab = Path("/content").exists()
PASTA_DADOS = Path("/content/dados") if esta_no_colab else Path("../dados")
PASTA_MODELOS = Path("/content/modelos") if esta_no_colab else Path("../modelos")
PASTA_PROCESSADOS = PASTA_DADOS / "processados"
PASTA_PRINTS = Path("prints")
assert PASTA_PROCESSADOS.exists(), "Rode a Parte 1 ou descompacte dados_processados.zip (release parte-a)"
PASTA_MODELOS.mkdir(exist_ok=True)
PASTA_PRINTS.mkdir(exist_ok=True)

keras.utils.set_random_seed(SEMENTE)

# %% [markdown]
# ## 1. Carregar as imagens


# %%
def carregar(conjunto, embaralhar):
    return keras.utils.image_dataset_from_directory(
        PASTA_PROCESSADOS / conjunto,
        class_names=CLASSES,
        label_mode="binary",
        color_mode="grayscale",
        image_size=TAMANHO_DA_IMAGEM,
        batch_size=IMAGENS_POR_LOTE,
        shuffle=embaralhar,
        seed=SEMENTE,
    )


treino = carregar("treino", embaralhar=True)
validacao = carregar("validacao", embaralhar=False)
teste = carregar("teste", embaralhar=False)
arquivos_de_teste = [Path(caminho).name for caminho in teste.file_paths]

# %% [markdown]
# ## 2. Montar a rede
#
# Quatro blocos de convolução: cada um dobra os filtros e corta a imagem pela metade.
#
# O aumento de dados é leve e **sem espelhamento horizontal**: espelhar colocaria o coração do lado
# errado do peito.

# %%
aumento_de_dados = keras.Sequential(
    [
        layers.RandomRotation(factor=0.02),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomZoom(height_factor=0.05),
    ],
    name="aumento_de_dados",
)


def bloco_de_convolucao(entrada, quantidade_de_filtros):
    saida = layers.Conv2D(quantidade_de_filtros, kernel_size=3, padding="same", use_bias=False)(entrada)
    saida = layers.BatchNormalization()(saida)
    saida = layers.ReLU()(saida)
    return layers.MaxPooling2D()(saida)


entrada = keras.Input(shape=(*TAMANHO_DA_IMAGEM, 1))
saida = aumento_de_dados(entrada)
saida = layers.Rescaling(1 / 255)(saida)
for quantidade_de_filtros in [16, 32, 64, 128]:
    saida = bloco_de_convolucao(saida, quantidade_de_filtros)
saida = layers.GlobalAveragePooling2D()(saida)
saida = layers.Dropout(0.3)(saida)
saida = layers.Dense(1, activation="sigmoid", name="probabilidade_de_cardiomegalia")(saida)

cnn = keras.Model(entrada, saida, name="cnn_do_zero")
cnn.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
cnn.summary()

# %% [markdown]
# ## 3. Treinar
#
# O treino para quando a validação deixa de melhorar e volta para a melhor época.
# O conjunto de teste não participa de nenhuma escolha.

# %%
parar_quando_estagnar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
reduzir_passo = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)

historico = cnn.fit(
    treino,
    validation_data=validacao,
    epochs=MAXIMO_DE_EPOCAS,
    callbacks=[parar_quando_estagnar, reduzir_passo],
    verbose=2,
)
cnn.save(PASTA_MODELOS / "cnn_do_zero.keras")

# %%
figura, (eixo_perda, eixo_acuracia) = plt.subplots(1, 2, figsize=(11, 4))
eixo_perda.plot(historico.history["loss"], label="treino")
eixo_perda.plot(historico.history["val_loss"], label="validação")
eixo_perda.set(title="Perda", xlabel="época")
eixo_perda.legend()
eixo_acuracia.plot(historico.history["accuracy"], label="treino")
eixo_acuracia.plot(historico.history["val_accuracy"], label="validação")
eixo_acuracia.set(title="Acurácia", xlabel="época")
eixo_acuracia.legend()
figura.tight_layout()
figura.savefig(PASTA_PRINTS / "curvas_de_treino.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Avaliar no teste


# %%
def respostas_verdadeiras(dados):
    return np.concatenate([rotulos.numpy() for _, rotulos in dados]).ravel().astype(int)


verdadeiras = respostas_verdadeiras(teste)
probabilidades = cnn.predict(teste, verbose=0).ravel()
previstas = (probabilidades >= LIMIAR_DE_DECISAO).astype(int)

metricas = {
    "imagens_de_teste": len(verdadeiras),
    "acuracia": accuracy_score(verdadeiras, previstas),
    "precisao": precision_score(verdadeiras, previstas),
    "recall": recall_score(verdadeiras, previstas),
    "f1": f1_score(verdadeiras, previstas),
    "auc": roc_auc_score(verdadeiras, probabilidades),
}
metricas = {nome: round(float(valor), 3) for nome, valor in metricas.items()}
Path("metricas.json").write_text(json.dumps(metricas, indent=2) + "\n")
print(pd.Series(metricas).to_string())

# %%
matriz = confusion_matrix(verdadeiras, previstas)
grafico = ConfusionMatrixDisplay(matriz, display_labels=["sem achado", "cardiomegalia"])
grafico.plot(cmap="Blues", colorbar=False)
grafico.ax_.set(title="Matriz de confusão (teste)", xlabel="previsto", ylabel="verdadeiro")
grafico.figure_.tight_layout()
grafico.figure_.savefig(PASTA_PRINTS / "matriz_de_confusao.png", dpi=120)
plt.show()

# %%
metricas_para_o_grafico = {nome: metricas[nome] for nome in ["acuracia", "precisao", "recall", "f1", "auc"]}
figura, eixo = plt.subplots(figsize=(7, 3.8))
barras = eixo.bar(list(metricas_para_o_grafico.keys()), list(metricas_para_o_grafico.values()), color="#2b6cb0")
eixo.bar_label(barras, fmt="%.2f")
eixo.axhline(0.5, color="gray", linestyle="--", linewidth=1)
eixo.set(title="Métricas da CNN do zero (teste)", ylim=(0, 1))
figura.tight_layout()
figura.savefig(PASTA_PRINTS / "metricas.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Guardar as previsões para a análise de fairness

# %%
previsoes = pd.DataFrame(
    {
        "arquivo": arquivos_de_teste,
        "verdadeira": verdadeiras,
        "prevista": previstas,
        "probabilidade": probabilidades,
    }
)
previsoes.to_csv(PASTA_DADOS / "previsoes_teste.csv", index=False)
