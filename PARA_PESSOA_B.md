# Guia para a pessoa B

A parte A está pronta: dados, CNN do zero e fairness. Aqui está o que você recebe e o que falta.

## O que você recebe

Na [release `parte-a`](../../releases/tag/parte-a):

| Arquivo | O que é | Onde colocar |
|---|---|---|
| `dados_processados.zip` | imagens prontas + `selecao.csv` | descompactar na raiz do repo (cria `dados/processados/`) |
| `cnn_do_zero.keras` | CNN treinada | `modelos/cnn_do_zero.keras` |

Se preferir gerar tudo de novo, rode o notebook da Parte 1 e depois o da Parte 2a.

## Formato dos dados (não mudar, senão a comparação deixa de valer)

```
dados/processados/
├── selecao.csv          arquivo, classe, paciente, sexo, idade, conjunto
├── treino/
│   ├── cardiomegalia/
│   └── sem_achado/
├── validacao/           mesmas duas pastas
└── teste/               mesmas duas pastas
```

- Imagens: PNG, tons de cinza, **224×224**, pixels de **0 a 255**.
- Classes: `sem_achado` = 0, `cardiomegalia` = 1. Passe `class_names=["sem_achado", "cardiomegalia"]`
  ao carregar; sem isso o Keras usa ordem alfabética e inverte os rótulos.
- Limiar de decisão: 0,5.
- O teste só é usado **uma vez**, no fim. Toda escolha sai da validação.

## O que falta

### 1. Transfer Learning (2 pts)
- Carregar com `color_mode="rgb"`: o Keras copia o cinza nos 3 canais que a VGG16 e a ResNet esperam.
- Usar o `preprocess_input` da própria rede. **Não** dividir por 255 antes: ele espera pixels de 0 a 255.
- Não usar espelhamento horizontal no aumento de dados (coloca o coração do lado errado).
- Avaliar com as mesmas métricas de `parte2_cnn_do_zero/metricas.json` e comparar os dois modelos.
- **Rodar os dois no mesmo ambiente.** A CNN deu 0,68 de acurácia no Mac e 0,74 no Colab; comparar um modelo de cada lugar
  seria injusto. Se treinar no Colab, rode lá também o notebook da CNN e use os números dele na comparação.

### 2. Protótipo (2 pts)
Notebook interativo ou Flask que recebe a imagem e mostra classe e confiança. Antes de prever,
a imagem precisa passar pela **mesma** função `preprocessar` da Parte 1:

```python
imagem = preprocessar(Image.open(caminho))                    # cinza, contraste, 224x224
lote = np.asarray(imagem, dtype="float32")[None, ..., None]   # formato (1, 224, 224, 1)
probabilidade_de_cardiomegalia = cnn.predict(lote)[0, 0]      # a divisão por 255 já está no modelo
```

Mostrar na tela o aviso "trabalho acadêmico, não serve para diagnóstico".

Se for Flask: rodar só em `127.0.0.1`, limitar o tamanho do upload e conferir se o arquivo é imagem mesmo.

### 3. Documentação final (1 pt)
- Completar o `README.md` (integrantes, resultados do Transfer Learning, como rodar o protótipo).
- Juntar os entregáveis.

### 4. Antes de entregar
- Conferir se a dupla está no Grupo 39 no FIAP ON.
- Prazo: **04/12/2026, 23h59**. Depois disso, a nota máxima cai para 70%.
- O repo está privado. O enunciado proíbe compartilhar a resposta em grupos. Combinem como o professor
  vai acessar (deixar público na entrega ou enviar o zip).

### Opcional: Ir Além 2
App em React Native ligado ao Flask, com vídeo de até 3 minutos.
