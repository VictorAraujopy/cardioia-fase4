# Decisões e motivos

## Dados

| Decisão | Motivo |
|---|---|
| NIH Chest X-ray | É o exemplo citado no enunciado, é público e tem idade e sexo, que permitem analisar viés. |
| Cardiomegalia vs. sem achado | É a doença do dataset ligada ao coração, tema do CardioIA. Duas classes deixam o problema simples de explicar. |
| Baixar só 2.234 imagens (0,92 GB) | O dataset inteiro tem 45 GB. O espelho no Hugging Face permite ler imagens avulsas de dentro dos zips, sem login. |
| Só raio-X PA | O AP aumenta o coração na imagem e é mais comum na classe doente (44% contra 35%). A rede poderia aprender o tipo de exame no lugar da doença. |
| Só adultos (18 a 100 anos) | A proporção do coração na criança é diferente, e o dataset tem idades inválidas. |
| Uma imagem por paciente | Ninguém aparece em treino e teste ao mesmo tempo. Sem isso, a métrica sai inflada. |
| Normais só de quem nunca teve cardiomegalia | Há pacientes com os dois rótulos em exames diferentes, o que deixa o rótulo duvidoso. |
| 1.117 imagens de cada classe | São todos os pacientes com cardiomegalia que passam nos filtros; os normais são sorteados. Com classes iguais, a acurácia fica interpretável. |
| 70 / 15 / 15 | Como o dataset é pequeno, a maior parte vai para o treino, mas sobram 336 imagens para o teste. |

## Pré-processamento

| Decisão | Motivo |
|---|---|
| Tons de cinza, 8 bits | Raio-X não tem cor, e 11 das 2.234 originais vieram em RGBA. |
| Normalizar o contraste (ignorando 1% de cada ponta) | Algumas imagens usavam pouco mais da metade dos tons. A faixa usada foi de 220 ± 27 para 253 ± 1 (200 imagens de treino). O 1% ignorado evita que marcas brancas, como a letra "L", definam a escala. |
| 224×224 | É o tamanho padrão da VGG16 e da ResNet, então os dois modelos usam os mesmos dados. |
| Escala 0–1 dentro do modelo | A CNN divide por 255; a VGG16 e a ResNet têm `preprocess_input` próprio. |

## CNN do zero

| Decisão | Motivo |
|---|---|
| 4 blocos (16, 32, 64, 128 filtros) | Com 1.563 imagens de treino, uma rede maior decoraria os dados. |
| BatchNormalization | Deixa o treino mais estável. |
| GlobalAveragePooling no lugar de Flatten | Reduz muito os parâmetros e, com isso, o overfitting. |
| Dropout 0,3 | Mais uma proteção contra overfitting. |
| Aumento de dados leve, sem espelhar | Pequenas rotações e deslocamentos imitam variação de posição. Espelhar colocaria o coração do lado errado. |
| Adam 0,001, reduzido quando estagna | É o padrão do Keras; a redução refina o fim do treino. |
| Parada antecipada (12 épocas sem melhora) | Para no ponto certo e volta para a melhor época (24 de 36). |
| Limiar 0,5 | É o padrão. Ajustá-lo olhando o teste seria trapaça. |
| Semente 42 | Na mesma máquina, as execuções deram exatamente os mesmos números. |
| Keras 3.13 | É a versão do Colab. Um modelo salvo no Keras 3.15 não abre lá. |

### O resultado varia com o ambiente

Os mesmos notebooks, com os mesmos dados, rodados no Colab (GPU T4) em 24/11/2026:

| Ambiente | Acurácia | Precisão | Recall | F1 | AUC | Falso positivo (mulheres / homens) |
|---|---|---|---|---|---|---|
| Mac, CPU (publicado) | 0,679 | 0,708 | 0,607 | 0,654 | 0,779 | 37,5% / 13,6% |
| Colab, GPU | 0,735 | 0,728 | 0,750 | 0,739 | 0,813 | 40,0% / 17,0% |

Publicamos a primeira execução. Trocar pela melhor seria escolher o resultado olhando o teste. Com poucos
dados, essa variação é esperada, e o viés de sexo apareceu nos dois ambientes.

## O que não fizemos, de propósito

- **Ajustar a rede depois de ver o teste.** O resultado publicado é o da primeira configuração.
- **Corrigir o viés de sexo encontrado.** A correção mudaria os dados depois de ver o teste. Está documentada como mitigação proposta no relatório do Ir Além 1.
- **Colocar imagens e modelo no git.** São grandes; ficam na release.
