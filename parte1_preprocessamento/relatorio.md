# Relatório da Parte 1: pré-processamento das imagens

**CardioIA Fase 4 | Cap 1: Visão Computacional na Clínica**

## Objetivo

Preparar raios-X de tórax para treinar redes que diferenciam **cardiomegalia** (coração aumentado) de exames **sem achado**.

## Dataset

Usamos o NIH Chest X-ray, com 112.120 imagens de 30.805 pacientes. Como o conjunto completo tem 45 GB, lemos apenas as 2.234 imagens selecionadas (0,92 GB) de um espelho público no Hugging Face. A cardiomegalia foi escolhida por ser a doença do dataset ligada ao coração, tema do CardioIA.

## Seleção

| Filtro | Justificativa |
|---|---|
| Só exames PA | O AP aumenta o coração na imagem e aparece em 44% dos casos de cardiomegalia, contra 35% dos normais. A rede poderia aprender o tipo de exame. |
| Só adultos (18 a 100 anos) | A proporção do coração na criança é diferente, e há idades inválidas na base. |
| Uma imagem por paciente | Impede que a mesma pessoa esteja em treino e teste, o que inflaria o resultado. |
| Normais que nunca tiveram cardiomegalia | Evita pacientes com os dois rótulos. |
| Mesma quantidade por classe | 1.117 de cada. Com classes iguais, a acurácia não engana. |

## Divisão

Separação de 70 / 15 / 15, mantendo a proporção das classes e com semente fixa:

| Conjunto | Cardiomegalia | Sem achado | Total |
|---|---|---|---|
| Treino | 781 | 782 | 1.563 |
| Validação | 168 | 167 | 335 |
| Teste | 168 | 168 | 336 |

## Pré-processamento

1. **Conversão de formato.** Todas as imagens passam para tons de cinza de 8 bits. Das originais, 11 estavam em RGBA.
2. **Normalização do contraste.** Os tons são esticados para a faixa de 0 a 255, ignorando o 1% mais escuro e o 1% mais claro. Em 200 imagens de treino, a faixa de tons usada passou de 220 ± 27 para 253 ± 1.
3. **Redimensionamento.** De 1024×1024 para 224×224, com filtro Lanczos, o tamanho padrão da VGG16 e da ResNet.
4. **Escala de 0 a 1.** Fica dentro de cada modelo, porque a CNN do zero e as redes pré-treinadas usam escalas diferentes.

## Conferência

O notebook confere a quantidade de arquivos por pasta e que nenhum paciente se repete. Também não encontramos imagens duplicadas entre os arquivos baixados.

## Limitações

- Os rótulos foram extraídos dos laudos por processamento de texto e contêm erros.
- Na base completa, só 2,5% das imagens têm cardiomegalia. Nosso conjunto tem 50%, então a precisão medida não vale para uso real.
- As classes não têm o mesmo perfil: 57% de mulheres e 51 anos em média na cardiomegalia, contra 47% e 46 anos nos normais. O efeito disso é analisado no Ir Além 1.

## Fonte

NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC. Wang et al., *ChestX-ray8*, CVPR 2017.
