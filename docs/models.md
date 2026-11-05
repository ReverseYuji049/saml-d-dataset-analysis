
# MODELOS APLICADOS

## 1. Dummy Classifier
O [Dummy Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) é um classificador simples que utiliza estratégias básicas para realizar previsões, sem aprender relações relevantes entre as **características (*features*)** dos dados.

Ela serve como uma referência para avaliar se os modelos mais complexos estão realmente aprendendo algo.

### 1.1. Como funciona?
O `Dummy Classifier` possui diferentes estratégias, definidas pelo parâmetro `strategy`.
- `most_frequent`: Sempre prevê a classe mais frequente.
- `prior`: Prevê a classe mais frequente e utiliza as frequências observadas para calcular possibilidades. 
- `stratified`: Realiza previsões aleatórias considerando a distribuição das classes no treinamento.
- `uniform`: Escolhe as classes aleatoriamente, com probabilidades iguais.
- `constant`: Sempre prevê uma classe específica definida pelo usuário.

### 1.2. Quando utilizar?
O `DummyClassifier` é recomendado para quando se deseja estabelecer uma **referência de desempenho**.

Por exemplo, se ambos os modelos de `RandomForest` e o `Dummy Classifier` conseguirem 99% de acurácia, pode ser que o modelo mais complexo não esteja aprendendo padrões úteis.

A ideia é que os modelos desenvolvidos posteriormente consigam **superar a base de referência (*baseline*)** significativamente, especialmente nas métricas relacionadas à identificação de fraudes.

### 1.3. Caso de Exemplo: Dataset Bancário
- 99% de transações legítimas.
- 1% de transações fraudulentas

Um **Dummy Classifier** que sempre prevê a classe majoritária poderia responder que todas as transações são legítimas, conseguindo uma acurácia 99%, mesmo sem identificar uma única fraude. Isso comprova que a acurácia isolada pode ser enganosa em datasets desbalanceados.

O modelo não tenta descobrir padrões que expliquem por que determinada transação é fraudulenta ou legítima.

Com a estratégia `most_frequent`, se o treinamento aprensetar 99% de transações legítimas, todas as previsões serão de transações legítimas.

### 2. Logistic Regression
O `Logistic Regression` é utilizado para **problemas de classificação**, com o objetivo de **estimar a possibilidade de uma observação pertencer a determinada classe**.

### 2.1. Como funciona?
A `Logistic Regression` trabalha com uma **combinação linear das características de entrada**:

$$ z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b $$

- *x*: características dos dados (*features*)
- *w*: pesos aprendidos durante o treinamento.
- *b*: bias
- *z*: resultado da combinação linear

O resultado da equação pode assumir qualquer valor real, positivo ou negativo. Precisamos transformar esse resultado em uma probabilidade entre 0 e 1. Para isso, a `Logistic Regression` utiliza a **função sigmoide** (*sigmoid*).

$$ sigma(z)=\frac{1}{1+e^{-z}} $$

![1793712748783](image/SAML-D/1793712748783.png)

**Exemplo:**

| Resultado de z | Probabilidade estimada |
| :--: | :--: |
| -4 | 1.8% |
| -2 | 11,9% |
| 0 | 50% |
| 2 | 88,1% |
| 4 | 98,2% |

O modelo utiliza essas probabilidades para decidir a classificação.

### 2.2. Como ocorre o aprendizado?

Durante o treinamento, o algoritmo **ajusta os pesos e o intercepto (*bias*)** para aproximar suas previsões dos resultados reais.

Ele utiliza uma **função de otimização** baseada na *log loss* (perda logarítmica ou entropia cruzada binária no caso de duas classes).

Ou seja, o modelo busca **encontrar pesos que produzam probabilidades coerentes com os exemplos de treinamento**.

### 2.3. Principais parâmetros

1 - `C`: controla a intensidade da regularização inversamente:
- `c`: regularização mais forte
- `C`: regularização mais fraca

Ajuda a controlar a **complexidade do modelo** e a **reduzir o risco de *overfitting***.

2 - `penalty`: define o tipo de regularização.

3 - `solver`: define o algoritmo utilizado para otimização e para encontar os melhores parâmetros.

4 - `max_iter`: número máximo de iterações do otimizador. 

5 - `class_weight`: ajusta o peso atribuído às classes.

- O modo `balanced` calcula os pesos inversamente proporcionais à frequência das classes no conjunto de treinamento. Isso faz com que a classe minoritária tenha maior influência no treinamento.

6 - `random_state`: controla a aleatoriedade quando aplicável.

7 - `fit_intercept`: define se o modelo utilizará intercepto (*bias*).

### 2.4. Quando utilizar?
A `Logistic Regression` é recomendada para quando queremos:
- Trabalhar com classificação binária ou multiclasse.

- Estimar possibilidades de pertencimento às classes.

- Ter um modelo relativamente simples e interpretável.

- Estabelecer uma comparação com algoritmos mais complexos.

**Observações:**
- A Logistic Regression é um modelo linear, então pode apresentar dificuldades quando a separação entre as classes depende de relações muito complexas e não lineares. 

- É recomendado padronizar as características numéricas, principalmente quando possuem escalas muito diferentes.


## 3. RandomForestClassifier
O [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) é um algoritmo de classificação baseado em um conjunto de **árvores de decisão**. 

Pertence à uma categoria de algoritmos **Ensemble Learning**, que consiste na **combinação de múltiplos modelos para produzir uma previsão conjunta**. 

Em vez de depender de apenas uma árvore de decisão, o `Random Forest` constrói **várias árvores e combina suas previsões**.

**Exemplo: 100 árvores de decisão**

- 70 árvores classificam uma transação como legítima.

- 30 árvores classificam como fraudulenta.

A previsão final será legítima, pois essa classe recebeu a maioria dos votos. Esse processo é conhecido como **votação majoritária** (*majority voting*).

### 3.1. Como funciona?

**1 - Árvore de Decisão**

Uma **árvore de decisão** utiliza perguntas e condições para classificar os dados.

**Exemplo: Fraude Bancária**

![](https://media.licdn.com/dms/image/v2/D4E12AQEHpznLbz8nVg/article-cover_image-shrink_720_1280/B4EZkk6QsUKcAI-/0/1757260886778?e=2147483647&v=beta&t=Bb8VHF4faouSPD7QsLI31ErszHXLpUrL4aqigNFpLnI)
Fonte da imagem: https://www.linkedin.com/pulse/decision-tree-vs-random-forest-choosing-right-model-financial-lively-nstse

As perguntas poderiam ser:

1. O valor da transação é elevado?
2. A operação ocorreu em um horário incomum?
3. A transação é internacional?
4. O comportamento é diferente do histórico do cliente?

Dependendo das respostas, a árvore percorre diferentes caminhos até chegar a uma classificação. Cada divisão representa uma condição utilizada para separar os dados.

**3.1.1 - Construção da Random Forest**

O `Random Forest` utiliza diversas árvores de decisão, construídas com duas características fundamentais:

`Bootstrap Sampling`: 
    
- Cada árvore recebe uma amostra dos dados de treinamento, selecionada aleatoriamente com reposição.

- Um mesmo registro pode aparecer mais de uma vez na amostra de uma árvore, enquanto outras não.

`Random Feature Selection`:

- Durante a construção das árvores, o algoritmo seleciona aleatoriamente um conjunto de características para avaliar as divisões dos nós.

- Isso introduz diversidade entre as árvores, não precisando aprender exatamente os mesmos padrões.

**3.1.2 - Votação Final**

Depois da construção das árvores de decisão, cada uma realiza sua própria previsão. O `Random Forest` combina os resultados e escolhe a classe com maior votação:

![1793716079391](image/SAML-D/1793716079391.png)

Fonte da imagem: https://www.researchgate.net/figure/Typical-Random-Forest-classification-through-majority-voting-20_fig4_374675439

Além da classificação, o Random Forest pode estimar probabilidade utilizando a média das probabilidades previstas pelas árvores individuais.

### 3.2. Principais parâmetros

- `n_estimators`: quantidade de árvores na floresta.

- `criterion`: critério utilizado para avaliar as decisões.

    - `gini`: impureza de Gini
    - `entropy`: entropia 
    - `log_loss`: perda logarítmica

    A impureza de Gini e a entropia ajudam a medir o quanto as classes estão sendo misturadas em determinado nó.

- `max_depth`: profundidade máxima das árvores, o quanto elas podem crescer (no máximo 5).

- `min_samples_split`: número mínimo de amostras necessário para dividir um nó (pelo menos 10).

- `min_samples_leaf`: número mínimo de amostras permitido em uma folha.

    - A configuração padrão atual para classificação é `max_features="sqrt"`.
    - Se tivermos 100 features, cada divisão terá 10 features candidatas (raiz quadrada de 100).

- `max_features`: quantidade de características consideradas em cada decisão.

- `bootstrap`: define se serão utilizados amostras boostrap.

- `class_weight`: pesos das classes.

- `random_state`: controla a aleatoriedade.

- `n_jobs`: quantidade de processos utilizados em paralelo.

### 3.3. Quando utilizar?

O `Random Forest` é interessante nos seguintes casos:

- Existem relações complexas entre as features.
- Os dados possuem relações não lineares.
- Queremos combinar vários modelos de árvore de decisão.
- Queremos analisar a importância das características.
- Precisamos de um modelo relativamente robusto para classificação.

Diferente da `Logistic Regression`, não necessita de padronização de features numéricas.

## 4. Lighting Gradiente Boosting Machine (LightGBM)

O [LightGBM](https://lightgbm.readthedocs.io/en/stable/) é um **framework de Machien Learning** desenvolvido pela **Microsoft** que utiliza **algoritmos de Gradient Boosting** baseado em `Gradient Boosting Decision Tress (GBDT)`.

Ele utiliza conjuntos de árvores de decisão para resolver problemas de **classificação**, **regressão** e outras tarefas de **aprendizado supervisionado**. Seu principal diferencial é a maneira **como constrói e combina árvores**, buscando melhorar progressivamente as previsões.

Para entender o `LightGBM`, precisamos conhecer dois conceitos: `Decision Tree` (Árvore de Decisão) e `Gradient Boosting` (Impulsionamento por Gradiente).

### 4.1. Como funciona o Gradient Boosting?

A ideia principal do `Gradient Boosting` é combinar vários modelos de aprendizado, geralmente **árvores de decisão**, para construir um modelo preditivo mais preciso. 

Esses modelos são **treinados sequencialmente**, e não de maneira independente, fazendo com que cada novo modelo tente corrigir os erros cometidos pelos modelos anteriores.

**1. Modelo Inicial:** Criação de um modelo simples que realiza previsões iniciais.

**2. Cálculo dos Erros:** Depois das previsões iniciais, o algoritmo compara os valores previstos com os valores reais.

**3. Treinamento de um Novo Modelo:**  O algoritmo treina uma nova árvore de decisão para aprender os padrões dos erros anteriores. Esse processo é chamado de **aprendizado dos resíduos** (*residuals*), no caso de uma **função de perda quadrática**.

**4. Atualização das Previsões:** O novo modelo adiciona suas correções às previsões anteriores:

$$ F_m(x)=F_{m-1}(x)+\eta h_m(x) $$

- $F_m(x)$: previsão atualizada.
- $F_{m-1}(x)$: previsão anterior.
- $h_m(x)$: correção aprendida pela nova árvore.
- $\eta$: taxa de aprendizado (*Learning Rate*).

**5. Repetição do Processo:** O algoritmo repete as etapas de calcular os erros, treinar novas árvores e atualizar as previsões. Ao final, todas as árvores são combinadas para produzir uma previsão.

### 4.2. O que diferencia o LightGBM de outros Gradient Boosting?
O `LightGBM` foi desenvolvido para oferecer eficiência computacional e escalabilidade, especialmente em datasets grandes.

**1. `Leaf-wise Growth`:** A estratégia *`leaf-wise`* faz com que o algoritmo procura dividir a folha que apresenta a maior redução da função de perda.

![](https://miro.medium.com/1*M2ujrKHCTHwOHa0GPHCZIQ.png)
Fonte da imagem: https://medium.com/@origoldbsc/understanding-lightgbm-a-friendly-guide-9fe08db5effa

Essa estratégia pode produzir árvores mais eficientes, mas também pode aumentar o risco de *overfitting* quando não controlamos sua complexidade. Por isso, os parâmetros como `num_leaves` e `max_depth` são importantes.

**2. `Histogram-based Learning`:** 

- Utiliza-se **histogramas** para agrupar os valores das características em intervalos (*bins*). 

- Em vez de avaliar todas as possíveis decisões considerando cada valor individual, ele usa os agrupamentos para **encontrar divisões eficientes**.

- Isso reduz o custo computacional e o consumo de memória em muitos cenários.

**3 - `GOSS e EFB`:** 
- `Gradient-based One-Side Sampling (GOSS):` Prioriza amostras com gradientes maiores, preservando também uma parcela das amostras com gradientes menores. A intenção é reduzir o volume de dados utilizado para encontrar divisões sem perder informações importantes.

- `Exclusive Feature Bundling (EFB):` Combina características que raramente apresentam valores diferentes de zero simultaneamente, reduzindo a quantidade de características efetivamente processadas. 

### 4.3. Principais Parâmetros

- `n_estimators`: quantidade de árvores construídas durante o processo de `Gradient Boosting`.

    - **Exemplo:** `LGBMClassifier(n_estimators=100)` -> 100 iterações de boosting.
    - Aumentar o número de árvores pode melhorar o aprendizado, mas também pode aumentar o tempo de treinamento e o risco de *overfitting*.

- `learning_rate`: controla a contribuição de cada árvore.
    
    - **Exemplo:** `LGBMClassifier(learning_rate=0.1)`.
    - Um valor menor faz com que cada árvore tenha uma contribuição mais gradual.

- `num_leaves`: número máximo de folhas por árvore.

    - **Exemplo:** `LGBMClassifier(num_leaves=31)`.
    - Quanto maior a quantidade, maior pode ser a capacidade de representar padrões complexos, mas também pode provocar *overfitting*.

- `max_depth`: profundidade máxima das árvores.

- `objective`: objetivo do aprendizado.

    - **Exemplos:** `binary`, `multiclass`, `regression`, `cross_entropy`.

- `max_bin`: quantidade máxima de intervalos utilizados nos histogramas.

- `min_child_samples`: número mínimo de amostras em uma folha. Ajuda a evitar que o algoritmo crie folhas excessivamente específicas, reduzindo o risco de *overfitting*.

- `subsample`: fração de amostras utilizadas em cada etapa de **bagging**.

- `colsample_bytree`: fração de características consideradas por árvore.

- `reg_alpha`: regularização L1.

- `reg_lambda`: regularização L2.

- `random_state`: controla a aleatoriedade.

- `n_jobs`: controla a quantidade de threads utilizadas. 

- *Desbalanceamento* - `scale_pos_weight` e `is_unbalance`: lida com problemas binários desbalanecados.

    - `scale_pos_weight`: permite aumentar o peso da classe positiva, normalmente representada por 1.

    $$scale\_pos\_weight=\frac{N_{\text{negativos}}}{N_{\text{positivos}}}$$

### 4.4. Quando utilizar?

O `LightGBM` é recomendado para quando:

- Trabalhamos com datasets tabulares.
- Existem relações não lineares entre as características.
- Existem interações complexas entre as features.
- Precisamos de um algoritmo eficiente para grandes volumes de dados.
- Queremos explorar diferentes configurações de hiperparâmetros.
- Precisamos de um modelo competitivo para classificação de fraudes.

### 4.5. Importação

**Instalando a biblioteca:**

    pip install lightgbm

**Importando o classificador:**

    from lightgbm import LGBMClassifier

## 5. Comparação entre os Modelos

| Característica | DummyClassifier | LogisticRegression | RandomForest | LightGBM |
| :--: | :--: | :--: | :--: | :--: |
| Aprende padrões? | Não | Sim | Sim | Sim |
| Tipo | Baseline | Modelo Linear | Ensemble | Ensemble
| Estrutura | Regras simples | Equação Linear | Árvores independentes | Árvores sequenciais |
| Relações não lineares | Não | Não diretamente | Sim | Sim |
| Estima probabilidades | Sim | Sim | Sim | Sim |
| Normalização necessária? | Não | Altamente recomendável | Geralmente não | Geralmente não |
| Pode sofrer *overfitting* | Não aprende padrões | Sim | Sim | Sim |
| Interpretabilidade | Muito simples | Boa | Moderada | Moderada |
| Complexidade da configuração | Muito baixa | Baixa a moderada | Moderada | Moderada a alta |

**Diferenças fundamentais:**

- `DummyClassifier` não aprende relações entre as **features** e as **targets**.

- `LogisticRegression` aprende uma relação linear entre as **características** e a **classificação**.

- `RandomForest` aprende padrões através de várias árvores construídas independentemente.

- `LightGBM` constrói árvores sequencialmente, buscando **corrigir os erros do conjunto anterior**.

**Complexidade dos modelos:**

1. `LightGBM`: combina árvores sequenciais e exige atenção especial à otimização dos hiperparâmetros.

2. `RandomForest`: combina múltiplas árvores de decisão.

3. `LogisticRegression`: aprende pesos e relações lineares.

4. `DummyClassifier`: não aprende padrões a partir de features.

### 6. Aplicação no SAML-D

| Modelo | Aplicação no sistema | O que esperamos dele? |
| :--: | :--: | :-- |
| DummyClassifier | Referência de desempenho | Demonstrar o quanto um modelo que ignora as características consegue acertar. |
| LogisticRegression | Classificação inicial | Identificar relações lineares entre as características e a ocorrência de lavagem de dinheiro |
| RandomForest | Identificação de padrões | Identificar combinações de características associadas a operações suspeitas |
| LightGBM | Classificação avançada | Aprender interações e padrões complexos, buscando melhorar a identificação de operações suspeitas |
