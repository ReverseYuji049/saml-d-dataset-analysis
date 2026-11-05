
1. Balanceamento das classes

O SAML-D apresenta um desbalanceamento muito elevado. Aproximadamente 99,8961% das transações pertencem à classe 0 (não lavagem), enquanto apenas 0,1039% pertencem à classe 1 (lavagem de dinheiro). Isso representa um desafio porque o modelo pode alcançar uma acurácia muito alta simplesmente prevendo a classe majoritária. Por isso, estamos priorizando métricas como Precision, Recall, F1-score e PR-AUC.

2. Quantidade de variáveis

O dataset original possui 12 colunas. Após o processamento, nossa documentação registra uma representação com 10 dimensões. Precisamos distinguir as features utilizadas como entrada da variável-alvo (Is_laundering), que é aquilo que o modelo deve prever.

3. Redução de dimensionalidade com PCA

Realizamos experimentos com PCA para avaliar a redução de dimensionalidade e seu impacto na Logistic Regression. Nos resultados documentados, as configurações com PCA apresentaram F1-score e PR-AUC inferiores às configurações sem PCA. Portanto, a redução não trouxe melhoria nesse experimento. Também precisamos confirmar no notebook a quantidade de features antes e depois do PCA, pois a documentação registra PCA14 e uma representação processada com 10 dimensões.
