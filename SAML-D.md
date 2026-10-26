# Anti Money Laudering Transaction Dataset (SAML-D)
Link: https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/data

### Estrutura
O dataset SAML-D possui aproximadamente **9,5 milhões de transações**, **12 colunas** e cerca de **9.873 registros positivos de lavagem de dinheiro** (~0,104% do conjunto).

| Coluna | O que representa | O que estudar |
| :--- | :---: | ---: |
| Time | Horário da transação | Conversão para horário e análise temporal |
| Date | Data da transação | Datas, períodos e ordenação |
| Sender_account | Conta remetente | Recorrência e comportamento de envio |
| Receiver_account | Conta destinatária | Recorrência e comportamento de recebimento |
| Amount | Valor da transação | Distribuição, assimetria e valores extremos |
| Payment_currency | Moeda do pagamento | Variáveis categóricas |
| Received_currency	| Moeda recebida | Conversões e diferenças de moeda |
|Sender_bank_location | Localização bancária do remetente	| Comparações por localização |
| Receiver_bank_location | Localização bancária do destinatário | Operações entre localizações |
| Payment_type | Tipo de pagamento | Frequências e padrões |
| Is_laundering | Indicador de lavagem de dinheiro | Variável-alvo para classificação |
| Laundering_type | Tipo de comportamento identificado | Análise das categorias, com cuidado para não vazar o alvo |  


