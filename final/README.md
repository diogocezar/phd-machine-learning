# Especificação Tema Comum

O trabalho consiste em classificar cliente inadimplentes utilizando algoritmos de aprendizagem de máquina.
A base de dados contem 672 atributos e dados de clientes que contraíram empréstimos junto a instituição financeira. O que realizaram o pagamento em até 30 dias possuem valor zero na variável Y. Caso contrário, Y=1. A base contém 219984 registros.

## Protocolo Experimental

A base deve ser dividida em 50%, 20% e 30% para treinamento, validação e teste, respectivamente.
Note que a base é desbalanceada. Nesse caso, reporte métricas como Precision, Recall, F_Score e curva ROC.
O que deve ser entregue:

- Artigo reportando seus resultados
- Estrutura do artigo
  - Breve introdução
  - Metodologia empregada
  - Experimentos realizados
  - Conclusão
- Apresentação (slides em PDF) para uma apresentação de 10 minutos.

Os trabalhos podem ser realizados em grupo de no máximo DUAS pessoas.

Critérios de correção:

Qualidade da escrita do artigo
Qualidade e complexidade da solução adotada
Resultados alcançados na base de teste
Data de entrega:

A data de entrega assim como o cronograma das apresentações serão definidos em breve.

---

Dúvidas:

- MinMaxScaler does not support sparse input. Consider using MaxAbsScaler instead.
-

TODO:

Params no JSON;
Salvar Quantidades de Classes Nos Splits;
