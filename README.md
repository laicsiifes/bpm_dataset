Guia para Anotação de Textos Descritivos de Processos de Negócio

1. Anotação de Atores

Definição: Atores são entidades que realizam ações.

Diretriz: Não anotar nomes convenientes de atores que aparecem como complementos de atividades.

Exemplo: Em "Consult the cost center managers", "cost center managers" é um complemento da atividade "consult" e não deve ser anotado como ator.

2. Anotação de Atividades

Definição: Atividades são ações realizadas por atores.

Diretriz: Identificar e separar atividades compostas.

Exemplo: Em "Customer's profile is created, or selected", "create" e "select" são atividades diferentes e devem ser anotadas separadamente.

3. Anotação de Triggers e Catches

Definição: Triggers marcam onde uma notificação é criada por um ator; Catches indicam quando uma ação ou condição é notada ou registrada por um ator.

Diretriz: Use "catch" para marcar o objeto relacionado a um evento, transformando-o no sujeito que realiza a captura.

Exemplo: Em "a confirmation e-mail is sent to the customer", o "customer" deve ser anotado como "catch" da ação de captura do e-mail.

4. Anotação de Condições

Definição: Condições são estados ou requisitos que influenciam a realização de atividades.

Diretriz: Separar condições de atividades mesmo quando estão integradas na mesma sentença.

Exemplo: Em "write at least three status updates every week", "every week" é uma condição que deve ser anotada separadamente da atividade "write status updates".

5.  Complementos de Atividades

Definição: Nomes Atividades podem ser seguidos de detalhes adicionais.

Diretriz: Anotar a atividade mínima necessária. Evitar detalhes complementares que não são essenciais para caracterizar a atividade principal.

Exemplo: Em "find out if you have all the games your friends want to play", anotar a atividade como "find out if you have all the games", deixando o complemento "your friends want to play" de fora.

6. Dependências e Representação de Eventos

Definição: Dependências são relações entre atividades; eventos são condições ou notificações externas que afetam o processo.

Diretriz: Usar "condition" para estados e "catch" para eventos externos.

Exemplo: "If one shop has not enough parts" deve ser anotado como uma "condition", enquanto "When the first parts arrive" deve ser anotado como um "catch".

7. Atividade vs. Trigger

Definição: Diferenciar entre atividades que envolvem trabalho e triggers que marcam onde uma notificação é criada por um ator.

Diretriz: Se a ação é uma resposta direta a uma condição externa, anotá-la como trigger.

Exemplo: "When the machine is disconnected stop the process" deve ser anotado com "stop the process" como trigger.

8. Captura de Eventos

Definição: Eventos indicam situações onde o processo espera ou é ativado por notificações externas.

Diretriz: Anotar situações de espera ou disponibilidade como eventos interruptivos.

Exemplo: "I am available for clarifications" deve ser anotado como um evento onde o ator está aguardando.

9. Sentenças Omitidas

Definição: Partes de atividades ou eventos podem ser omitidas em frases.

Diretriz: Completar a anotação com a parte omitida implícita.

Exemplo: Em "keep this registration with him... and in his close proximity", a segunda atividade deve ser anotada como "keep this registration in his close proximity".

10. Meta-informações sobre o Processo

Definição: Informações explicativas que não são diretamente parte do processo.

Diretriz: Anotar meta-informações para posterior avaliação no contexto do processo.

Exemplo: "An employee purchases a product. For example, a sales person on a trip rents a car" deve ser anotado considerando a explicação como meta-informação.

11. Nomes de Atores em Outras Entidades

Definição: Nomes de atores que aparecem dentro de outras entidades.

Diretriz: Anotar o nome completo incluindo o nome do ator quando ele aparece novamente.

Exemplo: "it goes to the treasurer" deve ser anotado como um trigger completo "it goes to the treasurer", e "The treasurer checks that all the receipts have been submitted" deve ter "treasurer" anotado como ator.

12. Ambiguidade entre Triggers e Catches

Definição: Diferença entre triggers que marcam onde uma notificação é criada por um ator e catches que registram um evento.

Diretriz: Considerar um trigger quando o evento é uma ação de quem o lança e está semanticamente mais próximo deste. A ambiguidade deve ser resolvida baseando-se na proximidade semântica e na continuidade do processo.

Exemplo: Em "it goes to the treasurer", pode-se questionar se é um trigger para o supervisor ou um catch para o treasurer. Convenciona-se considerar um trigger quando o evento é descrito como uma ação de quem o lança e está semanticamente mais próximo deste. Neste caso, o evento está condicionado a uma condição relacionada ao supervisor e não há um próximo passo do treasurer explicitamente associado ao trigger.

Anotar descrições de processos de negócios é uma tarefa desafiadora devido à complexidade e à ambiguidade inerentes às linguagens naturais. Uma das dificuldades mais comuns é diferenciar entre atividades e condições quando elas estão integradas na mesma sentença. Por exemplo, em "write at least three status updates every week", a condição "every week" está embutida na atividade "write status updates", o que pode dificultar a separação clara e a anotação adequada de cada componente. Essa integração pode levar a interpretações equivocadas ou à perda de detalhes importantes que são essenciais para o entendimento completo do processo.

Outra dificuldade significativa é a diferenciação entre triggers e catches. Triggers são destinados a marcar onde uma notificação é criada por um ator, enquanto catches indicam quando uma ação ou condição é notada ou registrada. Em sentenças como "When the first parts arrive", decidir se deve ser anotado como um trigger ou um catch pode ser complicado. Se anotado como um catch, implica que o processo está aguardando a chegada das peças; se anotado como um trigger, sugere que a chegada das peças inicia uma nova ação. A ambiguidade nesta escolha pode afetar a precisão do modelo de anotação e, consequentemente, a qualidade do aprendizado de padrões pelos algoritmos.

Adicionalmente, a anotação de informações implícitas representa outra dificuldade. Em frases como "keep this registration with him .... and in his close proximity", a segunda atividade "keep this registration in his close proximity" tem a parte essencial da sentença omitida. Anotar adequadamente esses casos exige a inserção da informação implícita, o que demanda um entendimento profundo do contexto para garantir que a anotação capture a totalidade da ação descrita. Esses desafios exemplificam a complexidade de capturar fielmente as nuances dos processos de negócios em anotações estruturadas, destacando a necessidade de diretrizes claras e detalhadas para orientar o trabalho de anotação.



