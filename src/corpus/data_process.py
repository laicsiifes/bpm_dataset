import pandas as pd
import random


class Entity:
    def __init__(self, id: int, label: str, start_offset: int, end_offset: int):
        """
        Essa classe representa uma entidade
        :param id: identificador da entidade
        :param label: label do NER - Named Entity Recognition.
        :param start_offset: caracter que a entidade começa
        :param end_offset: caracter que a entidade termina
        """
        self.id = id
        self.label = label
        self.start_offset = start_offset
        self.end_offset = end_offset


class Relation:
    def __init__(self, id: int, src: int, target: int, type_rel: str):
        """
        Essa classe representa uma relação

        :param id: identificador da relação
        :param src: entidade de origem (>>de quem<< para quem)
        :param target: entidade de destino (de quem para >>quem<<)
        :param type_rel: tipo de relação
        """
        self.id = id
        self.src = src
        self.target = target
        self.type = type_rel


class ExampleRelation:
    def __init__(self, id: int, text: str, relation: str, entity_src: int, entity_trg: int):
        self.id = id
        self.text = text
        self.relation = relation
        self.entity_src = entity_src
        self.entity_trg = entity_trg


def get_negatives_rel_ex(
        entities_with_rel: list[tuple],
        entities: list[dict],
        text: str,
        text_id: int,
        quantity: int
) -> list[ExampleRelation]:
    """
    Essa função gera 4 exemplos negativos para cada frase do Dataset, ou seja, frases onde a relação é "O" (sem relação).
    :param entities_with_rel:  Lista com tuplas contendo (id origem, id destino).
    :param entities: Lista com todas as entidades do texto.
    :param text: Texto do Dataset.
    :param text_id: ID do texto.
    :param quantity: Quantidade de exemplos negativos que eu devo gerar
    :return: Frase demarcada com os as entidades que não possuem relação.
    """
    # ---------------------------------------------------------------------
    #                       Inicializando variáveis
    # ---------------------------------------------------------------------

    text_split = list(text)

    demarcated_text = text_split.copy()

    examples = []

    # ---------------------------------------------------------------------

    # Criando mapa ID -> Entidade
    id2entity = {ent['id']: ent for ent in entities}

    random.seed(1)      # Garantindo reprodutibilidade com seed = 1

    while len(examples) < quantity:
        random_ids = random.sample(sorted(id2entity.keys()), k=2)
        random_ids_inv = (random_ids[1], random_ids[0])

        if random_ids not in entities_with_rel and random_ids_inv not in entities_with_rel:

            entity_src = Entity(
                id=random_ids[0],
                label=id2entity[random_ids[0]]['label'],
                start_offset=id2entity[random_ids[0]]['start_offset'],
                end_offset=id2entity[random_ids[0]]['end_offset']
            )

            entity_trg = Entity(
                id=random_ids[1],
                label=id2entity[random_ids[1]]['label'],
                start_offset=id2entity[random_ids[1]]['start_offset'],
                end_offset=id2entity[random_ids[1]]['end_offset']
            )

            if entity_src.id < entity_trg.id:  # Origem antes do alvo

                demarcated_text.insert(entity_src.start_offset, '[E1]')
                demarcated_text.insert(entity_src.end_offset + 1, '[/E1]')

                demarcated_text.insert(entity_trg.start_offset + 2, '[E2]')
                demarcated_text.insert(entity_trg.end_offset + 3, '[/E2]')

            else:  # Origem depois do alvo

                demarcated_text.insert(entity_trg.start_offset, '[E2]')
                demarcated_text.insert(entity_trg.end_offset + 1, '[/E2]')

                demarcated_text.insert(entity_src.start_offset + 2, '[E1]')
                demarcated_text.insert(entity_src.end_offset + 3, '[/E1]')

            completed_text = ''.join(demarcated_text)  # Texto demarcado final

            example = ExampleRelation(
                id=text_id,
                text=completed_text,
                relation='O',
                entity_src=entity_src.id,
                entity_trg=entity_trg.id
            )

            examples.append(example)

            demarcated_text = text_split.copy()     # Resetando texto

    return examples


def get_positives_rel_ex(
        relations: list[dict],
        entities: list[dict],
        text: str,
        text_id: int
) -> (list[ExampleRelation], list[tuple[str, str]]):
    """
    Essa função gera os exemplos positivos (com base nas anotações do Dataset).
    :param relations: Lista com dicionários contendo as chaves: id, from_id, to_id, type.
    :param entities: Lista com dicionários contendo as chaves: id, label, start_offset, end_offset.
    :param text: Texto do dataset
    :param text_id: ID do texto
    :return: Frase demarcada => "Essa [E1]palavra_01[/E1] se relaciona com essa [E2]palavra_02[/E2]" e lista com todas as entidades (tokens) que estão relacionadas"
    """
    # ------------------------------------------------------------------
    #                       Inicializando variáveis
    # ------------------------------------------------------------------

    text_split = list(text)  # Transformando a frase em uma lista de caracteres

    ent_src: int = 0  # Vai armazenar o ID da entidade de origem
    ent_trg: int = 0  # Vai armazenar o ID da entidade de destino

    examples = []  # Lista com objetos ExampleRelation()

    move_origin: bool = False
    move_destiny: bool = False

    markup_e1: bool = False
    markup_e2: bool = False

    demarcated_text = text_split.copy()

    entities_relation = []  # Armazenando pares de entidades que possuem relações

    # ------------------------------------------------------------------
    #                         Demarcando o texto
    # ------------------------------------------------------------------

    for dict_rel in relations:

        relation = Relation(
            id=dict_rel['id'],
            src=dict_rel['from_id'],
            target=dict_rel['to_id'],
            type_rel=dict_rel['type']
        )

        for dict_ent in entities:

            # Quando formos adicionar as marcações no texto, vamos adicionar o bloco [E1] ou [/E1] de uma só vez.
            #   [E1] valerá como 1 caracter de deslocamento na lista
            #   [/E1] valerá como 1 caracter de deslocamento na lista
            # O mesmo vale para as marcações de [E2][/E2]

            entity = Entity(
                id=dict_ent['id'],
                label=dict_ent['label'],
                start_offset=dict_ent['start_offset'],
                end_offset=dict_ent['end_offset']
            )

            if entity.id == relation.src:
                ent_src = entity.id      # Guardando o ID da entidade de origem

                if move_origin:
                    demarcated_text.insert(entity.start_offset + 2, '[E1]')
                    demarcated_text.insert(entity.end_offset + 3, '[/E1]')
                    markup_e1 = True
                else:
                    demarcated_text.insert(entity.start_offset, '[E1]')
                    demarcated_text.insert(entity.end_offset + 1, '[/E1]')
                    markup_e1 = True

                    if relation.src < relation.target:
                        move_destiny = True

            elif entity.id == relation.target:
                ent_trg = entity.id      # Guardando o ID da entidade de destino

                if move_destiny:
                    demarcated_text.insert(entity.start_offset + 2, '[E2]')
                    demarcated_text.insert(entity.end_offset + 3, '[/E2]')
                    markup_e2 = True
                else:
                    demarcated_text.insert(entity.start_offset, '[E2]')
                    demarcated_text.insert(entity.end_offset + 1, '[/E2]')
                    markup_e2 = True

                    if relation.target < relation.src:
                        move_origin = True
            else:
                continue

            if markup_e1 and markup_e2:
                completed_text = ''.join(demarcated_text)  # Texto finalizado

                example = ExampleRelation(
                    id=text_id,
                    text=completed_text,
                    relation=relation.type,
                    entity_src=ent_src,
                    entity_trg=ent_trg
                )

                entities_relation.append((example.entity_src, example.entity_trg))
                examples.append(example)

                # -----------------------------------------------------------------------
                #                           Resetando variáveis
                # -----------------------------------------------------------------------

                demarcated_text = text_split.copy()  # Resetando o texto

                move_origin = False
                move_destiny = False

                markup_e1 = False
                markup_e2 = False

                break

    return examples, entities_relation


def get_relation_example(relations: list[dict], entities: list[dict], text: str, text_id: int) -> list[ExampleRelation]:
    """
    Essa função gera um texto demarcado por [E1][/E1] e [E2][/E2].
    :param relations: Lista com dicionários contendo as chaves: id, from_id, to_id, type.
    :param entities: Lista com dicionários contendo as chaves: id, label, start_offset, end_offset.
    :param text: Texto do dataset
    :param text_id: ID do texto
    :return: Frase demarcada => "Essa [E1]palavra_01[/E1] se relaciona com essa [E2]palavra_02[/E2]"
    """

    examples = []

    positives, entities_with_relation = get_positives_rel_ex(relations, entities, text, text_id)

    # Gerando k vezes mais exemplos negativos do que positivos

    k = 4

    qnt_negatives_ex = len(positives) * k

    negatives = get_negatives_rel_ex(entities_with_relation, entities, text, text_id,
                                     qnt_negatives_ex)

    examples.extend(positives)
    examples.extend(negatives)

    return examples


def get_train_data(dataframe: pd.DataFrame) -> list[ExampleRelation]:
    """
    Essa função recebe um DataFrame e retorna os dados de input de treino e validação.
    Basicamente é uma função centralizadora que utiliza outras funções para processar os dados.

    :param dataframe: DataFrame da biblioteca pandas
    :return: Dados para o algoritmo de aprendizado
    """

    all_examples = []

    for i, row in dataframe.iterrows():

        id_ = row['id']                      # ID do texto (será usado para gerar os diretórios)

        text = row['text']                  # [str, str, str, ...]

        entities = row['entities']          # [dict, dict, dict, ...]

        relations = row['relations']        # [dict, dict, dict, ...]

        examples = get_relation_example(relations, entities, text, id_)      # Lista de objetos ExampleRelation

        all_examples.extend(examples)

    return all_examples
