# Linha de base com quatro agentes

Este ensaio aumenta o cenario de dois para quatro agentes sem introduzir
os mecanismos que serao avaliados como contribuicao da pesquisa. Portanto,
ele deve ser interpretado como uma **linha de base local e segura**.

## O que permanece no controlador

- um MPC/PANOC local e independente por agente;
- modelo cinematico diferencial;
- um ponto final fixo por agente;
- campo vetorial para indicar o fluxo permitido na via;
- geofence suave para manter cada agente na regiao de sua rota;
- CBF como penalidade suave para evitar colisoes;
- previsoes dos vizinhos congeladas durante cada solucao local.

## O que deliberadamente nao foi incluido

- prioridade de passagem;
- custo economico ou custo de fluxo;
- formacao de coalizoes;
- ADMM ou outro metodo de otimizacao cooperativa;
- controlador centralizado;
- waypoints ou seguimento de caminho.

Essa separacao e importante: a CBF deve ser avaliada como mecanismo de
seguranca, enquanto as coalizoes e o custo de fluxo serao posteriormente
avaliados pelo ganho de desempenho global. Um bloqueio seguro neste ensaio
e um resultado da linha de base, e nao deve ser escondido por uma regra de
prioridade introduzida antecipadamente.

## Configuracao

Os quatro agentes entram em duas vias horizontais, com sentido para a
direita, e em duas vias verticais, com sentido para cima:

```text
Agente 1: (-1.30,-0.50) -> ( 1.30,-0.50)
Agente 2: (-0.80,-0.90) -> (-0.80, 0.90)
Agente 3: ( 0.00,-0.90) -> ( 0.00, 0.90)
Agente 4: (-1.30, 0.50) -> ( 1.30, 0.50)
```

Isso produz conflitos no cruzamento inferior esquerdo, na rotatoria
inferior, na rotatoria superior esquerda e no cruzamento superior central.
Cada agente considera os outros tres como vizinhos. Existem seis pares de
distancia registrados: `A1-A2`, `A1-A3`, `A1-A4`, `A2-A3`, `A2-A4` e
`A3-A4`.

Foram usados:

```text
Ts                         = 0.033 s
horizonte                  = 30 passos
iteracoes maximas PANOC    = 5
distancia fisica requerida = 0.200 m
distancia interna da DCBF  = 0.270 m
rho_DCBF                   = 2.0e8
```

A distancia interna maior que a distancia fisica e uma margem de robustez
para a penalidade suave e para a diferenca temporaria entre as previsoes
locais comunicadas e os comandos efetivamente escolhidos pelos agentes.
Isso nao introduz prioridade nem cooperacao. Os mesmos parametros de
seguranca deverao ser mantidos quando a estrategia coalizional for
comparada a esta linha de base.

## Como executar

Abra a pasta descompactada no MATLAB e execute:

```matlab
results4 = main_quatro_agentes_dcbf(true, 60, 5);
```

Na simulacao local, essa funcao pula a movimentacao preliminar do
Robotarium e coloca os agentes diretamente nas condicoes iniciais do
experimento. Isso evita esperas e bloqueios aleatorios antes de o MPC
comecar. Para reproduzir tambem essa etapa preliminar, use:

```matlab
results4 = main_quatro_agentes_dcbf(true, 60, 5, 0.27, 2.0e8, false);
```

O Robotarium fisico e seu verificador sempre executam a inicializacao real;
essa opcao nao modifica o controlador nem os resultados a partir das
condicoes iniciais definidas.

Na primeira execucao, os quatro campos sao gerados e salvos em
`campos_quatro_agentes.mat`. As execucoes seguintes reutilizam esse cache.

## Resultado validado

O ensaio completo validado no macOS esta salvo em
`resultado_4ag_codegen_mac.mat`.

```text
Missao concluida            : [1 1 1 1]
Tempos de chegada [s]       : [34.419 22.506 36.432 52.107]
Erros finais [m]            : [0.0590 0.0593 0.0596 0.0589]
Distancia minima global [m] : 0.2465
Distancia minima A1-A2 [m]  : 0.2681
Distancia minima A1-A3 [m]  : 0.7768
Distancia minima A1-A4 [m]  : 0.7626
Distancia minima A2-A3 [m]  : 0.5908
Distancia minima A2-A4 [m]  : 0.3284
Distancia minima A3-A4 [m]  : 0.2465
Possivel bloqueio seguro    : nao
```

O `debug()` do simulador terminou sem erros ou avisos e nao houve violacao
medida da geofence nesse ensaio.

Uma tentativa anterior com os parametros do caso de dois agentes
(`d_safe = 0.215 m` e `rho_DCBF = 2.0e7`) foi deliberadamente descartada:
embora todos chegassem, o par A3-A4 atingiu `0.1555 m` e o Robotarium
registrou 24 instantes de possivel colisao. Aumentar somente o numero de
iteracoes do PANOC nao resolveu essa perda de robustez. Por isso a margem e
o peso da DCBF foram ajustados antes de aceitar a linha de base.

## Resultados a verificar

```matlab
results4.completed
results4.completion_times
results4.final_errors
results4.pair_indices
results4.minimum_distance_per_pair
results4.minimum_distance
results4.required_distance
results4.max_measured_dcbf_violation
results4.max_geofence_violation
results4.solver_times
```

`results4.completed` informa quais agentes chegaram. A distancia minima
global deve ser comparada a `results4.required_distance`, enquanto
`minimum_distance_per_pair` permite localizar qual interacao foi mais
critica. Se nem todos chegarem, as trajetorias e os comandos devem ser
analisados para distinguir falta de tempo de um possivel deadlock seguro.

## Custo computacional observado

Na simulacao validada, considerando apenas o tempo medido dentro das quatro
chamadas sequenciais ao PANOC:

```text
media por PANOC                    = 6.854 ms
media da soma dos PANOCs por ciclo = 27.417 ms
percentil 95 da soma por ciclo     = 77.585 ms
ciclos acima de Ts                 = 660 de 1580
```

Portanto, esta versao e adequada para o estudo local da linha de base, mas
ainda nao deve ser enviada ao Robotarium fisico com quatro agentes. Antes
disso sera necessario reduzir o custo, compilar as funcoes ou executar os
controladores locais em paralelo, sem alterar o funcional usado na
comparacao cientifica.

## Escalonamento posterior

O Robotarium admite experimentos com mais agentes, mas o aumento deve ser
gradual. A sequencia recomendada e validar 4, depois 6, 8 e somente entao
configuracoes maiores. Com `N_r` agentes, o numero de pares DCBF e
`N_r(N_r-1)/2`; para 18 agentes seriam 153 pares. Alem disso, cada um dos
18 problemas locais receberia 17 previsoes de vizinhos. Portanto, aceitar
18 robos na plataforma nao implica que esta implementacao interpretada em
MATLAB satisfaca automaticamente o periodo de amostragem com 18 agentes.
